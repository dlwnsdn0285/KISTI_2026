"""Single-query pipeline: seed search -> base pool -> expansion -> judging -> output.

Base pool: the seeds plus their references. Expansion adds neighbours of a frontier drawn
from the base pool, up to a budget of 0.4 x base pool.
The result is one trace dict; candidates_of turns it into ranked candidate rows.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import time
from dataclasses import asdict, dataclass

import numpy as np

from psa.expand import (DIRECTIONS, Bandit, design, frontier, harvest, load_prior,
                        vector)
from psa.judge import judge_shortlist, select_output
from psa.store import arxiv_of_doi


@dataclass
class Config:
    """Fixed settings for expansion and output."""
    budget_share: float = 0.4          # expansion budget = base pool size x this
    frontier_top: int = 200            # frontier: top by cosine
    frontier_sample: int = 100         # frontier: random from the rest
    hub_max_citations: int = 1000      # skip nodes cited more than this
    cap: int = 500                     # max neighbours per node and direction
    window_years: int = 15             # neighbours within N years before the cutoff
    cited_by_floor: int = 5            # min citations at cutoff for cited_by neighbours
    harvest_batch: int = 60            # nodes per store query
    reward_quantile: float = 0.9       # reward / judging threshold: quantile of base-pool cosine
    intercept_precision: float = 0.01
    link_quantile: float = 0.75
    rating_min: float = 8.0
    seed: int = 0
    intercept_rounds: int = 10
    prior_path: str = ""

    @classmethod
    def for_backend(cls, backend, prior_path):
        cfg = cls(prior_path=prior_path)
        if backend == "datalake":
            # datalake has sparser citations, and one large query is faster there
            cfg.cited_by_floor = 1
            cfg.harvest_batch = 300
        return cfg


def _stage(meter, name):
    return meter.stage(name) if meter is not None else contextlib.nullcontext()


def build_pool(seeds, cutoff, store, blocklist):
    """Base pool: the seeds plus their references (one hop)."""
    use_cid = store.name == "s2"
    kept, origin, owner, drops, seen = [], {}, {}, {}, set()

    def admit(keys, papers, parent_of):
        for key in keys:
            paper = papers[key]
            if blocklist.hit(paper, use_corpusid=use_cid):
                reason = "blocked"
            elif key in seen:
                reason = "duplicate"
            elif store.after_cutoff(paper, cutoff):
                reason = "after_cutoff"
            else:
                reason = None
            if reason:
                drops[reason] = drops.get(reason, 0) + 1
                continue
            seen.add(key)
            kept.append(key)
            if parent_of is not None:
                owner[key] = parent_of[key]

    papers = store.papers(seeds)
    admit(seeds, papers, None)
    seed_keys = list(kept)
    if not seed_keys:
        return kept, set(), owner, drops, 0
    hop = store.neighbours(seed_keys, "references", cap=None, cutoff=cutoff)
    fresh, parent_of = [], {}
    for parent in seed_keys:
        for child in hop.edges.get(parent) or []:
            if child not in parent_of:
                parent_of[child] = parent
                fresh.append(child)
    papers = store.papers(fresh)
    admit(fresh, papers, parent_of)
    got = store.papers(kept)
    not_in_store = sum(1 for k in kept if not got[k]["title"] and got[k]["year"] is None)
    return kept, set(seed_keys), owner, drops, not_in_store


def prepare(title, qid, *, cutoff, store, searcher, relevance, blocklist, cfg: Config,
            meter=None, log=print):
    """Everything before expansion: seed search, base pool, relevance, reward threshold.

    Returns (state, trace).
    """
    timings = {}
    trace = {"qid": qid, "question": title,
             "cutoff": cutoff.isoformat() if cutoff else None, "config": asdict(cfg),
             "backend": store.name, "searcher": searcher.name}

    # seed search and base pool
    t0 = time.perf_counter()
    with _stage(meter, "search"):
        seeds = searcher.search(title, cutoff=cutoff)
    timings["search"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    with _stage(meter, "expand"):
        kept, seed_keys, owner, drops, not_in_store = build_pool(seeds, cutoff, store,
                                                                 blocklist)
    timings["pool"] = time.perf_counter() - t0
    pool = set(kept)
    trace["search"] = {"counts": searcher.last_counts, "queries": searcher.last_queries,
                       "llm": searcher.last_llm, "seeds": len(seed_keys)}
    trace["pool"] = {"size": len(pool), "dropped": drops, "not_in_store": not_in_store}
    if searcher.last_counts.get("api_fail"):
        log(f"  ⚠ {qid}: 검색 호출 실패 {searcher.last_counts['api_fail']}건 — "
            "시드가 줄었을 수 있다")
    if not pool:
        trace["error"] = "기준 풀이 비었다"
        return None, trace

    with _stage(meter, "expand"):
        # relevance, reward threshold, frontier
        t0 = time.perf_counter()
        papers = store.papers(sorted(pool))
        cos = relevance.cosine(list(papers.values()))
        threshold = float(np.quantile(list(cos.values()), cfg.reward_quantile))
        ranked = sorted((k for k in pool if k not in seed_keys), key=lambda k: (-cos[k], k))
        nodes, hubs = [], 0
        for rank, key in enumerate(ranked, 1):
            count = papers[key]["citationcount"]
            if count is not None and count > cfg.hub_max_citations:
                hubs += 1
                continue
            nodes.append({"corpusid": key, "rank": rank})
        chosen, n_top, n_sampled = frontier(nodes, cfg.frontier_top, cfg.frontier_sample,
                                            cfg.seed)
        ids = [n["corpusid"] for n in chosen]
        timings["embed_pool"] = time.perf_counter() - t0

        # neighbours per (node, direction)
        t0 = time.perf_counter()
        block = blocklist.keys_for(store.name)
        sets = {d: harvest(ids, d, cutoff, store, block, cap=cfg.cap,
                           window_years=cfg.window_years, floor=cfg.cited_by_floor,
                           batch=cfg.harvest_batch) for d in DIRECTIONS}
        indeg = store.indegree_before(ids, cutoff)
        reachable = {c for d in DIRECTIONS for kids in sets[d].values() for c in kids}
        store.prefetch(sorted(reachable - pool))
        timings["harvest"] = time.perf_counter() - t0
    trace["frontier"] = {"nodes_available": len(nodes), "hubs_excluded": hubs,
                         "harvest_order": getattr(store, "harvest_order",
                                                  "influential_first"),
                         "cited_by_floor": cfg.cited_by_floor, "top": n_top,
                         "sampled": n_sampled,
                         "harvested": {d: sum(len(v) for v in sets[d].values())
                                       for d in DIRECTIONS}}
    trace["reward"] = {"quantile": cfg.reward_quantile, "threshold": threshold}
    state = {"pool": pool, "seeds": seed_keys, "owner": owner, "papers": papers,
             "cos": cos, "threshold": threshold, "ids": ids, "sets": sets,
             "indeg": indeg, "timings": timings}
    return state, trace


def arm_row(paper, cutoff, indeg, cosine, sets, key):
    """Feature inputs for one frontier node."""
    year = paper["year"]
    return {"age": (year - cutoff.year) if (year and cutoff) else None,
            "citationcount": paper["citationcount"], "indegree_before": indeg.get(key, 0),
            "embed_cosine": cosine, "expected_pairs": {d: len(sets[d][key])
                                                       for d in DIRECTIONS}}


def _judge_and_select(trace, title, everything, all_cos, *, judge, store, cutoff, threshold,
                      cfg, meter, timings):
    """Judge the candidates and apply the output rule."""
    t0 = time.perf_counter()
    with _stage(meter, "judge"):
        judged = judge_shortlist(title, everything, all_cos, judge, threshold=threshold)
        timings["judge"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        links = store.links([p["corpusid"] for p in everything], cutoff)
        timings["links"] = time.perf_counter() - t0
        output, k_links = select_output(judged, links, link_quantile=cfg.link_quantile,
                                        rating_min=cfg.rating_min)
    trace["output_rule"] = {"link_quantile": cfg.link_quantile, "links_threshold": k_links,
                            "rating_min": cfg.rating_min}
    trace["judging"] = {"candidates": len(everything), "judged": len(judged),
                        "output": len(output), "judge_usage_last": judge.last_usage,
                        "true_not_in_top": judge.true_not_in_top}
    trace["judged"] = judged
    trace["output"] = output


def run_query(title, qid, *, cutoff, store, searcher, judge, relevance, blocklist,
              cfg: Config, meter=None, log=print):
    """Run one query end to end and return its trace dict."""
    t_start = time.perf_counter()
    cutoff = dt.date.fromisoformat(cutoff) if isinstance(cutoff, str) else cutoff
    state, trace = prepare(title, qid, cutoff=cutoff, store=store, searcher=searcher,
                           relevance=relevance, blocklist=blocklist, cfg=cfg,
                           meter=meter, log=log)
    trace["judge"] = judge.name
    if state is None:
        return trace
    pool, seed_keys, owner = state["pool"], state["seeds"], state["owner"]
    papers, cos, threshold = state["papers"], state["cos"], state["threshold"]
    ids, sets, indeg, timings = state["ids"], state["sets"], state["indeg"], state["timings"]

    expand_stage = _stage(meter, "expand")
    expand_stage.__enter__()
    w, precision, scale, prior, columns = load_prior(cfg.prior_path, cfg.intercept_precision)
    x = {}
    for key in ids:
        row = arm_row(papers[key], cutoff, indeg, cos[key], sets, key)
        for d in DIRECTIONS:
            if row["expected_pairs"][d] > 0:
                raw = [vector(row, d)[i] for i in columns]
                x[(key, d)] = np.array(design(raw, scale))
    budget = cfg.budget_share * len(pool)
    trace["frontier"]["arms"] = len(x)
    trace["budget"] = budget
    trace["prior"] = {"path": cfg.prior_path, "features": prior["features"],
                      "fitted_on": prior.get("fitted_on"), "w0": w.tolist()}

    # bandit expansion
    t0 = time.perf_counter()
    bandit = Bandit(x.keys(), x, w, precision, np.random.default_rng(cfg.seed))
    use_cid = store.name == "s2"
    collected, spent, rounds, intercepts, via = {}, 0, [], [], {}
    blocked_late = 0
    while bandit.arms and spent < budget:
        key, d = bandit.pick()
        fresh = [c for c in sets[d][key] if c not in pool and c not in collected]
        got = store.papers(fresh)
        survivors = []
        for c in fresh:
            # the harvest filtered by key only; filter again by DOI / arXiv / title
            if blocklist.hit(got[c], use_corpusid=use_cid):
                blocked_late += 1
                continue
            survivors.append(got[c])
        rel = relevance.cosine(survivors)
        top = sum(1 for p in survivors if rel[p["corpusid"]] >= threshold)
        for p in survivors:
            collected[p["corpusid"]] = p
            owner[p["corpusid"]] = key
            via[p["corpusid"]] = d
        spent += len(fresh)
        bandit.observe((key, d), top, len(fresh))
        rounds.append([key, d, len(fresh), top])
        if len(intercepts) < cfg.intercept_rounds:
            intercepts.append(round(float(bandit.w[0]), 4))
    timings["walk"] = time.perf_counter() - t0
    expand_stage.__exit__(None, None, None)
    trace["walk"] = {"rounds": len(rounds), "spent": spent,
                     "overflow": max(0.0, spent - budget), "collected": len(collected),
                     "blocked_late": blocked_late, "top_total": sum(r[3] for r in rounds),
                     "spent_by_direction": {d: sum(r[2] for r in rounds if r[1] == d)
                                            for d in DIRECTIONS},
                     "intercept_first_rounds": intercepts, "w_final": bandit.w.tolist(),
                     "trace_cols": ["node", "direction", "pairs", "top"],
                     "trace": rounds}

    # judging and output
    everything = list(papers.values()) + list(collected.values())
    all_cos = {**cos, **relevance.cosine(list(collected.values()))}
    _judge_and_select(trace, title, everything, all_cos, judge=judge, store=store,
                      cutoff=cutoff, threshold=threshold, cfg=cfg, meter=meter,
                      timings=timings)
    trace["pool_ids"] = sorted(pool)
    trace["collected_ids"] = sorted(collected)
    trace["owner"] = {str(c): p for c, p in owner.items()}
    trace["via"] = {str(c): d for c, d in via.items()}
    trace["seed_ids"] = sorted(seed_keys)
    trace["cosine_all"] = {str(c): v for c, v in all_cos.items()}
    timings["total"] = time.perf_counter() - t_start
    trace["timings_sec"] = {k: round(v, 2) for k, v in timings.items()}
    return trace


def _key(c):
    """String key of the cosine table -> store key."""
    return int(c) if str(c).isdigit() else c


def candidates_of(trace, backend):
    """Ranked candidate rows: output first, then other judged papers, then the rest by cosine."""
    rows, seen = [], set()
    collected = set(trace.get("collected_ids", []))
    owner, via = trace.get("owner", {}), trace.get("via", {})
    seeds = set(trace.get("seed_ids", []))
    judged = {r["corpusid"]: r for r in trace.get("judged", [])}
    output = {r["corpusid"] for r in trace.get("output", [])}

    def row(cid, score, cosine):
        seen.add(cid)
        if backend == "datalake":
            arxiv = arxiv_of_doi(cid)
            ident = {"doi": cid, "parent": owner.get(str(cid)),
                     **({"arxiv_id": arxiv} if arxiv else {})}
        else:
            ident = {"corpusid": cid, "parent_corpusid": owner.get(str(cid))}
        out = {**ident, "score": score, "cosine": cosine,
               "origin": "search" if cid in seeds else
                         ("expand" if cid in collected else "pool"),
               "direction": via.get(str(cid)), "in_output": cid in output}
        if cid in judged and "links" in judged[cid]:
            out["rating"] = judged[cid]["rating"]
            out["links"] = judged[cid]["links"]
        return out

    for r in trace.get("output", []):
        rows.append(row(r["corpusid"], r["score"], r["cosine"]))
    rest = sorted((r for r in trace.get("judged", []) if r["corpusid"] not in seen),
                  key=lambda r: (-(r["score"] or 0.0), -r["cosine"], r["corpusid"]))
    for r in rest:
        rows.append(row(r["corpusid"], r["score"], r["cosine"]))
    cos = trace.get("cosine_all", {})
    for cid in sorted((_key(c) for c in cos if _key(c) not in seen),
                      key=lambda c: (-cos[str(c)], c)):
        rows.append(row(cid, None, cos[str(cid)]))
    return rows
