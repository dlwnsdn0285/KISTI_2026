"""Scoring: compare a run.py result folder with gold.jsonl and report recall / precision.

Units: pool (all candidates), @200, output (final list).
Denominators: all gold, and the gold the store has.
Each run is scored with its own store: s2 runs match gold by corpusId, datalake runs by DOI.

    python score.py --config config.yaml --run results/s2 --gold data/eval/gold.jsonl \
        --blocklist data/eval/blocklist.jsonl --report reports/s2.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

from psa import util

TOP_N = 200
BANDS = ((1, 2), (3, 5), (6, 10), (11, 10_000))     # years before the cutoff


def band_of(years):
    for lo, hi in BANDS:
        if lo <= years <= hi:
            return f"{lo}-{hi}" if hi < 10_000 else f"{lo}+"
    return None


def reach_key(backend):
    """Reader for whether the store has a gold paper."""
    if backend == "datalake":
        return lambda f: bool(f.get("datalake_any", f.get("datalake_doi", False)))
    return lambda f: bool(f.get(backend, False))


def keyed(rows, backend, alias):
    """Give each candidate row a `key` / `parent_key` comparable with gold keys, and its store id."""
    if backend != "datalake":
        return [dict(r, key=int(r["corpusid"]), store_id=int(r["corpusid"]),
                     parent_key=(int(r["parent_corpusid"])
                                 if r.get("parent_corpusid") is not None else None))
                for r in rows]
    return [dict(r, key=alias.get(r["doi"], r["doi"]), store_id=r["doi"],
                 parent_key=alias.get(r.get("parent"), r.get("parent")))
            for r in rows]


def dedup(rows):
    """One row per key (best rank)."""
    best, parents = {}, defaultdict(set)
    for i, c in enumerate(rows):
        cid = c["key"]
        rank = (-(c.get("score") if c.get("score") is not None else 0.0), i)
        if cid not in best or rank < best[cid][0]:
            best[cid] = (rank, c)
        if c.get("parent_key") is not None:
            parents[cid].add(c["parent_key"])
    return sorted(best, key=lambda cid: best[cid][0]), parents


def score_query(order, parents, gold, cutoff, violated, reach, output):
    """Metrics for one query. `violated`: candidates published after the cutoff."""
    top = order[:TOP_N]
    out = {"n_pool": len(order), "n_top": len(top), "n_output": len(output),
           "n_cutoff_violations": len(violated)}
    pool, topset = set(order), set(top)
    for label, keep in (("all", lambda g: True), ("reachable", lambda g: reach(g["in_store"]))):
        sub = {cid for cid, g in gold.items() if keep(g)}
        hit = sub & pool
        clean = [c for c in order if c not in violated]
        m = {"n_gold": len(sub),
             "recall@pool": len(hit) / len(sub) if sub else None,
             "recall@200": len(sub & topset) / len(sub) if sub else None,
             "recall@output": len(sub & output) / len(sub) if sub else None,
             "precision@pool": len(hit) / len(order) if order else None,
             "precision@200": len(sub & topset) / len(top) if top else None,
             "precision@output": len(sub & output) / len(output) if output else None,
             "precision@pool_excl_violations": (len(hit - violated) / len(clean)
                                                if clean else None)}
        bands = defaultdict(lambda: {"gold": 0, "hit": 0})
        for cid in sub:
            pub = gold[cid]["pubdate"]
            b = (band_of(max(1, round((cutoff - pub).days / 365.25)))
                 if pub is not None and cutoff is not None else "미상")
            if b:
                bands[b]["gold"] += 1
                bands[b]["hit"] += cid in hit
        m["recall_by_band"] = {k: {**v, "recall": v["hit"] / v["gold"] if v["gold"] else None}
                               for k, v in sorted(bands.items())}
        # gold found only through a parent after the cutoff
        m["gold_only_via_violating_parent"] = sum(
            1 for cid in hit if parents.get(cid) and parents[cid] <= violated)
        out[label] = m
    return out


def mm(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return {"mean": round(statistics.fmean(vals), 4), "median": round(statistics.median(vals), 4),
            "min": round(min(vals), 4), "max": round(max(vals), 4), "n": len(vals)}


def summarize(per_query, extra):
    out = {}
    for label in ("all", "reachable"):
        s = {m: mm([q[label][m] for q in per_query])
             for m in ("recall@pool", "recall@200", "recall@output", "precision@pool",
                       "precision@200", "precision@output",
                       "precision@pool_excl_violations")}
        bands = defaultdict(lambda: {"gold": 0, "hit": 0})
        for q in per_query:
            for k, v in q[label]["recall_by_band"].items():
                bands[k]["gold"] += v["gold"]
                bands[k]["hit"] += v["hit"]
        s["recall_by_band_pooled"] = {k: {**v, "recall": round(v["hit"] / v["gold"], 4)
                                          if v["gold"] else None}
                                      for k, v in sorted(bands.items())}
        s["recall_pool_before_expansion"] = mm([e[label] for e in extra])
        s["gold_only_via_violating_parent"] = sum(
            q[label]["gold_only_via_violating_parent"] for q in per_query)
        out[label] = s
    for k in ("n_pool", "n_output", "n_cutoff_violations"):
        vals = [q[k] for q in per_query]
        out[k] = {"sum": sum(vals), "median": statistics.median(vals) if vals else None}
    out["n_empty_output"] = sum(1 for q in per_query if q["n_output"] == 0)
    return out


def main():
    ap = argparse.ArgumentParser(description="run.py 결과 채점")
    ap.add_argument("--run", required=True, help="run.py --out 폴더")
    ap.add_argument("--gold", required=True,
                    help="gold.jsonl: 한 줄에 {qid, corpusid 또는 doi/arxiv, pubdate, in_store}")
    ap.add_argument("--blocklist", default=None, help="차단 목록 jsonl (점검용)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--report", default=None, help="보고서 json 경로")
    args = ap.parse_args()

    from psa.store import open_backend
    cfg = util.load_config(args.config)
    stores, gold_sets = {}, {}
    blocklist = util.Blocklist(args.blocklist)

    runs = [json.load(open(p, encoding="utf-8"))
            for p in sorted(glob.glob(os.path.join(args.run, "*.json")))
            if not os.path.basename(p).startswith("_")]
    if not runs:
        sys.exit(f"결과 파일이 없다: {args.run}")
    per_query, extra, inflow = [], [], defaultdict(Counter)
    totals = Counter()
    for r in runs:
        backend = r.get("backend", "s2")
        reach = reach_key(backend)
        if backend not in stores:
            stores[backend] = open_backend(backend, config=cfg["store"])
            gold_sets[backend] = util.load_gold(
                args.gold, by="doi" if backend == "datalake" else "corpusid")
        store = stores[backend]
        totals["errors"] += bool(r.get("error"))
        gold = gold_sets[backend].get(r["qid"], {})
        cutoff = util._date(r.get("cutoff"))
        rows = keyed(r["candidates"], backend, util.gold_aliases(gold))
        totals["rows"] += len(rows)
        order, parents = dedup(rows)
        store_id = {}
        for c in rows:
            store_id.setdefault(c["key"], c["store_id"])
        papers = store.papers(list(store_id.values()))
        paper = {k: papers[sid] for k, sid in store_id.items()}
        if backend == "datalake":       # years only
            violated = {k for k in order if store.after_cutoff(paper[k], cutoff)}
        else:
            violated = {k for k in order if cutoff is not None
                        and paper[k]["pubdate"] is not None and paper[k]["pubdate"] > cutoff}
        output = {c["key"] for c in rows if c.get("in_output")}
        q = score_query(order, parents, gold, cutoff, violated, reach, output)
        q["qid"] = r["qid"]
        q["blocklist_hits"] = sum(1 for k in order
                                  if blocklist.hit(paper[k], use_corpusid=backend == "s2"))
        per_query.append(q)
        before = {c["key"] for c in rows if c["origin"] != "expand"}
        e = {}
        for label, keep in (("all", lambda g: True), ("reachable", lambda g: reach(g["in_store"]))):
            sub = {cid for cid, g in gold.items() if keep(g)}
            e[label] = len(sub & before) / len(sub) if sub else None
        extra.append(e)
        for c in rows:
            g = gold.get(c["key"])
            if g is None:
                continue
            how = ({"search": "seed", "pool": "pool"}.get(c["origin"])
                   or f"walk_{c.get('direction') or '?'}")
            b = (band_of(max(1, round((cutoff - g["pubdate"]).days / 365.25)))
                 if g["pubdate"] and cutoff else None) or "미상"
            inflow[b][how] += 1
            if c.get("in_output"):
                inflow[b]["in_output"] += 1
        m = r.get("meter") or {}
        for k in ("search", "expand", "judge"):
            totals[f"stage_sec_{k}"] += (m.get("stage_wall_sec") or {}).get(k, 0.0)
        cache = m.get("search_cache") or {}
        totals["cache_hit"] += cache.get("hit", 0)
        totals["cache_lookups"] += cache.get("hit", 0) + cache.get("miss", 0)

    report = {"run": args.run, "n_queries": len(per_query), "errors": totals["errors"],
              "summary": summarize(per_query, extra),
              "rows": totals["rows"],
              "blocklist_hits": sum(q["blocklist_hits"] for q in per_query),
              "gold_found_by_band_and_inflow": {k: dict(v) for k, v in sorted(inflow.items())},
              "stage_sec_mean": {k: round(totals[f"stage_sec_{k}"] / len(runs), 2)
                                 for k in ("search", "expand", "judge")},
              "search_cache_hit_rate": (round(totals["cache_hit"] / totals["cache_lookups"], 4)
                                        if totals["cache_lookups"] else None),
              "per_query": per_query}
    s = report["summary"]
    print(f"{args.run}: 질의 {len(per_query)} · 오류 {totals['errors']} · 후보 {totals['rows']:,}")
    for label in ("all", "reachable"):
        x = s[label]
        fmt = lambda v: f"{v['mean']:.4f} ({v['median']:.4f})" if v else "—"  # noqa: E731
        print(f"  [{label}] recall@pool {fmt(x['recall@pool'])} · recall@output "
              f"{fmt(x['recall@output'])} · precision@output {fmt(x['precision@output'])}")
    print(f"  cutoff 위반 {s['n_cutoff_violations']['sum']:,} · 차단 목록 적중 "
          f"{report['blocklist_hits']:,} (둘 다 0 이어야 한다)")
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=1, default=str)
        print(f"→ {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
