"""Pipeline tests with a fake store, searcher, judge and relevance (no GPU, LLM or data files)."""
import datetime as dt
import json
import os
import random

import numpy as np
import pytest

from psa import util
from psa.expand import Bandit, design, fit_poisson, frontier, load_prior, vector
from psa.judge import select_output
from psa.pipeline import Config, build_pool, candidates_of, run_query
from psa.search import parse_lines, past_cutoff
from psa.store import Hop, HopStats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRIOR = os.path.join(ROOT, "priors", "expansion_prior.json")
CUTOFF = dt.date(2024, 6, 1)


# --------------------------------------------------------------------------- fakes

class FakeStore:
    """S2-shaped store with integer keys and deterministic references."""

    name = "s2"
    harvest_order = "influential_first"

    def __init__(self, n=4000, seed=1):
        rng = random.Random(seed)
        self.meta = {}
        for k in range(1, n + 1):
            year = rng.randint(2000, 2025)
            self.meta[k] = {"corpusid": k, "title": f"paper {k}", "year": year,
                            "pubdate": dt.date(year, rng.randint(1, 12), 1),
                            "citationcount": rng.choice([3, 10, 50, 200, 2000]),
                            "doi": f"10.1/{k}", "arxiv": None,
                            "abstract": f"abstract {k % 17}"}
        self.refs = {k: sorted({rng.randint(1, n) for _ in range(rng.randint(0, 12))} - {k})
                     for k in self.meta}
        self.citers = {k: [] for k in self.meta}
        for k, rs in self.refs.items():
            for r in rs:
                self.citers[r].append(k)

    def papers(self, keys):
        return {k: dict(self.meta[k]) for k in keys}

    def prefetch(self, keys):
        pass

    def after_cutoff(self, paper, cutoff):
        return cutoff is not None and paper["pubdate"] > cutoff

    def indegree_before(self, keys, cutoff):
        return {k: sum(1 for c in self.citers[k] if self.meta[c]["pubdate"] <= cutoff)
                for k in keys}

    def neighbours(self, parents, direction, *, cap=None, cutoff=None, window_years=None,
                   min_indegree_before=None, **_):
        hop = Hop()
        src = self.refs if direction == "references" else self.citers
        counts = self.indegree_before(self.meta, cutoff) if min_indegree_before else {}
        for p in dict.fromkeys(parents):
            kids = src[p]
            st = hop.stats[p] = HopStats(total=len(kids))
            keep = [c for c in kids if not self.after_cutoff(self.meta[c], cutoff)]
            if window_years is not None:
                keep = [c for c in keep if self.meta[c]["year"] >= cutoff.year - window_years]
            if min_indegree_before is not None:
                keep = [c for c in keep if counts.get(c, 0) >= min_indegree_before]
            keep = sorted(keep)[:cap] if cap else sorted(keep)
            st.produced = len(keep)
            if keep:
                hop.edges[p] = keep
        return hop

    def links(self, ids, cutoff):
        s = set(ids)
        deg = dict.fromkeys(s, 0)
        for a in s:
            for b in self.refs[a]:
                if b in s and b != a:
                    deg[a] += 1
                    deg[b] += 1
        return deg

    def key_of(self, hit):
        return hit.get("corpusId")


class FakeSearcher:
    name = "fake"

    def __init__(self, seeds):
        self.seeds = seeds
        self.last_counts, self.last_queries, self.last_llm = {}, {}, {}

    def search(self, title, *, cutoff=None):
        return sorted(self.seeds)


class FakeJudge:
    name = "fake-judge"
    last_usage = None
    true_not_in_top = 0

    def score(self, question, papers):
        return [((p["corpusid"] * 37) % 10) / 9 for p in papers]


class FakeRelevance:
    def __init__(self):
        self.cache = {}

    def cosine(self, papers):
        for p in papers:
            self.cache.setdefault(p["corpusid"], ((p["corpusid"] * 7919) % 1000) / 1000)
        return {p["corpusid"]: self.cache[p["corpusid"]] for p in papers}


def _blocklist(tmp_path, rows):
    path = tmp_path / "block.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return util.Blocklist(str(path))


# --------------------------------------------------------------------------- pipeline

def test_run_query_end_to_end(tmp_path):
    store = FakeStore()
    seeds = list(range(10, 60))
    blocked_id = seeds[3]
    blocklist = _blocklist(tmp_path, [{"corpusid": blocked_id}, {"doi": "10.1/55"}])
    trace = run_query("a title", "q1", cutoff=CUTOFF, store=store,
                      searcher=FakeSearcher(seeds), judge=FakeJudge(),
                      relevance=FakeRelevance(), blocklist=blocklist,
                      cfg=Config.for_backend("s2", PRIOR))
    assert not trace.get("error")
    seed_ids = set(trace["seed_ids"])
    pool = set(trace["pool_ids"])
    collected = set(trace["collected_ids"])
    # candidates are the base pool (seeds + their references) plus expansion
    assert seed_ids <= pool and not pool & collected
    assert {int(k) for k in trace["cosine_all"]} == pool | collected
    assert blocked_id not in pool | collected and 55 not in pool | collected
    assert all(store.meta[k]["pubdate"] <= CUTOFF for k in pool | collected)
    # budget = 0.4 x base pool
    assert trace["budget"] == pytest.approx(0.4 * len(pool))
    walk = trace["walk"]
    assert walk["rounds"] > 0
    # each (node, direction) is expanded once, and only base-pool nodes are expanded
    arms = [(r[0], r[1]) for r in walk["trace"]]
    assert len(arms) == len(set(arms))
    assert {a[0] for a in arms} <= pool
    rows = candidates_of(trace, "s2")
    assert {r["origin"] for r in rows} <= {"search", "pool", "expand"}
    assert {r["corpusid"] for r in rows} == pool | collected
    for r in trace["output"]:
        assert r["rating"] >= 8 and r["links"] >= trace["output_rule"]["links_threshold"]


def test_run_query_is_deterministic():
    def once():
        cfg = Config.for_backend("s2", PRIOR)
        return run_query("t", "q", cutoff=CUTOFF, store=FakeStore(),
                         searcher=FakeSearcher(list(range(100, 140))), judge=FakeJudge(),
                         relevance=FakeRelevance(), blocklist=util.Blocklist(None), cfg=cfg)
    a, b = once(), once()
    assert a["walk"]["trace"] == b["walk"]["trace"]
    assert a["output"] == b["output"]


def test_empty_pool_is_an_error():
    store = FakeStore()
    trace = run_query("t", "q", cutoff=dt.date(1990, 1, 1), store=store,
                      searcher=FakeSearcher([1, 2, 3]), judge=FakeJudge(),
                      relevance=FakeRelevance(), blocklist=util.Blocklist(None),
                      cfg=Config.for_backend("s2", PRIOR))
    assert trace["error"]
    assert trace["pool"]["dropped"]["after_cutoff"] == 3


def test_build_pool_owner_is_first_parent():
    store = FakeStore()
    kept, seeds, owner, drops, _ = build_pool([20, 21, 21], CUTOFF, store, util.Blocklist(None))
    assert drops.get("duplicate") == 1 or 21 in seeds
    for child, parent in owner.items():
        assert child in store.refs[parent]
        assert parent == min(p for p in (20, 21) if child in store.refs[p]
                             and not store.after_cutoff(store.meta[p], CUTOFF))


# --------------------------------------------------------------------------- parts

def test_parse_lines():
    text = '1. graph neural networks\n- "Graph Neural Networks"\n* survey of GNNs\n\nlink prediction'
    assert parse_lines(text) == ["graph neural networks", "link prediction"]


def test_past_cutoff():
    c = dt.date(2024, 6, 1)
    assert not past_cutoff({"publicationDate": "2024-05-31"}, c)
    assert past_cutoff({"publicationDate": "2024-06-02"}, c)
    assert past_cutoff({"year": 2024}, c)          # year only: the cutoff year counts as after
    assert not past_cutoff({"year": 2023}, c)
    assert past_cutoff({}, c)


def test_blocklist(tmp_path):
    bl = _blocklist(tmp_path, [{"doi": "10.1145/X"}, {"arxiv": "2101.00001v2", "corpusid": 7},
                               {"title": "A Survey: Of Things!"}])
    assert bl.hit({"doi": "https://doi.org/10.1145/x"}, use_corpusid=True) == "doi"
    assert bl.hit({"doi": "10.48550/arxiv.2101.00001"}, use_corpusid=False) == "doi"
    assert bl.hit({"corpusid": 7}, use_corpusid=True) == "corpusid"
    assert bl.hit({"corpusid": 7}, use_corpusid=False) is None
    assert bl.hit({"title": "a survey of things"}, use_corpusid=True) == "title"
    assert bl.keys_for("s2") == {7}
    assert util.Blocklist(None).hit({"doi": "10.1145/x"}, use_corpusid=True) is None


def test_select_output():
    judged = [{"corpusid": i, "cosine": 0.5, "score": s}
              for i, s in enumerate([1.0, 0.95, 0.9, 0.5, 0.0])]
    links = {0: 5, 1: 1, 2: 5, 3: 9, 4: 9}
    out, k = select_output(judged, links)
    assert k == 9.0
    assert [r["corpusid"] for r in out] == []
    out, k = select_output(judged, {0: 9, 1: 9, 2: 9, 3: 9, 4: 9})
    assert [r["corpusid"] for r in out] == [0, 1, 2]     # rating >= 8
    assert [r["rank"] for r in out] == [1, 2, 3]


def test_fit_poisson_recovers_rate():
    rng = np.random.default_rng(0)
    x = rng.normal(size=4000)
    X = np.column_stack([np.ones_like(x), x])
    expo = rng.integers(1, 50, size=4000)
    y = rng.poisson(expo * np.exp(-3.0 + 0.5 * x))
    w, prec = fit_poisson(X, y, np.zeros(2), np.eye(2) * 1e-6, offset=np.log(expo))
    assert abs(w[0] + 3.0) < 0.1 and abs(w[1] - 0.5) < 0.1
    assert prec.shape == (2, 2)


def test_load_prior_and_vector():
    w, precision, scale, data, cols = load_prior(PRIOR)
    assert data["features"][0] == "intercept" and len(w) == len(cols) + 1
    assert precision[0, 0] == pytest.approx(0.01) and precision[0, 1:].sum() == 0
    row = {"age": -3, "citationcount": 99, "indegree_before": 4, "embed_cosine": 0.4,
           "expected_pairs": {"references": 10, "cited_by": 0}}
    raw = [vector(row, "references")[i] for i in cols]
    x = design(raw, scale)
    assert len(x) == len(w) and x[0] == 1.0


def test_bandit_picks_every_arm_once():
    arms = [(1, "references"), (1, "cited_by"), (2, "references")]
    x = {a: np.array([1.0, i]) for i, a in enumerate(arms)}
    b = Bandit(arms, x, [0.0, 1.0], np.eye(2) * 1e6, np.random.default_rng(0))
    got = [b.pick() for _ in range(3)]
    assert got == [(2, "references"), (1, "cited_by"), (1, "references")]
    b.observe(got[0], 3, 10)
    assert len(b.seen_y) == 1


def test_frontier_is_seeded():
    nodes = [{"corpusid": i, "rank": i} for i in range(1, 501)]
    a, top, sampled = frontier(nodes)
    b, _, _ = frontier(nodes)
    assert (top, sampled) == (200, 100) and a == b
    assert [n["corpusid"] for n in a[:200]] == list(range(1, 201))


def test_meter_union():
    m = util.Meter("q")
    m.intervals["search"] = [(0, 2), (1, 3), (5, 6)]
    assert m._union(m.intervals["search"]) == 4
    with m.stage("judge"):
        with m.call("llm", n=3):
            pass
    assert m.summary()["calls"]["judge"]["llm"] == 3


def test_redact(monkeypatch):
    monkeypatch.setenv("S2_API_KEY", "abcdefghijklmnop")
    assert "abcdefghijklmnop" not in util.redact("key abcdefghijklmnop here")
    assert util.redact("?api_key=zzz&x=1") == "?api_key=***&x=1"
