"""Train the expansion score model (bandit prior) from queries with gold (run.py --train).

harvest: collect seeds, base pool, frontier and neighbours per query (needs the LLM server).
fit:     Poisson MAP fit on the harvest and the gold file; writes a prior json.
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import math
import os
import sys
import time

import numpy as np

from psa import util

# Snapshot citation counts are left out: they include post-cutoff citations.
TRAIN_FEATURES = ["age", "log_indegree_before", "is_cited_by", "log_expected_pairs",
                  "relevance"]


def harvest(cfg, rows, backend, blocklist, out_dir):
    """Write one harvest file per query into out_dir (existing files are skipped)."""
    from psa.expand import MiniLM, Relevance
    from psa.pipeline import Config, prepare
    from psa.search import LLMQuerySearcher, SearchCache
    from psa.store import open_backend

    os.makedirs(out_dir, exist_ok=True)
    run_cfg = Config.for_backend(backend, cfg["prior"])
    model = MiniLM(cfg["embed"]["model"], cfg["embed"]["device"], cfg["embed"]["batch_size"])
    for i, q in enumerate(rows, 1):
        path = os.path.join(out_dir, f"{q['qid']}.json")
        if os.path.exists(path):
            continue
        meter = util.Meter(q["qid"], backend=backend)
        store = open_backend(backend, meter=meter, config=cfg["store"])
        cache = SearchCache(cfg, meter=meter)
        searcher = LLMQuerySearcher(cache=cache, cfg=cfg,
                                    key_of=store.key_of, meter=meter)
        relevance = Relevance(q["title"], model)
        t0 = time.perf_counter()
        out = {"qid": q["qid"], "question": q["title"], "backend": backend,
               "cutoff": q["cutoff"].isoformat() if q["cutoff"] else None}
        try:
            state, trace = prepare(q["title"], q["qid"], cutoff=q["cutoff"], store=store,
                                   searcher=searcher, relevance=relevance,
                                   blocklist=blocklist, cfg=run_cfg, meter=meter)
            out["error"] = trace.get("error", "")
            if state is not None:
                p, sets, indeg, cos = state["papers"], state["sets"], state["indeg"], state["cos"]
                out.update({
                    "pool": sorted(state["pool"]), "seeds": sorted(state["seeds"]),
                    "threshold": state["threshold"],
                    "nodes": [{"corpusid": c, "year": p[c]["year"],
                               "citationcount": p[c]["citationcount"],
                               "indegree_before": indeg.get(c, 0),
                               "cosine": round(cos[c], 6),
                               "references": sets["references"][c],
                               "cited_by": sets["cited_by"][c]} for c in state["ids"]]})
        except Exception as exc:  # noqa: BLE001
            out["error"] = util.redact(f"{type(exc).__name__}: {exc}")
        out["wall_sec"] = round(time.perf_counter() - t0, 2)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(out, fh)
        print(f"[{i}/{len(rows)}] {q['qid']}: 풀 {len(out.get('pool', []))} · 노드 "
              f"{len(out.get('nodes', []))} · {out['wall_sec']:.0f}s"
              + (f" · 오류 {out['error']}" if out.get("error") else ""), flush=True)


def training_rows(harvests, golds):
    """Feature rows, labels (gold not in the base pool) and offsets per (node, direction)."""
    from psa.expand import DIRECTIONS, FEATURE_NAMES, vector
    cols = [FEATURE_NAMES.index(n) for n in TRAIN_FEATURES]
    X, y, off = [], [], []
    for h in harvests:
        gold = set(golds.get(h["qid"], {}))
        alias = util.gold_aliases(golds.get(h["qid"], {})) if h.get("backend") == "datalake" else {}
        pool = set(h["pool"])
        cutoff = dt.date.fromisoformat(h["cutoff"]) if h.get("cutoff") else None
        for n in h["nodes"]:
            row = {"age": (n["year"] - cutoff.year) if (n["year"] and cutoff) else None,
                   "citationcount": n["citationcount"],
                   "indegree_before": n["indegree_before"], "embed_cosine": n["cosine"],
                   "expected_pairs": {d: len(n[d]) for d in DIRECTIONS}}
            for d in DIRECTIONS:
                pairs = row["expected_pairs"][d]
                if pairs <= 0:
                    continue
                full = vector(row, d)
                X.append([full[i] for i in cols])
                y.append(float(sum(1 for c in n[d] if c not in pool and alias.get(c, c) in gold)))
                off.append(math.log(max(1, pairs)))
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float), off


def fit(harvest_dir, gold_path, out_path):
    """Fit the prior on the harvest files and write it to out_path."""
    from psa.expand import fit_poisson
    golds, harvests = None, []
    for path in sorted(glob.glob(os.path.join(harvest_dir, "*.json"))):
        with open(path, encoding="utf-8") as fh:
            h = json.load(fh)
        if h.get("error") or not h.get("nodes"):
            continue
        if golds is None:
            golds = util.load_gold(gold_path,
                                   by="doi" if h.get("backend") == "datalake" else "corpusid")
        if h["qid"] not in golds:
            continue
        harvests.append(h)
    if not harvests:
        sys.exit(f"학습할 수확이 없다 (정답이 있는 질의): {harvest_dir}")
    X, y, off = training_rows(harvests, golds)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    dim = X.shape[1] + 1
    Xs = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    w, precision = fit_poisson(Xs, y, np.zeros(dim), np.eye(dim), offset=off)
    out = {"features": ["intercept"] + TRAIN_FEATURES,
           "fitted_on": {"queries": len(harvests), "rows": int(len(X)),
                         "positive_rows": int((y > 0).sum()), "gold_total": float(y.sum())},
           "w": w.tolist(), "precision": precision.tolist(),
           "scale": {"mean": mu.tolist(), "sd": sd.tolist()},
           "created": dt.datetime.now().isoformat(timespec="seconds")}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"질의 {len(harvests)} · 행 {len(X):,} (양성 {int((y > 0).sum()):,}) · "
          + " · ".join(f"{n} {v:+.3f}" for n, v in zip(out["features"], w))
          + f"\n→ {out_path}")

