"""Scoring and prior-training arithmetic tests (no data files)."""
import datetime as dt

import numpy as np

import score
from psa import train as train_prior

CUTOFF = dt.date(2024, 6, 1)


def test_score_query_numbers():
    gold = {cid: {"pubdate": dt.date(2023, 1, 1), "in_store": {"s2": cid != 10}}
            for cid in range(1, 11)}
    order = [1, 2, 3, 4, 5, 6, 99, 98]               # 6 gold + 2 non-gold
    parents = {6: {98}}                              # 6 is found only through a violating parent
    q = score.score_query(order, parents, gold, CUTOFF, {98},   # 1 cutoff violation
                          score.reach_key("s2"), output={1, 2, 99})
    assert q["n_cutoff_violations"] == 1
    assert q["all"]["recall@pool"] == 0.6
    assert q["reachable"]["n_gold"] == 9
    assert q["all"]["recall@output"] == 0.2
    assert q["all"]["precision@output"] == 2 / 3
    assert q["all"]["gold_only_via_violating_parent"] == 1
    assert q["all"]["recall_by_band"]["1-2"]["hit"] == 6


def test_reach_key_datalake_falls_back_to_doi():
    reach = score.reach_key("datalake")
    assert reach({"datalake_doi": True})
    assert not reach({"datalake_any": False, "datalake_doi": True})
    assert reach({"datalake_any": True, "datalake_doi": False})


def test_dedup_keeps_best_and_all_parents():
    rows = [{"corpusid": 5, "score": 0.2, "parent_corpusid": 1},
            {"corpusid": 5, "score": 0.9, "parent_corpusid": 2},
            {"corpusid": 6, "score": None}]
    order, parents = score.dedup(score.keyed(rows, "s2", {}))
    assert order == [5, 6] and parents[5] == {1, 2}


def test_training_rows_labels():
    h = {"qid": "q", "cutoff": "2024-06-01", "pool": [1, 2],
         "nodes": [{"corpusid": 3, "year": 2020, "citationcount": 10, "indegree_before": 4,
                    "cosine": 0.3, "references": [1, 7, 8], "cited_by": []}]}
    gold = {"q": {1: {}, 7: {}}}
    X, y, off = train_prior.training_rows([h], gold)
    assert X.shape == (1, len(train_prior.TRAIN_FEATURES))
    assert y.tolist() == [1.0]                         # 1 is in the base pool, so only 7 counts
    assert np.isclose(off[0], np.log(3))


def test_datalake_rows_match_gold_by_doi(tmp_path):
    from psa import util
    path = tmp_path / "gold.jsonl"
    path.write_text(
        '{"qid": "q", "corpusid": 1, "doi": "10.1/A", "arxiv": "2101.00001"}\n'
        '{"qid": "q", "corpusid": 2, "arxiv": "2101.00002"}\n'
        '{"qid": "q", "corpusid": 3}\n', encoding="utf-8")
    gold = util.load_gold(str(path), by="doi")["q"]
    assert len(gold) == 3                              # gold without a DOI stays in the denominator
    rows = [{"doi": "10.48550/arxiv.2101.00001", "score": 0.9, "parent": None},
            {"doi": "10.1/a", "score": 0.5, "parent": "10.48550/arxiv.2101.00002"},
            {"doi": "10.9/other", "score": 0.1, "parent": None}]
    order, parents = score.dedup(score.keyed(rows, "datalake", util.gold_aliases(gold)))
    assert order == ["10.1/a", "10.9/other"]           # both DOIs of paper 1 collapse to one key
    assert parents["10.1/a"] == {"10.48550/arxiv.2101.00002"}
    assert len(set(order) & set(gold)) == 1
    assert set(util.load_gold(str(path))["q"]) == {1, 2, 3}


def test_training_rows_datalake_aliases():
    h = {"qid": "q", "cutoff": "2024-06-01", "backend": "datalake", "pool": ["10.1/p"],
         "nodes": [{"corpusid": "10.1/n", "year": 2020, "citationcount": 10, "indegree_before": 4,
                    "cosine": 0.3, "references": ["10.48550/arxiv.2101.00001", "10.1/x"],
                    "cited_by": []}]}
    gold = {"q": {"10.1/a": {"dois": ["10.1/a", "10.48550/arxiv.2101.00001"]}}}
    _, y, _ = train_prior.training_rows([h], gold)
    assert y.tolist() == [1.0]
