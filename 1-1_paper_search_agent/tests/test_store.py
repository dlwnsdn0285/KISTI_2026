"""Tests for psa/store.py.

Most build a small graph in in-memory DuckDB and run the real SQL.
The last few run only when real store files are set (PSA_S2_GRAPH_DB, PSA_DATALAKE_ROOT).
"""
import datetime as dt
import os

import duckdb
import pytest

from psa import store

CUTOFF = dt.date(2020, 6, 1)


# ---------------------------------------------------------------- normalisation

def test_norm_doi():
    assert store.norm_doi(" https://doi.org/10.1145/ABC ") == "10.1145/abc"
    assert store.norm_doi("") is None and store.norm_doi(None) is None


def test_arxiv_ids_and_dois():
    assert store.normalize_arxiv_id("arXiv:2101.00001v3") == "2101.00001"
    assert store.normalize_arxiv_id("hep-th/9901001v2") == "hep-th/9901001"
    assert store.normalize_arxiv_id("not an id") is None
    assert store.arxiv_id_to_doi("2101.00001v1") == "10.48550/arxiv.2101.00001"
    assert store.arxiv_of_doi("10.48550/arXiv.2101.00001") == "2101.00001"
    assert store.arxiv_of_doi("10.1145/1") is None


def test_normalize_title():
    assert store.normalize_title("  Deep  Learning: A Survey! ") == "deep learning a survey"
    # U+0130 does not split the word
    assert store.normalize_title("İlişkin") == "ilişkin"
    assert store.normalize_title(None) == ""


def test_years_before_handles_29_february():
    assert store.years_before(dt.date(2020, 2, 29), 1) == dt.date(2019, 2, 28)


def test_config_from_env_and_dict(monkeypatch):
    monkeypatch.setenv("PSA_S2_GRAPH_DB", "/x/g.duckdb")
    monkeypatch.setenv("PSA_DUCKDB_THREADS", "4")
    monkeypatch.delenv("PSA_DUCKDB_MEMORY", raising=False)
    cfg = store.StoreConfig.coerce(None)
    assert cfg.s2_graph_db == "/x/g.duckdb" and cfg.duckdb_threads == 4
    assert cfg.duckdb_memory == "8GB"
    cfg = store.StoreConfig.coerce({"s2_graph_db": "/y.duckdb", "duckdb_memory": "2GB"})
    assert cfg.s2_graph_db == "/y.duckdb" and cfg.duckdb_memory == "2GB"


def test_missing_path_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("PSA_S2_GRAPH_DB", raising=False)
    with pytest.raises(RuntimeError, match="PSA_S2_GRAPH_DB"):
        store.open_backend("s2")
    with pytest.raises(ValueError):
        store.open_backend("nope", con=duckdb.connect())


# ---------------------------------------------------------------- S2 (in-memory graph)
# Parent 1 cites 10, 11, 12. Parent 2 is cited by 20..24: 24 is after the cutoff,
# 23 has no date, and only 20 is an influential citation.

S2_PAPERS = [  # corpusid, year, pubdate, citationcount, doi, arxiv, title
    (1, 2010, dt.date(2010, 1, 1), 40, "10.1/one", None, "One"),
    (2, 2005, dt.date(2005, 1, 1), 5000, "10.1/two", "0501.00002", "Two"),
    (10, 1999, dt.date(1999, 5, 5), 100, None, None, "Ten"),
    (11, 2016, dt.date(2016, 5, 5), 7, None, None, None),
    (12, 2018, dt.date(2018, 5, 5), 2, None, None, None),
    (20, 2019, dt.date(2019, 1, 1), 30, None, None, None),
    (21, 2018, dt.date(2018, 1, 1), 3, None, None, None),
    (22, 2004, dt.date(2004, 1, 1), 60, None, None, None),
    (23, None, None, 9, None, None, None),
    (24, 2021, dt.date(2021, 1, 1), 9, None, None, None),
]
S2_EDGES = [(1, 10, False), (1, 11, False), (1, 12, True),
            (20, 2, True), (21, 2, False), (22, 2, False), (23, 2, False), (24, 2, False)]


@pytest.fixture
def s2con():
    con = duckdb.connect()
    con.execute("CREATE TABLE idmap (corpusid INTEGER, year INTEGER, pubdate DATE, "
                "citationcount INTEGER, doi VARCHAR, arxiv VARCHAR, title VARCHAR)")
    con.executemany("INSERT INTO idmap VALUES (?, ?, ?, ?, ?, ?, ?)", S2_PAPERS)
    for table in ("citing", "cited"):
        con.execute(f"CREATE TABLE {table} (citing INTEGER, cited INTEGER, "
                    "infl BOOLEAN, intents INTEGER)")
        con.executemany(f"INSERT INTO {table} VALUES (?, ?, ?, 0)", S2_EDGES)
    con.execute("CREATE TABLE doi2corpus (doi VARCHAR, corpusid INTEGER, year INTEGER)")
    con.executemany("INSERT INTO doi2corpus VALUES (?, ?, NULL)",
                    [("10.1/one", 1), ("10.1/two", 7), ("10.1/two", 2)])
    con.execute("CREATE TABLE arxiv2corpus (arxiv VARCHAR, corpusid INTEGER)")
    con.execute("INSERT INTO arxiv2corpus VALUES ('0501.00002', 2)")
    con.execute("ATTACH ':memory:' AS ab")
    con.execute("CREATE TABLE ab.abstracts (corpusid INTEGER, abstract VARCHAR)")
    con.execute("INSERT INTO ab.abstracts VALUES (1, 'abstract one')")
    return con


@pytest.fixture
def s2(s2con):
    return store.open_backend("s2", con=s2con)


def hop(s2, **kw):
    kw.setdefault("cutoff", CUTOFF)
    return s2.neighbours([2], "cited_by", **kw)


def test_s2_cutoff_before_the_cap_and_influential_first(s2):
    one = hop(s2, cap=1)
    assert one.edges[2] == [20]
    assert one.stats[2].after_cutoff == 1 and one.stats[2].total == 5


def test_s2_truncated_is_counted(s2):
    h = hop(s2, cap=2)
    assert h.stats[2].produced == 2 and h.stats[2].truncated == 2
    full = hop(s2, cap=None)
    assert full.stats[2].truncated == 0 and full.stats[2].produced == 4


def test_s2_window_counts_separately_from_the_cutoff(s2):
    h = hop(s2, window_years=3)
    assert set(h.edges[2]) == {20, 21}
    assert h.stats[2].after_cutoff == 1 and h.stats[2].filtered == 2


def test_s2_undated_passes_the_cutoff_but_fails_a_window(s2):
    assert 23 in hop(s2).edges[2]
    assert 23 not in hop(s2, window_years=3).edges[2]


def test_s2_references_direction(s2):
    h = s2.neighbours([1], "references", cutoff=CUTOFF)
    assert set(h.edges[1]) == {10, 11, 12} and h.stats[1].total == 3
    assert s2.neighbours([10], "references", cutoff=CUTOFF).stats == {}


def test_s2_all_excluded_parent_still_reports_its_lookup(s2):
    h = s2.neighbours([2], "cited_by", cutoff=dt.date(2003, 1, 1), window_years=1)
    assert h.edges == {} and h.stats[2].total == 5 and h.reached() == 1
    assert h.stats[2].after_cutoff == 4 and h.stats[2].filtered == 1


def test_s2_bad_arguments_refused(s2):
    with pytest.raises(ValueError, match="cap"):
        s2.neighbours([2], "cited_by", cap=0)
    with pytest.raises(ValueError, match="direction"):
        s2.neighbours([2], "forward")


def test_s2_indegree_floor_before_the_cap(s2, monkeypatch):
    monkeypatch.setattr(store, "s2_indegree_before",
                        lambda con, ids, *, before: {20: 12, 21: 9, 22: 1, 23: 4})
    h = hop(s2, cap=1, min_indegree_before=5)
    assert h.edges[2] == [20]
    assert h.stats[2].truncated == 1 and h.stats[2].filtered == 2
    assert h.counts[22] == 1


def test_s2_indegree_before_is_strict(s2):
    # 20, 21, 22 count; 23 (no date) and 24 (after) do not
    # papers nobody cites still get a key
    assert s2.indegree_before([2, 10, 20], CUTOFF) == {2: 3, 10: 1, 20: 0}


def test_s2_papers_every_id_gets_a_dict(s2):
    got = s2.papers([1, 999])
    assert got[1]["title"] == "One" and got[1]["abstract"] == "abstract one"
    assert got[1]["pubdate"] == dt.date(2010, 1, 1) and got[1]["citationcount"] == 40
    assert got[999] == {"corpusid": 999, "title": "", "year": None, "pubdate": None,
                        "citationcount": None, "doi": None, "arxiv": None,
                        "abstract": None}


def test_s2_after_cutoff_day_then_year(s2):
    assert s2.after_cutoff({"pubdate": dt.date(2020, 6, 2), "year": 2020}, CUTOFF)
    assert not s2.after_cutoff({"pubdate": dt.date(2020, 6, 1), "year": 2020}, CUTOFF)
    assert s2.after_cutoff({"pubdate": None, "year": 2021}, CUTOFF)
    assert not s2.after_cutoff({"pubdate": None, "year": None}, CUTOFF)
    assert not s2.after_cutoff({"year": 2030}, None)


def test_s2_links_inside_the_set(s2):
    # only edges inside the set; a citing paper after the cutoff still counts
    assert s2.links([1, 10, 11, 2, 24], CUTOFF) == {1: 2, 10: 1, 11: 1, 2: 1, 24: 1}


def test_s2_key_of_and_resolve(s2):
    assert s2.key_of({"corpusId": "20"}) == 20 and s2.key_of({}) is None
    rows = [{"doi": "https://doi.org/10.1/TWO"}, {"doi": "10.9/none", "arxiv": "0501.00002v2"},
            {"doi": "10.9/none"}, {}]
    # smallest corpusId on multiple matches; the arxiv value when there is no DOI
    assert s2.resolve(rows) == [2, 2, None, None]


# ---------------------------------------------------------------- datalake (in-memory)

@pytest.fixture
def dl():
    con = duckdb.connect()
    for cat in ("g", "m", "s"):
        con.execute(f"ATTACH ':memory:' AS {cat}")
    con.execute("CREATE TABLE g.edges (citing_doi VARCHAR, cited_doi VARCHAR)")
    con.execute("CREATE TABLE s.edges (citing_doi VARCHAR, cited_doi VARCHAR)")
    for cat in ("m", "s"):
        con.execute(f"CREATE TABLE {cat}.meta (doi VARCHAR, title VARCHAR, year INTEGER, "
                    "abstract VARCHAR, cited_by_count BIGINT)")
    years = {"p": 2015, "a": 2019, "b": 2018, "c": 2021, "d": None, "e": 2000,
             "x1": 2019, "x2": 2020, "x3": 2021, "x4": None, "z": 2017}
    for k, y in years.items():
        con.execute("INSERT INTO m.meta VALUES (?, ?, ?, NULL, NULL)", [k, f"T{k}", y])
    con.execute("INSERT INTO s.meta VALUES ('f', 'Tf', 2019, 'abs f', 4)")
    for citer in "abcde":
        con.execute("INSERT INTO g.edges VALUES (?, 'p')", [citer])
    con.execute("INSERT INTO s.edges VALUES ('f', 'p')")
    for citer, cited in [("x1", "a"), ("x2", "a"), ("x3", "a"), ("x4", "b")]:
        con.execute("INSERT INTO g.edges VALUES (?, ?)", [citer, cited])
    con.execute("INSERT INTO s.edges VALUES ('x1', 'f')")
    con.execute("INSERT INTO g.edges VALUES ('p', 'z'), ('p', 'e')")
    return store.open_backend("datalake", con=con)


def test_dl_cited_by_cutoff_window_floor_and_order(dl):
    h = dl.neighbours(["p"], "cited_by", cap=500, cutoff=CUTOFF, window_years=15,
                      min_indegree_before=1)
    assert h.edges["p"] == ["a", "f"]
    st = h.stats["p"]
    assert (st.total, st.after_cutoff, st.filtered, st.produced) == (6, 1, 3, 2)
    assert h.counts["a"] == 2 and h.counts["b"] == 0 and h.counts["f"] == 1


def test_dl_cap_keeps_the_most_cited_before_the_cutoff(dl):
    h = dl.neighbours(["p"], "cited_by", cap=1, cutoff=CUTOFF, window_years=15,
                      min_indegree_before=1)
    assert h.edges["p"] == ["a"] and h.stats["p"].truncated == 1


def test_dl_unknown_year_survives_the_cutoff(dl):
    h = dl.neighbours(["p"], "cited_by", cutoff=CUTOFF)
    assert set(h.edges["p"]) == {"a", "b", "d", "e", "f"}


def test_dl_references_window(dl):
    assert dl.neighbours(["p"], "references", cap=500, cutoff=CUTOFF,
                         window_years=15).edges["p"] == ["z"]


def test_dl_indegree_before_is_strict(dl):
    assert dl.indegree_before(["a", "b", "f", "nope"], CUTOFF) == {
        "a": 2, "b": 0, "f": 1, "nope": 0}


def test_dl_links_both_directions(dl):
    dl.con.execute("INSERT INTO g.edges VALUES ('a', 'c')")
    assert dl.links(["p", "a", "c", "x1"], CUTOFF) == {"p": 2, "a": 2, "c": 1, "x1": 1}


def test_dl_papers_main_then_sidecar_and_missing_keys(dl):
    got = dl.papers(["a", "f", "10.48550/arxiv.1901.00001"])
    assert got["a"]["year"] == 2019 and got["a"]["citationcount"] == 0
    assert got["f"]["abstract"] == "abs f" and got["f"]["citationcount"] == 4
    miss = got["10.48550/arxiv.1901.00001"]
    assert miss["doi"] == "10.48550/arxiv.1901.00001" and miss["arxiv"] == "1901.00001"
    assert miss["title"] == "" and miss["year"] is None


def test_dl_after_cutoff_is_year_level(dl):
    assert dl.after_cutoff({"year": 2020}, CUTOFF) is False
    assert dl.after_cutoff({"year": 2021}, CUTOFF) is True
    assert dl.after_cutoff({"year": None}, CUTOFF) is False


def test_dl_key_of(dl):
    assert dl.key_of({"externalIds": {"DOI": "10.1/X"}}) == "10.1/x"
    assert dl.key_of({"externalIds": {"ArXiv": "2101.00001"}}) == "10.48550/arxiv.2101.00001"
    assert dl.key_of({"externalIds": {}}) is None


def test_backend_attributes(s2, dl):
    assert (s2.name, s2.harvest_order) == ("s2", "influential_first")
    assert (dl.name, dl.harvest_order) == ("datalake", "indegree_before_desc")


class _Meter:
    def __init__(self):
        self.calls, self.counts = [], {}

    def call(self, name, n=1):
        import contextlib
        self.calls.append((name, n))
        return contextlib.nullcontext()

    def count(self, name, v=1):
        self.counts[name] = self.counts.get(name, 0) + v


def test_meter_sees_store_calls(s2con):
    m = _Meter()
    b = store.open_backend("s2", con=s2con, meter=m)
    b.neighbours([1, 2], "cited_by", cutoff=CUTOFF)
    assert ("local_db", 2) in m.calls and "local_db_wait_ms" in m.counts


# ---------------------------------------------------------------- real stores (if present)

needs_s2 = pytest.mark.skipif(not os.environ.get("PSA_S2_GRAPH_DB"),
                              reason="PSA_S2_GRAPH_DB 없음")
needs_dl = pytest.mark.skipif(not os.environ.get("PSA_DATALAKE_ROOT"),
                              reason="PSA_DATALAKE_ROOT 없음")


@needs_s2
def test_real_s2_store_answers():
    b = store.open_backend("s2")
    # "Attention Is All You Need"
    cid = b.resolve([{"arxiv": "1706.03762"}])[0]
    assert cid is not None
    p = b.papers([cid])[cid]
    assert "attention" in p["title"].lower() and p["year"] == 2017
    h = b.neighbours([cid], "references", cap=5, cutoff=dt.date(2018, 1, 1))
    assert 0 < len(h.edges[cid]) <= 5


@needs_dl
def test_real_datalake_store_answers():
    b = store.open_backend("datalake")
    doi = b.con.execute("SELECT cited_doi FROM g.edges LIMIT 1").fetchone()[0]
    h = b.neighbours([doi], "cited_by", cap=3, cutoff=dt.date(2026, 1, 1))
    assert h.stats[doi].total >= 1
