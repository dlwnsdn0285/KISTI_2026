"""Store access: paper metadata, abstracts, one citation hop, id resolution.

Two stores behind one interface, chosen with open_backend(name):
  s2        DuckDB files built from the Semantic Scholar bulk data; key = corpusId.
  datalake  KISTI science datalake DuckDB files; key = DOI, years only.
The cutoff rule (drop papers after the query date) is applied here. Connections are read-only.
Paths and DuckDB limits come from PSA_* environment variables (see StoreConfig.from_env).
"""
from __future__ import annotations

import datetime as dt
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import duckdb

BACKENDS = ("s2", "datalake")


# ============================================================================
# identifiers and title normalisation
# ============================================================================

_DOI_PREFIX_RE = re.compile(r'^https?://(dx\.)?doi\.org/', re.IGNORECASE)
# new-style arXiv id
_ARXIV_NEW = re.compile(r'^(\d{4}\.\d{4,5})(v\d+)?$')
# old-style arXiv id
_ARXIV_OLD = re.compile(r'^([a-z-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?$', re.IGNORECASE)
_PUNCT_RE = re.compile(r'[^\w\s]', re.UNICODE)
_WS_RE = re.compile(r'\s+')
_ARXIV_DOI_PREFIX = "10.48550/arxiv."


def norm_doi(doi):
    """Lowercase, strip, drop the https://doi.org/ prefix."""
    if not doi:
        return None
    d = doi.strip().lower()
    d = _DOI_PREFIX_RE.sub('', d)
    return d or None


def normalize_arxiv_id(raw):
    """arXiv id without prefix and version suffix, or None."""
    if not raw:
        return None
    s = re.sub(r'^arxiv:', '', raw.strip(), flags=re.IGNORECASE)
    m = _ARXIV_NEW.match(s)
    if m:
        return m.group(1)
    m = _ARXIV_OLD.match(s)
    if m:
        return m.group(1).lower()
    return None


def arxiv_id_to_doi(raw):
    """arXiv id -> 10.48550/arxiv.<id>."""
    norm = normalize_arxiv_id(raw)
    return f"{_ARXIV_DOI_PREFIX}{norm}" if norm else None


def arxiv_of_doi(doi):
    """10.48550/arxiv.<id> -> arXiv id, or None."""
    if doi and doi.lower().startswith(_ARXIV_DOI_PREFIX):
        return normalize_arxiv_id(doi[len(_ARXIV_DOI_PREFIX):])
    return None


def normalize_title(title):
    """Title for exact matching: lowercase, punctuation to spaces, single spaces."""
    if not title:
        return ""
    t = title.replace('İ', 'i').lower()
    t = _PUNCT_RE.sub(' ', t)
    t = _WS_RE.sub(' ', t).strip()
    return t


def years_before(d: dt.date, n: int) -> dt.date:
    """Same day n years before d."""
    try:
        return d.replace(year=d.year - n)
    except ValueError:
        return d.replace(year=d.year - n, day=28)


# ============================================================================
# config
# ============================================================================

@dataclass
class StoreConfig:
    """Store file paths and DuckDB limits."""
    s2_graph_db: str | None = None
    s2_abstracts_db: str | None = None
    datalake_root: str | None = None
    sidecar_db: str | None = None
    duckdb_memory: str = "8GB"
    duckdb_threads: int = 8

    @classmethod
    def from_env(cls):
        """Read from PSA_* environment variables."""
        env = os.environ.get
        return cls(
            s2_graph_db=env("PSA_S2_GRAPH_DB") or None,
            s2_abstracts_db=env("PSA_S2_ABSTRACTS_DB") or None,
            datalake_root=env("PSA_DATALAKE_ROOT") or None,
            sidecar_db=env("PSA_SIDECAR_DB") or None,
            duckdb_memory=env("PSA_DUCKDB_MEMORY") or "8GB",
            duckdb_threads=int(env("PSA_DUCKDB_THREADS") or 8),
        )

    @classmethod
    def coerce(cls, config):
        """Build from None (environment), a dict, or an object with the same attributes."""
        if config is None:
            return cls.from_env()
        if isinstance(config, cls):
            return config
        base = cls.from_env()
        get = config.get if isinstance(config, dict) else (
            lambda k, d=None: getattr(config, k, d))
        for name in cls.__dataclass_fields__:
            value = get(name, None)
            if value is not None:
                setattr(base, name, value)
        base.duckdb_threads = int(base.duckdb_threads)
        return base


def _need(path, what, env):
    if not path:
        raise RuntimeError(f"{what} 경로가 없습니다. 환경변수 {env}를 설정하세요.")
    if not Path(path).exists():
        raise FileNotFoundError(f"{what}가 없습니다: {path} (환경변수 {env})")
    return str(path)


def _limits(con, cfg):
    # cap memory per connection so parallel processes do not exceed RAM
    con.execute(f"SET memory_limit='{cfg.duckdb_memory}'")
    con.execute(f"SET threads={int(cfg.duckdb_threads)}")


# ============================================================================
# result types
# ============================================================================

@dataclass
class HopStats:
    """How one parent's neighbours were reduced at each step."""
    total: int = 0            # neighbours in the store
    after_cutoff: int = 0     # dropped: after cutoff
    filtered: int = 0         # dropped: window / citation filters
    truncated: int = 0        # dropped: per-parent cap
    produced: int = 0         # returned

    def as_dict(self):
        return {"total": self.total, "after_cutoff": self.after_cutoff,
                "filtered": self.filtered, "truncated": self.truncated,
                "produced": self.produced}


@dataclass
class Hop:
    """Result of neighbours()."""
    edges: dict = field(default_factory=dict)   # parent -> [neighbour, ...]
    stats: dict = field(default_factory=dict)   # parent -> HopStats
    #: candidate -> citations up to the cutoff
    counts: dict = field(default_factory=dict)

    def reached(self):
        """Number of parents with at least one neighbour before filtering."""
        return sum(1 for s in self.stats.values() if s.total)


def _paper(key, title=None, year=None, pubdate=None, citationcount=None,
           doi=None, arxiv=None, abstract=None):
    """Paper dict returned by papers()."""
    return {"corpusid": key, "title": title or "", "year": year,
            "pubdate": pubdate, "citationcount": citationcount, "doi": doi,
            "arxiv": arxiv, "abstract": abstract or None}


# ============================================================================
# S2 store: SQL
# ============================================================================
# Tables (built by prepare_data.py):
#   citing(citing, cited, infl, intents)   sorted by citing
#   cited (citing, cited, infl, intents)   sorted by cited
#   idmap(corpusid, doi, arxiv, year, pubdate, citationcount, title, venue, authors)
#   doi2corpus(doi, corpusid, year), arxiv2corpus(arxiv, corpusid)
#   ab.abstracts(corpusid, abstract)       s2_abstracts.duckdb attached as ab
# No indexes: each table is queried by its sort column only, in batches.

_ABSTRACTS_SCHEMA = "ab"
# references follow the citing table, cited_by the cited table
_HOP_TABLES = {"references": ("citing", "citing", "cited"),
               "cited_by": ("cited", "cited", "citing")}
_S2_META_FIELDS = ("title", "year", "pubdate", "doi", "arxiv", "citationcount")


def s2_connect(cfg):
    """Read-only connection to s2_graph.duckdb (abstracts attached as ab if present)."""
    con = duckdb.connect(_need(cfg.s2_graph_db, "s2_graph.duckdb", "PSA_S2_GRAPH_DB"),
                         read_only=True)
    _limits(con, cfg)
    path = cfg.s2_abstracts_db
    if path and Path(path).exists():
        try:
            con.execute(f"ATTACH '{path}' AS {_ABSTRACTS_SCHEMA} (READ_ONLY)")
        except duckdb.Error:
            pass                      # loading or locked
    return con


def _after(p, alias="m."):
    """SQL: this neighbour is certainly after cutoff p."""
    return (f"coalesce({alias}pubdate > {p} OR "
            f"({alias}pubdate IS NULL AND {alias}year > year({p}::DATE)), false)")


def _ints(xs):
    return [int(x) for x in xs] if xs else []


def _group(ids, rows):
    out = {i: [] for i in ids}
    for k, v in rows:
        out[k].append(v)
    return out


def s2_meta(con, corpusids):
    """corpusId -> metadata row."""
    ids = _ints(corpusids)
    if not ids:
        return {}
    rows = con.execute(
        f"SELECT corpusid, {', '.join(_S2_META_FIELDS)} FROM idmap "
        "WHERE corpusid IN (SELECT unnest($1::INTEGER[]))", [ids]).fetchall()
    return {r[0]: dict(zip(_S2_META_FIELDS, r[1:])) for r in rows}


def s2_abstracts(con, corpusids):
    """corpusId -> abstract."""
    ids = _ints(corpusids)
    if not ids:
        return {}
    try:
        con.execute(f"SELECT 1 FROM {_ABSTRACTS_SCHEMA}.abstracts LIMIT 0")
    except duckdb.Error:
        raise RuntimeError(
            "초록 저장소(s2_abstracts.duckdb)를 열 수 없습니다. "
            "PSA_S2_ABSTRACTS_DB를 확인하거나 prepare_data.py --backend s2 --step load-abstracts를 "
            "먼저 실행하세요.") from None
    return dict(con.execute(
        f"SELECT corpusid, abstract FROM {_ABSTRACTS_SCHEMA}.abstracts "
        "WHERE corpusid IN (SELECT unnest($1::INTEGER[]))", [ids]).fetchall())


def s2_references(con, corpusids, *, before=None):
    """corpusId -> corpusIds it cites (minus those after `before`)."""
    ids = _ints(corpusids)
    if not ids:
        return {}
    table, key, other = _HOP_TABLES["references"]
    if before is None:
        sql = (f"SELECT {key}, {other} FROM {table} "
               f"WHERE {key} IN (SELECT unnest($1::INTEGER[])) "
               f"ORDER BY {key}, {other}")
        return _group(ids, con.execute(sql, [ids]).fetchall())
    # faster than joining idmap per edge
    sql = f"""WITH e AS (
            SELECT {key} AS k, {other} AS nb, infl FROM {table}
            WHERE {key} IN (SELECT unnest($1::INTEGER[])) ),
        late AS (SELECT corpusid FROM idmap
                 WHERE corpusid IN (SELECT DISTINCT nb FROM e)
                   AND {_after("$2", "")})
        SELECT k, nb FROM (SELECT k, nb, infl FROM e WHERE nb NOT IN (SELECT corpusid FROM late)) ORDER BY k, nb"""
    return _group(ids, con.execute(sql, [ids, before]).fetchall())


def s2_indegree_before(con, corpusids, *, before):
    """corpusId -> number of papers citing it up to `before`."""
    if not isinstance(before, dt.date):
        raise TypeError(f"before는 datetime.date여야 합니다: {before!r}")
    ids = _ints(corpusids)
    if not ids:
        return {}
    out = dict.fromkeys(ids, 0)
    rows = con.execute("""
        WITH e AS (SELECT cited AS k, citing AS nb FROM cited
                   WHERE cited IN (SELECT unnest($1::INTEGER[]))),
             dated AS (SELECT corpusid FROM idmap
                       WHERE corpusid IN (SELECT DISTINCT nb FROM e)
                         AND pubdate IS NOT NULL AND pubdate <= $2)
        SELECT e.k, count(*) FROM e JOIN dated d ON d.corpusid = e.nb
        GROUP BY e.k""", [ids, before]).fetchall()
    out.update(dict(rows))
    return out


def s2_resolve(con, rows):
    """[{doi, arxiv}] -> [corpusId | None], in input order."""
    out = [None] * len(rows)
    pending = list(range(len(rows)))
    for field_name, table, col, norm in (
            ("doi", "doi2corpus", "doi", norm_doi),
            ("arxiv", "arxiv2corpus", "arxiv", normalize_arxiv_id)):
        keyed = {}
        for i in pending:
            v = norm((rows[i] or {}).get(field_name))
            if v:
                keyed.setdefault(v, []).append(i)
        if not keyed:
            continue
        hits = {}
        for k, cid in con.execute(
                f"SELECT {col}, corpusid FROM {table} "
                f"WHERE {col} IN (SELECT unnest($1::VARCHAR[]))",
                [list(keyed)]).fetchall():
            hits.setdefault(k, []).append(cid)
        for k, idxs in keyed.items():
            if k in hits:
                for i in idxs:
                    out[i] = min(hits[k])
        pending = [i for i in pending if out[i] is None]
    return out


def _s2_cutoff_sql(cutoff):
    if cutoff is None:
        return "FALSE"
    return _after(f"DATE '{cutoff.isoformat()}'")


def _s2_window_sql(cutoff, window_years):
    """SQL: true if outside the year window."""
    if window_years is None or cutoff is None:
        return "FALSE"
    lo = years_before(cutoff, window_years)
    clause = (f"NOT COALESCE(m.pubdate >= DATE '{lo.isoformat()}', "
              f"m.pubdate IS NULL AND m.year >= {cutoff.year - window_years}, "
              f"FALSE)")
    return f"({clause})"


def s2_neighbours(con, parents, direction, *, cap=None, cutoff=None,
                  window_years=None, min_indegree_before=None):
    """One hop: cutoff and filters first, then the per-parent cap."""
    if min_indegree_before is not None:
        return _s2_neighbours_indegree(
            con, parents, direction, cap=cap, cutoff=cutoff,
            window_years=window_years, floor=min_indegree_before)
    if direction not in _HOP_TABLES:
        raise ValueError(f"direction은 {tuple(_HOP_TABLES)} 중 하나: {direction!r}")
    if cap is not None and cap < 1:
        raise ValueError(f"cap은 1 이상이거나 None: {cap}")
    ids = [int(p) for p in dict.fromkeys(parents)]
    hop = Hop()
    if not ids:
        return hop

    table, key, other = _HOP_TABLES[direction]
    excluded = _s2_cutoff_sql(cutoff)
    filtered = _s2_window_sql(cutoff, window_years)
    # read edges first, then idmap once per distinct neighbour
    rank = "row_number() OVER (PARTITION BY k, keep ORDER BY infl DESC, n)"
    sql = f"""
        WITH e AS (
            SELECT {key} AS k, {other} AS nb, infl FROM {table}
            WHERE {key} IN (SELECT unnest($1::INTEGER[]))),
        m AS (SELECT corpusid, year, pubdate, citationcount FROM idmap
              WHERE corpusid IN (SELECT DISTINCT nb FROM e))
        SELECT k, n, keep, rn, total, cut_n, filt_n, surv FROM (
            SELECT k, n, keep, infl, year, citationcount,
                   {rank} AS rn,
                   count(*) OVER w AS total,
                   sum(cut::INT) OVER w AS cut_n,
                   sum((NOT cut AND excl)::INT) OVER w AS filt_n,
                   sum((NOT cut AND NOT excl)::INT) OVER w AS surv
            FROM (
                SELECT e.k AS k, e.nb AS n, e.infl AS infl,
                       m.year AS year, m.citationcount AS citationcount,
                       {excluded} AS cut, {filtered} AS excl,
                       NOT ({excluded}) AND NOT ({filtered}) AS keep
                FROM e LEFT JOIN m ON m.corpusid = e.nb)
            WINDOW w AS (PARTITION BY k))
        WHERE (keep AND rn <= $2) OR (NOT keep AND rn = 1)
        ORDER BY k, rn"""
    limit = cap if cap is not None else 2 ** 31 - 1
    for k, n, keep, _rn, total, cut_n, filt_n, surv in con.execute(sql, [ids, limit]).fetchall():
        stat = hop.stats.get(k)
        if stat is None:
            stat = hop.stats[k] = HopStats(
                total=total, after_cutoff=cut_n, filtered=filt_n,
                truncated=max(0, surv - limit))
        if keep:
            hop.edges.setdefault(k, []).append(n)
            stat.produced += 1
    return hop


def _s2_neighbours_indegree(con, parents, direction, *, cap, cutoff,
                            window_years, floor):
    """One hop with a citation floor: cutoff / window -> floor -> cap."""
    if floor < 1:
        raise ValueError(f"min_indegree_before는 1 이상이거나 None: {floor}")
    hop = s2_neighbours(con, parents, direction, cap=None, cutoff=cutoff,
                        window_years=window_years)
    if not hop.edges:
        return hop

    survivors = {n for children in hop.edges.values() for n in children}
    counts = s2_indegree_before(con, sorted(survivors), before=cutoff)
    passed = {n for n in survivors if counts.get(n, 0) >= floor}

    table, key, other = _HOP_TABLES[direction]
    flags = {}
    for k, n, infl in con.execute(
            f"SELECT {key}, {other}, infl FROM {table} "
            f"WHERE {key} IN (SELECT unnest($1::INTEGER[]))",
            [list(hop.edges)]).fetchall():
        if n in passed:
            flags[(k, n)] = bool(infl) or flags.get((k, n), False)

    out = Hop(stats=hop.stats, counts=counts)
    for parent, children in hop.edges.items():
        stat = hop.stats[parent]
        kept = [n for n in children if n in passed]
        stat.filtered += len(children) - len(kept)
        kept.sort(key=lambda n: (not flags.get((parent, n), False), n))
        if cap is not None and len(kept) > cap:
            stat.truncated += len(kept) - cap
            kept = kept[:cap]
        stat.produced = len(kept)
        if kept:
            out.edges[parent] = kept
    return out


def s2_links(con, ids, cutoff):
    """Citation edges inside a candidate set (both directions)."""
    idset = set(ids)
    refs = s2_references(con, sorted(idset), before=cutoff)
    deg = dict.fromkeys(idset, 0)
    for n, lst in refs.items():
        for m in lst:
            if m in idset and m != n:
                deg[m] += 1
                deg[n] += 1
    return deg


# ============================================================================
# KISTI datalake store: SQL
# ============================================================================
# Three files attached read-only on one connection:
#   g = search/citation_graph.duckdb   edges(citing_doi, cited_doi)
#   m = search/paper_meta.duckdb       meta(doi, title, year, abstract, cited_by_count, ...)
#   s = sidecar.duckdb                 meta, edges (optional supplement)
# Datalake rows win over sidecar rows for the same DOI.

_DL_YEAR = """
    SELECT n.k AS doi,
           CASE WHEN any_value(mm.doi) IS NOT NULL THEN any_value(mm.year)
                ELSE any_value(sm.year) END AS year
    FROM {src} n LEFT JOIN m.meta mm ON mm.doi = n.k
                 LEFT JOIN s.meta sm ON sm.doi = n.k
    GROUP BY n.k"""


def datalake_connect(cfg):
    """Read-only datalake connection (schemas g, m, s)."""
    root = Path(_need(cfg.datalake_root, "science_datalake 폴더", "PSA_DATALAKE_ROOT"))
    graph = _need(root / "search" / "citation_graph.duckdb",
                  "citation_graph.duckdb", "PSA_DATALAKE_ROOT")
    meta = _need(root / "search" / "paper_meta.duckdb",
                 "paper_meta.duckdb", "PSA_DATALAKE_ROOT")
    con = duckdb.connect()
    _limits(con, cfg)
    con.execute(f"ATTACH '{graph}' AS g (READ_ONLY)")
    con.execute(f"ATTACH '{meta}' AS m (READ_ONLY)")
    if cfg.sidecar_db and Path(cfg.sidecar_db).exists():
        con.execute(f"ATTACH '{cfg.sidecar_db}' AS s (READ_ONLY)")
    else:
        con.execute("ATTACH ':memory:' AS s")
        con.execute("CREATE TABLE s.meta(doi VARCHAR, title VARCHAR, year INTEGER, "
                    "abstract VARCHAR, cited_by_count BIGINT)")
        con.execute("CREATE TABLE s.edges(citing_doi VARCHAR, cited_doi VARCHAR)")
    return con


def datalake_papers(con, dois):
    """DOI -> (doi, title, year, abstract, cited_by_count)."""
    keys = list(dict.fromkeys(d for d in (norm_doi(x) for x in dois) if d))
    if not keys:
        return {}
    cols = "doi, title, year, abstract, cited_by_count"
    out = {r[0]: r for r in con.execute(
        f"SELECT {cols} FROM m.meta WHERE doi IN (SELECT unnest($1::VARCHAR[]))",
        [keys]).fetchall()}
    missing = [d for d in keys if d not in out]
    if missing:
        for r in con.execute(
                f"SELECT {cols} FROM s.meta WHERE doi IN (SELECT unnest($1::VARCHAR[]))",
                [missing]).fetchall():
            out[r[0]] = r
    return out


def datalake_edges(con, keys, direction):
    """(parent, neighbour, neighbour year) from datalake and sidecar edges."""
    key, other = (("citing_doi", "cited_doi") if direction == "references"
                  else ("cited_doi", "citing_doi"))
    sql = f"""
        WITH p AS (SELECT DISTINCT unnest($1::VARCHAR[]) AS k),
        e AS (SELECT {key} AS k, {other} AS nb FROM g.edges WHERE {key} IN (SELECT k FROM p)
              UNION
              SELECT {key}, {other} FROM s.edges WHERE {key} IN (SELECT k FROM p)),
        nbs AS (SELECT DISTINCT nb AS k FROM e WHERE nb IS NOT NULL),
        y AS ({_DL_YEAR.format(src="nbs")})
        SELECT e.k, e.nb, y.year FROM e JOIN y ON y.doi = e.nb
        WHERE e.nb <> e.k"""
    return con.execute(sql, [list(keys)]).fetchall()


def datalake_indegree(con, keys, year_max):
    """DOI -> number of citing papers with year <= year_max."""
    keys = list(dict.fromkeys(keys))
    out = dict.fromkeys(keys, 0)
    if not keys:
        return out
    sql = f"""
        WITH p AS (SELECT DISTINCT unnest($1::VARCHAR[]) AS k),
        e AS (SELECT cited_doi AS k, citing_doi AS nb FROM g.edges WHERE cited_doi IN (SELECT k FROM p)
              UNION
              SELECT cited_doi, citing_doi FROM s.edges WHERE cited_doi IN (SELECT k FROM p)),
        nbs AS (SELECT DISTINCT nb AS k FROM e WHERE nb IS NOT NULL),
        y AS ({_DL_YEAR.format(src="nbs")})
        SELECT e.k, count(*) FROM e JOIN y ON y.doi = e.nb
        WHERE y.year IS NOT NULL AND y.year <= $2 AND e.nb <> e.k
        GROUP BY e.k"""
    for k, n in con.execute(sql, [keys, int(year_max)]).fetchall():
        out[k] = n
    return out


def datalake_neighbours(con, parents, direction, *, cap=None, cutoff=None,
                        window_years=None, min_indegree_before=None):
    """One hop (datalake): cutoff -> window -> citation floor -> cap, all by year."""
    if direction not in ("references", "cited_by"):
        raise ValueError(f"direction은 references / cited_by 중 하나: {direction!r}")
    parents = list(dict.fromkeys(parents))
    hop = Hop()
    if not parents:
        return hop
    ymax = cutoff.year if cutoff is not None else None
    lo = (cutoff.year - window_years
          if (window_years is not None and cutoff is not None) else None)
    by_parent, pass_cut = {}, set()
    for k, nb, year in datalake_edges(con, parents, direction):
        stat = hop.stats.setdefault(k, HopStats())
        stat.total += 1
        if ymax is not None and year is not None and year > ymax:
            stat.after_cutoff += 1
            continue
        if lo is not None and (year is None or year < lo):
            stat.filtered += 1
            continue
        by_parent.setdefault(k, []).append(nb)
        pass_cut.add(nb)
    need_count = cap is not None or min_indegree_before is not None
    counts = (datalake_indegree(con, sorted(pass_cut), ymax if ymax is not None else 9999)
              if need_count and pass_cut else {})
    if min_indegree_before is not None:
        hop.counts = counts
    for k, kids in by_parent.items():
        stat = hop.stats[k]
        if min_indegree_before is not None:
            kept = [n for n in kids if counts.get(n, 0) >= min_indegree_before]
            stat.filtered += len(kids) - len(kept)
            kids = kept
        kids = sorted(set(kids), key=lambda n: (-counts.get(n, 0), n))
        if cap is not None and len(kids) > cap:
            stat.truncated += len(kids) - cap
            kids = kids[:cap]
        stat.produced = len(kids)
        if kids:
            hop.edges[k] = kids
    return hop


def datalake_links(con, ids, cutoff):
    """Citation edges inside a candidate set (both directions)."""
    idset = list(dict.fromkeys(ids))
    deg = dict.fromkeys(idset, 0)
    if not idset:
        return deg
    sql = f"""
        WITH p AS (SELECT DISTINCT unnest($1::VARCHAR[]) AS k),
        e AS (SELECT citing_doi AS a, cited_doi AS b FROM g.edges
              WHERE citing_doi IN (SELECT k FROM p) AND cited_doi IN (SELECT k FROM p)
              UNION
              SELECT citing_doi, cited_doi FROM s.edges
              WHERE citing_doi IN (SELECT k FROM p) AND cited_doi IN (SELECT k FROM p)),
        bs AS (SELECT DISTINCT b AS k FROM e),
        y AS ({_DL_YEAR.format(src="bs")})
        SELECT e.a, e.b FROM e JOIN y ON y.doi = e.b
        WHERE e.a <> e.b AND (y.year IS NULL OR y.year <= $2)"""
    for a, b in con.execute(sql, [idset, int(cutoff.year)]).fetchall():
        deg[a] += 1
        deg[b] += 1
    return deg


# ============================================================================
# backends (the interface the pipeline uses)
# ============================================================================

# one lock per backend: a DuckDB connection is not thread-safe
_LOCKS = {"s2": threading.RLock(), "datalake": threading.RLock()}


class _Backend:
    name = "base"
    harvest_order = ""

    def __init__(self, con, meter=None):
        self.con = con
        self.meter = meter
        self._papers = {}     # per-run cache

    def _metered(self, fn, n=1):
        """Run fn under the lock and record its time."""
        if self.meter is None:
            with _LOCKS[self.name]:
                return fn()
        t0 = time.perf_counter()
        with _LOCKS[self.name]:
            waited = time.perf_counter() - t0
            with self.meter.call("local_db", n=n):
                result = fn()
        self.meter.count("local_db_wait_ms", int(waited * 1000))
        return result

    def prefetch(self, keys):
        """Read many papers at once into the cache."""
        self.papers(keys)


class S2Backend(_Backend):
    """S2 store. Key: corpusId (int); dates are daily."""

    name = "s2"
    harvest_order = "influential_first"

    def papers(self, corpusids):
        """corpusId -> paper dict (with abstract)."""
        ids = [int(c) for c in corpusids]
        if not ids:
            return {}
        missing = [c for c in dict.fromkeys(ids) if c not in self._papers]
        if missing:
            meta = self._metered(lambda: s2_meta(self.con, missing))
            abstracts = self._metered(lambda: s2_abstracts(self.con, missing))
            for cid in missing:
                m = meta.get(cid)
                self._papers[cid] = (_paper(cid) if m is None else _paper(
                    cid, title=m["title"], year=m["year"], pubdate=m["pubdate"],
                    citationcount=m["citationcount"], doi=m["doi"], arxiv=m["arxiv"],
                    abstract=abstracts.get(cid)))
        return {cid: self._papers[cid] for cid in ids}

    def after_cutoff(self, paper, cutoff):
        """True if the paper is after the cutoff; unknown dates are not."""
        if cutoff is None:
            return False
        if paper.get("pubdate") is not None:
            return paper["pubdate"] > cutoff
        year = paper.get("year")
        return year is not None and year > cutoff.year

    def neighbours(self, parents, direction, *, cap=None, cutoff=None,
                   window_years=None, min_indegree_before=None):
        """One hop (see s2_neighbours)."""
        return self._metered(
            lambda: s2_neighbours(self.con, parents, direction, cap=cap,
                                  cutoff=cutoff, window_years=window_years,
                                  min_indegree_before=min_indegree_before),
            n=len(parents))

    def indegree_before(self, corpusids, cutoff):
        """Citations up to the cutoff."""
        return self._metered(lambda: s2_indegree_before(
            self.con, list(corpusids), before=cutoff))

    def links(self, ids, cutoff):
        """Citation edges inside a candidate set."""
        return s2_links(self.con, ids, cutoff)

    def key_of(self, hit):
        """S2 search result -> corpusId."""
        cid = hit.get("corpusId")
        return int(cid) if cid is not None else None

    def resolve(self, rows):
        """[{doi, arxiv}] -> [corpusId | None]."""
        return self._metered(lambda: s2_resolve(self.con, rows))


class DatalakeBackend(_Backend):
    """KISTI datalake store. Key: DOI (str); years only."""

    name = "datalake"
    harvest_order = "indegree_before_desc"

    def papers(self, keys):
        """DOI -> paper dict."""
        keys = list(keys)
        if not keys:
            return {}
        missing = [k for k in dict.fromkeys(keys) if k not in self._papers]
        if missing:
            got = self._metered(lambda: datalake_papers(self.con, missing))
            for k in missing:
                r = got.get(k)
                p = (_paper(k) if r is None else _paper(
                    k, title=r[1], year=r[2], citationcount=r[4] or 0,
                    doi=r[0], arxiv=arxiv_of_doi(r[0]), abstract=r[3]))
                p["doi"] = p["doi"] or k
                p["arxiv"] = p["arxiv"] or arxiv_of_doi(k)
                self._papers[k] = p
        return {k: self._papers[k] for k in keys}

    def after_cutoff(self, paper, cutoff):
        """True if the paper's year is after the cutoff year."""
        if cutoff is None:
            return False
        year = paper.get("year")
        return year is not None and year > cutoff.year

    def neighbours(self, parents, direction, *, cap=None, cutoff=None,
                   window_years=None, min_indegree_before=None):
        """One hop (see datalake_neighbours)."""
        return self._metered(
            lambda: datalake_neighbours(self.con, parents, direction, cap=cap,
                                        cutoff=cutoff, window_years=window_years,
                                        min_indegree_before=min_indegree_before),
            n=len(parents))

    def indegree_before(self, keys, cutoff):
        """Citations up to the cutoff year."""
        return self._metered(lambda: datalake_indegree(self.con, list(keys), cutoff.year))

    def links(self, ids, cutoff):
        """Citation edges inside a candidate set."""
        return datalake_links(self.con, ids, cutoff)

    def key_of(self, hit):
        """S2 search result -> DOI (arXiv DOI if only an arXiv id exists)."""
        ext = hit.get("externalIds") or {}
        d = norm_doi(ext.get("DOI"))
        if d:
            return d
        a = normalize_arxiv_id(ext.get("ArXiv"))
        return norm_doi(arxiv_id_to_doi(a)) if a else None


def open_backend(name, *, meter=None, config=None, con=None):
    """Open a store: "s2" or "datalake"."""
    cfg = StoreConfig.coerce(config)
    if name == "s2":
        return S2Backend(con if con is not None else s2_connect(cfg), meter)
    if name == "datalake":
        return DatalakeBackend(con if con is not None else datalake_connect(cfg), meter)
    raise ValueError(f"백엔드는 {BACKENDS} 중 하나: {name!r}")
