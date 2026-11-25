"""Data preparation for one backend.

    python prepare_data.py --backend datalake --datalake-root /path/to/science_datalake
    python prepare_data.py --backend s2 --data-dir /data/psa

datalake: the package is used as it is; this only checks that the files are there.
s2: downloads the Semantic Scholar bulk data (484GB, about 30 hours) and builds
    s2_graph.duckdb (67GB) and s2_abstracts.duckdb (108GB). Finished steps are skipped on re-run.
    The S2 API key is read from S2_API_KEY.
"""
import argparse
import json
import os
import shutil
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

S2_RELEASE = "2026-09-01"
S2_DATASETS = ["paper-ids", "abstracts", "papers", "citations"]   # smallest first
UA = "paper-search-agent/1.0"


def _paths(args):
    """Default paths under --data-dir; environment variables win."""
    root = Path(args.data_dir)
    env = os.environ.get
    return {
        "bulk": root / "s2_bulk",
        "graph": Path(env("PSA_S2_GRAPH_DB") or root / "s2_graph.duckdb"),
        "abstracts": Path(env("PSA_S2_ABSTRACTS_DB") or root / "s2_abstracts.duckdb"),
        "sidecar": Path(env("PSA_SIDECAR_DB") or root / "sidecar.duckdb"),
    }


def _duck(path, args):
    import duckdb
    con = duckdb.connect(str(path))
    con.execute(f"SET threads={args.threads}; SET preserve_insertion_order=false;"
                f" SET memory_limit='{args.memory}';")
    return con


# ============================================================================
# 1. download
# ============================================================================
# Resumable: re-running skips files that are already complete.
# Presigned URLs reject HEAD and expire, so sizes come from a 1-byte Range GET
# and the file list is fetched again on 400/401/403.

_MIN_FREE = 60e9
_STALL_FLOOR = 100e3     # abort and retry below this speed (bytes/s)
_STALL_GRACE = 300       # grace period (s)


def _fname(url):
    """File name from a presigned URL."""
    return url.split("?")[0].rsplit("/", 1)[-1]


def _listing(name, key, release):
    api = f"https://api.semanticscholar.org/datasets/v1/release/{release}/dataset/{name}"
    req = urllib.request.Request(api, headers={"User-Agent": UA, "x-api-key": key})
    last = None
    for attempt in range(7):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)["files"]
        except Exception as e:                      # noqa: BLE001
            last = e
            time.sleep(min(90, 15 * (attempt + 1)))
    raise RuntimeError(f"{name} 파일 목록을 받지 못했습니다: {last}")


class _Urls:
    """Presigned URLs of one dataset; refreshed when they expire."""

    def __init__(self, name, key, release):
        self.name, self.key, self.release = name, key, release
        self.lock = threading.Lock()
        self.at = time.time()
        self.by_name = {_fname(u): u for u in _listing(name, key, release)}

    def refresh(self, fname, seen_at):
        """Refresh unless another thread already did."""
        with self.lock:
            if self.at > seen_at:
                return self.by_name[fname], self.at
            self.by_name.update({_fname(u): u for u in
                                 _listing(self.name, self.key, self.release)})
            self.at = time.time()
            return self.by_name[fname], self.at


def _remote_size(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        cr = r.headers.get("Content-Range") or ""
        return int(cr.rsplit("/", 1)[-1]) if "/" in cr else 0


def _fetch(url, path, expected):
    """Resume with a Range request; returns the bytes fetched."""
    have = path.stat().st_size if path.exists() else 0
    if expected and have == expected:
        return 0
    if expected and have > expected:
        path.unlink()
        have = 0
    headers = {"User-Agent": UA}
    if have:
        headers["Range"] = f"bytes={have}-"
    t0 = time.time()
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                timeout=300) as r, \
            open(path, "ab" if have else "wb") as f:
        moved = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            moved += len(chunk)
            el = time.time() - t0
            if el > _STALL_GRACE and moved / el < _STALL_FLOOR:
                raise TimeoutError(f"{path.name}: {moved/el/1e3:.0f}KB/s로 늘어져 재시도")
    return moved


def _download_dry_run(args, key):
    """Print file counts and total size without downloading."""
    total = 0
    for name in args.only or S2_DATASETS:
        urls = list(_Urls(name, key, args.release).by_name.values())
        with ThreadPoolExecutor(args.parallel) as ex:
            sizes = list(ex.map(_remote_size, urls))
        total += sum(sizes)
        print(f"  {name:<10} {len(urls):>4}개  {sum(sizes)/1e9:>7.1f}GB", flush=True)
    print(f"  합계 {total/1e9:.1f}GB (release {args.release})")


def cmd_download_s2(args):
    """Download the S2 bulk datasets (resumable)."""
    key = os.environ.get("S2_API_KEY", "").strip().strip('"').strip("'")
    if not key:
        sys.exit("환경변수 S2_API_KEY가 없습니다.")
    dest = _paths(args)["bulk"]
    if args.dry_run:
        _download_dry_run(args, key)
        return
    dest.mkdir(parents=True, exist_ok=True)
    t0, grand, failures = time.time(), 0, []

    for name in args.only or S2_DATASETS:
        outdir = dest / name
        outdir.mkdir(exist_ok=True)
        urls = _Urls(name, key, args.release)
        fnames = sorted(urls.by_name)
        # cache verified sizes for re-runs
        cache_path = outdir / ".sizes.json"
        sizes = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        print(f"\n=== {name}: {len(fnames)}개 -> {outdir} ===", flush=True)
        state = {"n": 0, "bytes": 0, "t": time.time()}
        lock = threading.Lock()

        def one(fname, outdir=outdir, urls=urls, sizes=sizes, state=state,
                lock=lock, cache_path=cache_path, n_files=len(fnames), name=name):
            path = outdir / fname
            url, seen = urls.by_name[fname], urls.at
            for attempt in range(5):
                try:
                    want = sizes.get(fname)
                    if want and path.exists() and path.stat().st_size == want:
                        moved = 0
                    else:
                        want = _remote_size(url)
                        with lock:
                            sizes[fname] = want
                        moved = _fetch(url, path, want)
                    with lock:
                        state["n"] += 1
                        state["bytes"] += moved
                        if state["n"] % 10 == 0 or state["n"] == n_files:
                            run = time.time() - state["t"]
                            cache_path.write_text(json.dumps(sizes))
                            print(f"   {state['n']}/{n_files}  {state['bytes']/1e9:.1f}GB  "
                                  f"{state['bytes']/1e6/max(run, 1):.1f}MB/s", flush=True)
                    return
                except urllib.error.HTTPError as e:
                    if e.code in (400, 401, 403) and attempt < 4:   # URL expired
                        url, seen = urls.refresh(fname, seen)
                        continue
                    if attempt == 4:
                        with lock:
                            failures.append(f"{name}/{fname}: HTTP {e.code}")
                        return
                    time.sleep(10 * (attempt + 1))
                except Exception as e:                # noqa: BLE001
                    if attempt == 4:
                        with lock:
                            failures.append(f"{name}/{fname}: {type(e).__name__} {e}")
                        return
                    time.sleep(10 * (attempt + 1))

        if shutil.disk_usage(dest).free < _MIN_FREE:
            sys.exit("디스크 여유가 60GB보다 적어 멈춥니다.")
        with ThreadPoolExecutor(args.parallel) as ex:
            list(ex.map(one, fnames))
        cache_path.write_text(json.dumps(sizes))
        grand += state["bytes"]

    print(f"\n완료: 이번 실행 {grand/1e9:.1f}GB, {(time.time()-t0)/60:.0f}분")
    if failures:
        print(f"받지 못한 파일 {len(failures)}개 (다시 실행하면 이어받습니다):")
        for f in failures[:20]:
            print(f"   {f}")
        sys.exit(1)


# ============================================================================
# 2. load-graph
# ============================================================================
# Load only after all citation shards are downloaded: they are split by edge, not by paper.

_INTENT_BITS = {"background": 1, "methodology": 2, "result": 4}
_INTENT_HAS_CONTEXT = 8     # the edge had a citation context
# \p{L}\p{N} to match Python's normalize_title (RE2 \w is ASCII-only)
_SQL_NORM_TITLE = (r"trim(regexp_replace(regexp_replace(lower({col}), "
                   r"'[^\p{{L}}\p{{N}}_\s]', ' ', 'g'), '\s+', ' ', 'g'))")


def _shards(bulk, name, limit=None):
    shards = sorted((bulk / name).glob("*.gz"))
    return shards[:limit] if limit else shards


def _lit(paths):
    return "[" + ",".join(f"'{p}'" for p in paths) + "]"


def _say(label, t0, path=None):
    size = f"   DB {path.stat().st_size/1e9:>6.1f}GB" if path and path.exists() else ""
    print(f"  {label:<40}{time.time()-t0:>8.1f}s{size}", flush=True)


def _build_idmap(con, shards, out):
    """papers -> idmap, title_lookup, doi2corpus, arxiv2corpus, doi_multi."""
    t = time.time()
    con.execute(f"""CREATE OR REPLACE TABLE idmap AS
        SELECT corpusid::INTEGER                             AS corpusid,
               lower(nullif(trim(externalids.DOI), ''))      AS doi,
               nullif(trim(externalids.ArXiv), '')           AS arxiv,
               year::INTEGER                                 AS year,
               try_cast(publicationdate AS DATE)             AS pubdate,
               citationcount::INTEGER                        AS citationcount,
               nullif(trim(title), '')                       AS title,
               nullif(trim(venue), '')                       AS venue,
               list_transform(coalesce(authors, []), a -> a.name) AS authors
        FROM read_json({_lit(shards)}, columns={{
            corpusid:'BIGINT', year:'BIGINT', publicationdate:'VARCHAR',
            citationcount:'BIGINT', title:'VARCHAR', venue:'VARCHAR',
            authors:'STRUCT(name VARCHAR)[]',
            externalids:'STRUCT(DOI VARCHAR, ArXiv VARCHAR)'
        }}, format='newline_delimited')
        ORDER BY corpusid""")
    _say(f"idmap ({len(shards)}샤드)", t, out)

    t = time.time()
    con.execute(f"""CREATE OR REPLACE TABLE title_lookup AS
        SELECT {_SQL_NORM_TITLE.format(col="title")} AS title_norm, corpusid
        FROM idmap WHERE title IS NOT NULL AND length(title) > 0
        ORDER BY title_norm, corpusid""")
    con.execute("DELETE FROM title_lookup WHERE title_norm = ''")
    _say("title_lookup", t, out)

    # keep DOIs that map to several corpusIds
    t = time.time()
    con.execute("""CREATE OR REPLACE TABLE doi2corpus AS
        SELECT doi, corpusid, year FROM idmap WHERE doi IS NOT NULL
        ORDER BY doi, corpusid""")
    con.execute("""CREATE OR REPLACE TABLE arxiv2corpus AS
        SELECT arxiv, corpusid FROM idmap WHERE arxiv IS NOT NULL
        ORDER BY arxiv, corpusid""")
    con.execute("""CREATE OR REPLACE TABLE doi_multi AS
        SELECT doi, count(*) AS n_corpusid,
               list(corpusid ORDER BY corpusid) AS corpusids
        FROM doi2corpus GROUP BY doi HAVING count(*) > 1
        ORDER BY doi""")
    _say("doi2corpus / arxiv2corpus / doi_multi", t, out)


def _build_edges(con, shards, out, stage, batch):
    """citations -> citing, cited (same edges, sorted two ways)."""
    bits = " + ".join(f"CASE WHEN list_contains(f, '{k}') THEN {v} ELSE 0 END"
                      for k, v in _INTENT_BITS.items())
    con.execute(f"ATTACH '{stage}' AS stg")
    t0 = time.time()
    con.execute("""CREATE OR REPLACE TABLE stg.edges(
        citing INTEGER, cited INTEGER, infl BOOLEAN, intents UTINYINT)""")
    for i in range(0, len(shards), batch):
        part = shards[i:i + batch]
        # contexts are not read (most of the volume)
        con.execute(f"""INSERT INTO stg.edges
            SELECT citing, cited, infl,
                   ({bits} + CASE WHEN len(f) > 0 THEN {_INTENT_HAS_CONTEXT}
                                  ELSE 0 END)::UTINYINT
            FROM (
                SELECT citingcorpusid::INTEGER AS citing,
                       citedcorpusid::INTEGER  AS cited,
                       isinfluential::BOOLEAN  AS infl,
                       list_distinct(flatten(coalesce(intents, []))) AS f
                FROM read_json({_lit(part)}, columns={{
                    citingcorpusid:'BIGINT', citedcorpusid:'BIGINT',
                    isinfluential:'BOOLEAN', intents:'VARCHAR[][]'}},
                    format='newline_delimited')
                WHERE citedcorpusid IS NOT NULL AND citingcorpusid IS NOT NULL)""")
        print(f"  파싱 {min(i + batch, len(shards)):>4}/{len(shards)}샤드   "
              f"경과 {(time.time()-t0)/60:>5.1f}분", flush=True)

    # merge duplicate edges across shards
    t = time.time()
    con.execute("""CREATE OR REPLACE TABLE stg.uniq AS
        SELECT citing, cited, bool_or(infl) AS infl,
               bit_or(intents)::UTINYINT AS intents
        FROM stg.edges GROUP BY citing, cited""")
    _say("간선 중복 제거", t)
    t = time.time()
    con.execute("CREATE OR REPLACE TABLE citing AS SELECT * FROM stg.uniq ORDER BY citing, cited")
    _say("citing 정렬", t, out)
    t = time.time()
    con.execute("CREATE OR REPLACE TABLE cited AS SELECT * FROM stg.uniq ORDER BY cited, citing")
    _say("cited 정렬", t, out)
    con.execute("DETACH stg")
    stage.unlink(missing_ok=True)
    Path(str(stage) + ".wal").unlink(missing_ok=True)


def _check_titles_agree(con, n=5000):
    """Check on a sample that SQL and Python title normalisation agree."""
    from psa.store import normalize_title
    rows = con.execute(f"""SELECT title, {_SQL_NORM_TITLE.format(col='title')}
        FROM idmap WHERE title IS NOT NULL
        USING SAMPLE reservoir({n} ROWS) REPEATABLE (11)""").fetchall()
    bad = [(t, s) for t, s in rows if s != normalize_title(t)]
    print(f"  제목 정규화 SQL vs 파이썬: {len(rows)-len(bad)}/{len(rows)} 일치", flush=True)


def cmd_load_graph(args):
    """Load papers and citations shards into s2_graph.duckdb."""
    p = _paths(args)
    out, stage = p["graph"], p["graph"].with_name(p["graph"].stem + ".stage.duckdb")
    cit = _shards(p["bulk"], "citations", args.limit_shards)
    pap = _shards(p["bulk"], "papers", args.limit_shards)
    if not cit or not pap:
        sys.exit(f"샤드가 없습니다: {p['bulk']} (--step download를 먼저 실행)")
    print(f"citations {len(cit)}샤드   papers {len(pap)}샤드   -> {out}\n")
    stage.unlink(missing_ok=True)
    con = _duck(out, args)
    t0 = time.time()
    if args.only in (None, "idmap"):
        _build_idmap(con, pap, out)
        _check_titles_agree(con)
    if args.only in (None, "edges"):
        _build_edges(con, cit, out, stage, args.batch)
    print(f"\n=== 적재 완료  {(time.time()-t0)/60:.1f}분   {out.stat().st_size/1e9:.1f}GB ===")
    for t in ("citing", "cited", "idmap", "title_lookup", "doi2corpus",
              "arxiv2corpus", "doi_multi"):
        try:
            print(f"  {t:<16}{con.execute(f'SELECT count(*) FROM {t}').fetchone()[0]:>15,}")
        except Exception:                             # noqa: BLE001
            pass
    con.close()


# ============================================================================
# 3. load-abstracts
# ============================================================================

def cmd_load_abstracts(args):
    """Load abstracts shards into s2_abstracts.duckdb (replaces the file)."""
    p = _paths(args)
    out = p["abstracts"]
    shards = _shards(p["bulk"], "abstracts", args.limit_shards)
    if not shards:
        sys.exit(f"샤드가 없습니다: {p['bulk']}/abstracts")
    print(f"abstracts {len(shards)}샤드 -> {out}", flush=True)
    out.unlink(missing_ok=True)
    Path(str(out) + ".wal").unlink(missing_ok=True)
    con = _duck(out, args)
    t0 = time.time()
    con.execute(f"""CREATE TABLE abstracts AS
        SELECT corpusid::INTEGER AS corpusid,
               nullif(trim(abstract), '') AS abstract
        FROM read_json({_lit(shards)}, columns={{corpusid:'BIGINT', abstract:'VARCHAR'}},
            format='newline_delimited')
        WHERE corpusid IS NOT NULL AND abstract IS NOT NULL
        ORDER BY corpusid""")
    con.execute("CHECKPOINT")
    n = con.execute("SELECT count(*) FROM abstracts").fetchone()[0]
    print(f"=== 완료  {(time.time()-t0)/60:.1f}분   {out.stat().st_size/1e9:.1f}GB   {n:,}행 ===")
    con.close()


# ============================================================================
# 4. sidecar (optional)
# ============================================================================
# Papers missing from the datalake, collected from Crossref by DOI prefix.

_SIDECAR_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    doi VARCHAR PRIMARY KEY, title VARCHAR, title_norm VARCHAR, year INTEGER,
    abstract VARCHAR, cited_by_count BIGINT, field VARCHAR, subfield VARCHAR,
    crawled_at TIMESTAMP, crawl_source VARCHAR);
CREATE TABLE IF NOT EXISTS edges (citing_doi VARCHAR, cited_doi VARCHAR, edge_source VARCHAR);
CREATE INDEX IF NOT EXISTS idx_sidecar_edges_citing ON edges(citing_doi);
CREATE INDEX IF NOT EXISTS idx_sidecar_edges_cited ON edges(cited_doi);
CREATE INDEX IF NOT EXISTS idx_sidecar_meta_title_norm ON meta(title_norm);
"""


def _crossref_page(prefix, cursor, rows, until, mailto):
    params = {"rows": rows, "cursor": cursor,
              "filter": f"until-pub-date:{until},has-references:true"}
    if mailto:
        params["mailto"] = mailto
    url = (f"https://api.crossref.org/prefixes/{prefix}/works?"
           + urllib.parse.urlencode(params))
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    last = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)["message"]
        except Exception as e:                      # noqa: BLE001
            last = e
            time.sleep(min(60, 5 * 2 ** attempt))
    raise RuntimeError(f"Crossref 요청 실패: {last}")


def _crossref_parse(item):
    """One Crossref work -> (meta row, edges, reference count)."""
    from psa.store import norm_doi, normalize_title
    doi = norm_doi(item.get("DOI"))
    if not doi:
        return None, [], 0
    title = (item.get("title") or [None])[0]
    year = None
    for key in ("issued", "published", "published-print", "published-online"):
        parts = (item.get(key) or {}).get("date-parts") or []
        if parts and parts[0] and parts[0][0]:
            year = int(parts[0][0])
            break
    meta = (doi, title, normalize_title(title) if title else None, year,
            item.get("abstract"), item.get("is-referenced-by-count"))
    refs, seen, total = [], set(), 0
    for r in item.get("reference") or []:
        total += 1
        cited = norm_doi(r.get("DOI"))
        if cited and cited != doi and cited not in seen:
            seen.add(cited)
            refs.append((doi, cited))
    return meta, refs, total


def cmd_sidecar(args):
    """Collect one DOI prefix from Crossref into sidecar.duckdb (resumable)."""
    import duckdb
    path = _paths(args)["sidecar"]
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute(_SIDECAR_SCHEMA)
    state_path = path.parent / f"crawl_{args.label}.state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    cursor, done, edges_n = state.get("cursor", "*"), state.get("works", 0), state.get("edges", 0)
    mailto = os.environ.get("CROSSREF_MAILTO") or None
    pages, t0 = 0, time.time()
    while True:
        msg = _crossref_page(args.prefix, cursor, args.rows, args.until, mailto)
        items = msg.get("items") or []
        if not items:
            break
        metas, edges = [], []
        for it in items:
            m, e, _ = _crossref_parse(it)
            if m:
                metas.append(m)
                edges.extend(e)
        con.execute("BEGIN TRANSACTION")
        con.executemany(
            "INSERT OR REPLACE INTO meta (doi,title,title_norm,year,abstract,"
            "cited_by_count,crawled_at,crawl_source) VALUES (?,?,?,?,?,?,now(),?)",
            [m + (args.label,) for m in metas])
        if edges:
            con.executemany(
                "INSERT INTO edges (citing_doi,cited_doi,edge_source) VALUES (?,?,?)",
                [(a, b, f"crossref:{args.label}") for a, b in edges])
        con.execute("COMMIT")
        done, edges_n, pages = done + len(metas), edges_n + len(edges), pages + 1
        cursor = msg.get("next-cursor")
        state_path.write_text(json.dumps({"cursor": cursor, "works": done,
                                          "edges": edges_n, "prefix": args.prefix}))
        if pages % 10 == 0:
            print(f"  {done:,}/{msg.get('total-results', 0):,}건  간선 {edges_n:,}", flush=True)
        if not cursor or len(items) < args.rows:
            break
        if args.max_pages and pages >= args.max_pages:
            print("max-pages에 도달. 다시 실행하면 이어서 받습니다.")
            break
    print(f"완료 {args.label}: {done:,}건, 간선 {edges_n:,}, {time.time()-t0:.0f}초")
    con.close()


# ============================================================================

def cmd_s2(args):
    """Download, then build the two DuckDB files. Steps already done are skipped."""
    p = _paths(args)
    steps = ["download", "load-graph", "load-abstracts"] if args.step == "all" else [args.step]
    built = {"load-graph": p["graph"].exists(), "load-abstracts": p["abstracts"].exists()}
    for step in steps:
        if args.step == "all" and (built.get(step) or (step == "download" and all(built.values()))):
            print(f"[{step}] 이미 되어 있어 건너뜀")
            continue
        print(f"\n[{step}]", flush=True)
        {"download": cmd_download_s2, "load-graph": cmd_load_graph,
         "load-abstracts": cmd_load_abstracts}[step](args)
        if args.dry_run:
            return
    print(f"\nconfig.yaml:\n  s2_graph_db: {p['graph']}\n  s2_abstracts_db: {p['abstracts']}")


def cmd_datalake(args):
    """Check the datalake package; optionally collect a Crossref sidecar."""
    import duckdb
    root = args.datalake_root or os.environ.get("PSA_DATALAKE_ROOT")
    if not root:
        sys.exit("--datalake-root (또는 PSA_DATALAKE_ROOT)가 필요합니다.")
    for name, table in (("paper_meta.duckdb", "meta"), ("citation_graph.duckdb", "edges")):
        path = Path(root) / "search" / name
        if not path.exists():
            sys.exit(f"파일이 없습니다: {path}")
        con = duckdb.connect(str(path), read_only=True)
        n = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        con.close()
        print(f"{name}: {table} {n:,}행")
    print(f"\nconfig.yaml:\n  datalake_root: {root}")
    if args.sidecar:
        args.prefix = args.sidecar
        cmd_sidecar(args)


def main(argv=None):
    ap = argparse.ArgumentParser(description="데이터 준비 (s2 또는 datalake)")
    ap.add_argument("--backend", choices=["s2", "datalake"], required=True)
    ap.add_argument("--data-dir", default=os.environ.get("PSA_DATA_DIR", "data"),
                    help="s2 파일을 둘 폴더 (기본: PSA_DATA_DIR, 없으면 ./data)")
    ap.add_argument("--datalake-root", default=None, help="science_datalake 폴더")
    s2 = ap.add_argument_group("s2")
    s2.add_argument("--step", default="all",
                    choices=["all", "download", "load-graph", "load-abstracts"])
    s2.add_argument("--release", default=S2_RELEASE)
    s2.add_argument("--parallel", type=int, default=8, help="동시 다운로드 수")
    s2.add_argument("--dry-run", action="store_true", help="받지 않고 파일 수와 크기만 출력")
    s2.add_argument("--memory", default="200GB", help="DuckDB 메모리 상한")
    s2.add_argument("--threads", type=int, default=32)
    s2.add_argument("--limit-shards", type=int, default=None, help="앞의 N개 샤드만 (시험용)")
    dl = ap.add_argument_group("datalake (선택: Crossref 보충)")
    dl.add_argument("--sidecar", default=None, metavar="DOI_PREFIX", help="예: 10.1145")
    dl.add_argument("--label", default="sidecar")
    dl.add_argument("--until", default="2026-03-03")
    dl.add_argument("--rows", type=int, default=500)
    dl.add_argument("--max-pages", type=int, default=0)
    args = ap.parse_args(argv)
    args.only, args.batch = None, 40
    (cmd_s2 if args.backend == "s2" else cmd_datalake)(args)


if __name__ == "__main__":
    main()
