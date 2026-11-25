"""Shared helpers: config, timing, network guard, key redaction, blocklist, input files."""
from __future__ import annotations

import contextlib
import datetime as dt
import io
import json
import logging
import os
import re
import socket
import sys
import threading
import time
from collections import defaultdict
from urllib.parse import urlparse

# --------------------------------------------------------------------------- config

DEFAULTS = {
    "store": {"duckdb_memory": "8GB", "duckdb_threads": 4},
    "search": {"cache_db": "cache/s2_search.sqlite",
               "throttle_file": "cache/.s2_throttle",
               "shared_throttle": True, "min_interval_sec": 1.1,
               "api_key_env": "S2_API_KEY"},
    "llm": {"url": "http://127.0.0.1:8000/v1", "model": "qwen/qwen3-8b",
            "parallel": 32},
    "embed": {"model": "sentence-transformers/all-MiniLM-L6-v2",
              "device": "cpu", "batch_size": 64},
    "prior": "priors/expansion_prior.json",
}


def _merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def load_config(path=None):
    """Load config.yaml over the defaults; relative paths are resolved against the repo."""
    data = {}
    if path:
        import yaml
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    cfg = _merge(DEFAULTS, data)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for sect, key in (("search", "cache_db"), ("search", "throttle_file")):
        if not os.path.isabs(cfg[sect][key]):
            cfg[sect][key] = os.path.join(root, cfg[sect][key])
    if not os.path.isabs(cfg["prior"]):
        cfg["prior"] = os.path.join(root, cfg["prior"])
    return cfg


# --------------------------------------------------------------------------- input files

def _date(value):
    if value in (None, ""):
        return None
    return dt.date.fromisoformat(str(value)[:10])


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_queries(path):
    """queries.jsonl -> [{qid, title, cutoff}]."""
    out = []
    for row in read_jsonl(path):
        title = row.get("title") or row.get("question")
        if not row.get("qid") or not title:
            raise ValueError(f"{path}: qid 와 title 이 필요하다: {row}")
        out.append({"qid": row["qid"], "title": title, "cutoff": _date(row.get("cutoff"))})
    return out


def load_gold(path, by="corpusid"):
    """gold.jsonl -> {qid: {key: {pubdate, in_store, dois}}}.

    by="corpusid" keys by corpusId (s2); by="doi" keys by DOI (datalake).
    """
    from psa.store import arxiv_id_to_doi, norm_doi
    out = defaultdict(dict)
    for row in read_jsonl(path):
        dois = [d for d in (norm_doi(row.get("doi")),
                            arxiv_id_to_doi(row["arxiv"]) if row.get("arxiv") else None) if d]
        if by == "doi":
            key = dois[0] if dois else (f"corpusid:{row['corpusid']}"
                                        if row.get("corpusid") is not None else None)
        else:
            key = int(row["corpusid"]) if row.get("corpusid") is not None else None
        if key is None:
            continue
        out[row["qid"]][key] = {"pubdate": _date(row.get("pubdate")),
                                "in_store": row.get("in_store") or {}, "dois": dois}
    return dict(out)


def gold_aliases(gold):
    """{doi: gold key} for one query (a gold paper can have a DOI and an arXiv DOI)."""
    return {d: key for key, g in gold.items() for d in g.get("dois", [])}


# --------------------------------------------------------------------------- timing

STAGES = ("search", "expand", "judge", "other")
KINDS = ("s2_api", "cache_hit", "local_db", "llm")


class Meter:
    """Counts calls and wall-clock time per stage for one query."""

    def __init__(self, qid, *, system="psa", backend=""):
        self.qid, self.system, self.backend = qid, system, backend
        self.calls = defaultdict(lambda: defaultdict(int))
        self.seconds = defaultdict(lambda: defaultdict(float))
        self.extra = defaultdict(int)
        self.intervals = defaultdict(list)
        self.waits = []
        self._local = threading.local()
        self._lock = threading.Lock()
        self.started = time.perf_counter()

    @property
    def current(self):
        return getattr(self._local, "stage", "other")

    @contextlib.contextmanager
    def stage(self, name):
        """Count calls inside this block under stage `name`."""
        if name not in STAGES:
            raise ValueError(f"단계는 {STAGES} 중 하나: {name}")
        previous = self.current
        self._local.stage = name
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._local.stage = previous
            with self._lock:
                self.intervals[name].append((t0, time.perf_counter()))

    @contextlib.contextmanager
    def call(self, kind, n=1):
        """Record one call (or n) in the current stage."""
        if kind not in KINDS:
            raise ValueError(f"종류는 {KINDS} 중 하나: {kind}")
        t0 = time.perf_counter()
        try:
            yield
        finally:
            wall = time.perf_counter() - t0
            with self._lock:
                self.calls[self.current][kind] += n
                self.seconds[self.current][kind] += wall

    def add(self, kind, wall):
        """Record a call that already finished."""
        with self._lock:
            self.calls[self.current][kind] += 1
            self.seconds[self.current][kind] += wall

    def wait_interval(self, start, end):
        with self._lock:
            self.waits.append((start, end))

    def count(self, name, n=1):
        with self._lock:
            self.extra[name] += n

    @staticmethod
    def _union(intervals):
        total, end = 0.0, None
        for a, b in sorted(intervals):
            if end is None or a > end:
                total += b - a
                end = b
            elif b > end:
                total += b - end
                end = b
        return total

    def summary(self):
        wall = time.perf_counter() - self.started
        stage_wall = {s: round(self._union(self.intervals[s]), 3)
                      for s in STAGES if self.intervals[s]}
        covered = self._union([iv for s in STAGES if s != "other"
                               for iv in self.intervals[s]])
        stage_wall["other"] = round(max(0.0, wall - covered), 3)
        s2_wait = self._union(self.waits)
        return {
            "qid": self.qid, "system": self.system, "backend": self.backend,
            "wall_sec": round(wall, 3),
            "s2_wait_wall_sec": round(s2_wait, 3),
            "wall_minus_s2_wait_sec": round(max(0.0, wall - s2_wait), 3),
            "stage_wall_sec": stage_wall,
            "calls": {s: dict(self.calls[s]) for s in STAGES if self.calls[s]},
            "thread_seconds": {s: {k: round(v, 3) for k, v in self.seconds[s].items()}
                               for s in STAGES if self.seconds[s]},
            "counts": dict(self.extra),
        }


# --------------------------------------------------------------------------- network guard

_LOOPBACK = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
_guard_lock = threading.Lock()
_blocked_hosts = {}
_allowed = set()
_installed = False


def netguard_install(allowed_hosts):
    """Block name lookups for hosts outside the allow list."""
    global _installed
    _allowed.update(h.lower() for h in allowed_hosts if h)
    _allowed.update(_LOOPBACK)
    if _installed:
        return
    original = socket.getaddrinfo

    def guarded(host, *args, **kwargs):
        name = (host.decode() if isinstance(host, bytes) else str(host or "")).lower()
        if name and name not in _allowed:
            with _guard_lock:
                _blocked_hosts[name] = _blocked_hosts.get(name, 0) + 1
            raise socket.gaierror(socket.EAI_NONAME,
                                  f"netguard: {name} is not an allowed host")
        return original(host, *args, **kwargs)

    socket.getaddrinfo = guarded
    _installed = True


def netguard_report():
    with _guard_lock:
        return dict(_blocked_hosts)


def host_of(url):
    return (urlparse(url).hostname or "").lower()


# --------------------------------------------------------------------------- key redaction

KEY_ENV = ("S2_API_KEY", "SEMANTIC_SCHOLAR_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
_MASK = "***"
_KEY_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key=)[^&\s'\"]+"),
    re.compile(r"(?i)(x-api-key['\"]?\s*[:=]\s*['\"]?)[^'\"\s,}]+"),
    re.compile(r"(?i)(authorization['\"]?\s*[:=]\s*['\"]?bearer\s+)[^'\"\s,}]+"),
]


def redact(text):
    """Mask API key values in text."""
    if not text:
        return text
    values = sorted({v for name in KEY_ENV if len(v := os.environ.get(name, "")) >= 12},
                    key=len, reverse=True)
    for value in values:
        text = text.replace(value, _MASK)
    for pattern in _KEY_PATTERNS:
        text = pattern.sub(lambda m: m.group(1) + _MASK, text)
    return text


class _Stream(io.TextIOBase):
    def __init__(self, stream):
        self._stream = stream

    def write(self, text):
        return self._stream.write(redact(text))

    def flush(self):
        return self._stream.flush()

    def isatty(self):
        return self._stream.isatty()

    def fileno(self):
        return self._stream.fileno()

    @property
    def encoding(self):
        return getattr(self._stream, "encoding", "utf-8")


class _Filter(logging.Filter):
    def filter(self, record):
        record.msg = redact(record.getMessage())
        record.args = ()
        return True


def redact_install():
    """Mask keys in stdout, stderr and logging output."""
    if not isinstance(sys.stdout, _Stream):
        sys.stdout = _Stream(sys.stdout)
    if not isinstance(sys.stderr, _Stream):
        sys.stderr = _Stream(sys.stderr)
    root = logging.getLogger()
    if not any(isinstance(f, _Filter) for f in root.filters):
        root.addFilter(_Filter())


# --------------------------------------------------------------------------- blocklist

class Blocklist:
    """Papers that must not appear as candidates, matched by DOI, corpusId, arXiv id or title."""

    def __init__(self, path=None):
        self.dois, self.arxiv, self.corpusids, self.titles = set(), set(), set(), set()
        if not path:
            return
        from psa.store import (arxiv_id_to_doi, norm_doi, normalize_arxiv_id,
                               normalize_title)
        for row in read_jsonl(path):
            if row.get("doi") and (d := norm_doi(row["doi"])):
                self.dois.add(d)
            if row.get("arxiv") and (a := normalize_arxiv_id(row["arxiv"])):
                self.arxiv.add(a)
                if d := norm_doi(arxiv_id_to_doi(a)):
                    self.dois.add(d)
            if row.get("corpusid") is not None:
                self.corpusids.add(int(row["corpusid"]))
            if row.get("title") and (t := normalize_title(row["title"])):
                self.titles.add(t)

    def __len__(self):
        return len(self.dois) + len(self.arxiv) + len(self.corpusids) + len(self.titles)

    def hit(self, paper, *, use_corpusid):
        """Which key the paper matched on, or None."""
        if not len(self):
            return None
        from psa.store import norm_doi, normalize_arxiv_id, normalize_title
        doi, arxiv, title = paper.get("doi"), paper.get("arxiv"), paper.get("title")
        cid = paper.get("corpusid") if use_corpusid else None
        if doi and norm_doi(doi) in self.dois:
            return "doi"
        if cid is not None and cid in self.corpusids:
            return "corpusid"
        if arxiv and normalize_arxiv_id(arxiv) in self.arxiv:
            return "arxiv"
        if title and normalize_title(title) in self.titles:
            return "title"
        return None

    def keys_for(self, backend_name):
        """Keys to pre-filter by during expansion."""
        return set(self.corpusids) if backend_name == "s2" else set(self.dois)
