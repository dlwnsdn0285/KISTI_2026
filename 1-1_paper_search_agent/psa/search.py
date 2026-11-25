"""Search step: the LLM writes search queries from the title, and S2 search results become the seeds.

S2 calls go through an SQLite cache and a machine-wide rate limit.
The API key is read only from an environment variable (default S2_API_KEY).
"""
from __future__ import annotations

import datetime as dt
import fcntl
import json
import os
import random
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

PROMPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
QUERY_PROMPT = "query.txt"
JUDGE_PROMPT = "judge_cite_rating.txt"

SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
SEARCH_FIELDS = "paperId,corpusId,externalIds,title,year,publicationDate"
TOP_K = 50                      # results per query
K_QUERIES = 4                   # number of queries
BANNED = re.compile(r"\b(survey|surveys|review|reviews|overview|overviews)\b", re.I)


# --------------------------------------------------------------------------- cache

class _SharedThrottle:
    """Request interval shared across processes (file lock)."""

    def __init__(self, interval, path):
        self.interval = interval
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def wait(self):
        t0 = time.time()
        with open(self.path, "a+") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                fh.seek(0)
                raw = fh.read().strip()
                last = float(raw) if raw else 0.0
                gap = last + self.interval - time.time()
                if gap > 0:
                    time.sleep(gap)
                fh.seek(0)
                fh.truncate()
                fh.write(repr(time.time()))
                fh.flush()
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)
        return time.time() - t0


class _Throttle:
    """Request interval within one process."""

    def __init__(self, interval):
        self.interval = interval
        self.lock = threading.Lock()
        self.last = 0.0

    def wait(self):
        t0 = time.perf_counter()
        with self.lock:
            gap = self.interval - (time.perf_counter() - self.last)
            if gap > 0:
                time.sleep(gap)
            self.last = time.perf_counter()
        return time.perf_counter() - t0


def _cache_key(query, params):
    return json.dumps({"q": query, **{k: params[k] for k in sorted(params)}},
                      ensure_ascii=False, sort_keys=True)


class SearchCache:
    """S2 relevance search with a cache."""

    def __init__(self, cfg, *, system="psa", meter=None):
        s = cfg["search"]
        self.system, self.meter = system, meter
        self.key = os.environ.get(s["api_key_env"], "")
        self.shared = bool(s["shared_throttle"])
        self.gate = (_SharedThrottle(s["min_interval_sec"], s["throttle_file"])
                     if self.shared else _Throttle(s["min_interval_sec"]))
        self.counts = {"hit": 0, "miss": 0, "api_call": 0, "api_fail": 0,
                       "empty": 0, "http_200": 0, "http_429": 0,
                       "http_403": 0, "http_other": 0, "network_error": 0}
        self.seconds = {"api_wait": 0.0, "cache": 0.0, "backoff": 0.0, "rate_wait": 0.0}
        os.makedirs(os.path.dirname(s["cache_db"]), exist_ok=True)
        self.con = sqlite3.connect(s["cache_db"], timeout=60, check_same_thread=False)
        # concurrent first opens can hit a lock; retry
        for attempt in range(30):
            try:
                self.con.execute("PRAGMA busy_timeout=60000")
                self.con.execute("PRAGMA journal_mode=WAL")
                self.con.execute(
                    "CREATE TABLE IF NOT EXISTS search ("
                    " key TEXT PRIMARY KEY, query TEXT, params TEXT, results TEXT,"
                    " total INTEGER, system TEXT, fetched_at TEXT)")
                self.con.commit()
                break
            except sqlite3.OperationalError as err:
                if "locked" not in str(err) or attempt == 29:
                    raise
                time.sleep(0.2 + random.random())
        self.lock = threading.Lock()

    def search(self, query, *, limit=TOP_K, date_until=None):
        """Search results for `query`, published up to date_until."""
        iso = lambda d: d.isoformat() if isinstance(d, dt.date) else d  # noqa: E731
        params = {"limit": limit, "offset": 0, "fields": SEARCH_FIELDS,
                  "date_until": iso(date_until)}
        key = _cache_key(query, params)
        t0 = time.perf_counter()
        with self.lock:
            row = self.con.execute("SELECT results FROM search WHERE key = ?",
                                   (key,)).fetchone()
        self.seconds["cache"] += time.perf_counter() - t0
        if row is not None:
            self.counts["hit"] += 1
            if self.meter is not None:
                self.meter.add("cache_hit", time.perf_counter() - t0)
            return json.loads(row[0])
        self.counts["miss"] += 1
        hits, total = self._call(query, params)
        if hits is None:
            return []
        with self.lock:
            self.con.execute(
                "INSERT OR REPLACE INTO search VALUES (?, ?, ?, ?, ?, ?, ?)",
                (key, query, json.dumps(params), json.dumps(hits), total,
                 self.system, dt.datetime.now().isoformat(timespec="seconds")))
            self.con.commit()
        return hits

    def stats(self):
        return {"system": self.system,
                "throttle": "shared" if self.shared else "per_process",
                **self.counts, **{f"sec_{k}": round(v, 3) for k, v in self.seconds.items()}}

    def _wait(self, start, end):
        if self.meter is not None:
            self.meter.wait_interval(start, end)

    def _call(self, query, params):
        """One API call, with backoff on 429."""
        api = {"query": query, "limit": params["limit"], "offset": params["offset"],
               "fields": params["fields"]}
        if params.get("date_until"):
            api["publicationDateOrYear"] = f":{params['date_until']}"
        url = SEARCH_URL + "?" + urllib.parse.urlencode(api)
        request = urllib.request.Request(url, headers={
            "User-Agent": "paper-search-agent/1.0",
            **({"x-api-key": self.key} if self.key else {})})
        for attempt in range(6):
            q0 = time.perf_counter()
            waited = self.gate.wait()
            if waited:
                self.seconds["rate_wait"] += waited
                self._wait(q0, time.perf_counter())
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    body = json.loads(response.read())
            except urllib.error.HTTPError as err:
                wall = time.perf_counter() - t0
                self.seconds["api_wait"] += wall
                self._wait(t0, t0 + wall)
                if err.code == 429:
                    self.counts["http_429"] += 1
                    self.seconds["backoff"] += 2 ** attempt
                    b0 = time.perf_counter()
                    time.sleep(2 ** attempt)
                    self._wait(b0, time.perf_counter())
                    continue
                self.counts["http_403" if err.code == 403 else "http_other"] += 1
                self.counts["api_fail"] += 1
                return None, None
            except Exception:  # noqa: BLE001 - retry on network errors
                wall = time.perf_counter() - t0
                self.seconds["api_wait"] += wall
                self._wait(t0, t0 + wall)
                self.counts["network_error"] += 1
                continue
            wall = time.perf_counter() - t0
            self.seconds["api_wait"] += wall
            self._wait(t0, t0 + wall)
            self.counts["api_call"] += 1
            self.counts["http_200"] += 1
            if self.meter is not None:
                self.meter.add("s2_api", wall)
            hits = body.get("data") or []
            if not hits:
                self.counts["empty"] += 1
            return hits, body.get("total")
        self.counts["api_fail"] += 1
        return None, None


# --------------------------------------------------------------------------- queries + search

def past_cutoff(hit, cutoff):
    """True if a result may be after the cutoff (unknown dates count as after)."""
    if cutoff is None:
        return False
    date = hit.get("publicationDate")
    if date:
        try:
            return dt.date.fromisoformat(date) > cutoff
        except ValueError:
            pass
    year = hit.get("year")
    if year:
        return int(year) >= cutoff.year
    return True


def parse_lines(text):
    """Parse one search query per line from the LLM answer."""
    out, seen = [], set()
    for line in text.splitlines():
        q = re.sub(r"^\s*(?:\d+[.)\]:]?|[-*•])\s*", "", line).strip()
        q = q.strip("\"'`“”‘’ ").strip()
        if not q or BANNED.search(q) or q.lower() in seen:
            continue
        seen.add(q.lower())
        out.append(q)
    return out


class LLMQuerySearcher:
    """Title -> k LLM queries -> S2 search -> seed keys."""

    name = "s2_search:llm"

    def __init__(self, *, cache, cfg, key_of=None, no_key_label="no_corpusid", meter=None):
        self.cache = cache
        self.llm_url = cfg["llm"]["url"].rstrip("/")
        self.model = cfg["llm"]["model"]
        self.prompt_file = os.path.join(PROMPT_DIR, QUERY_PROMPT)
        self.template = open(self.prompt_file, encoding="utf-8").read()
        self.key_of = key_of or (lambda hit: int(hit["corpusId"])
                                 if hit.get("corpusId") is not None else None)
        self.no_key_label = no_key_label
        self.meter = meter
        self.last_queries, self.last_counts, self.last_llm = {}, {}, {}

    def _ask(self, prompt):
        """One vLLM chat completion."""
        body = json.dumps({
            "model": self.model, "temperature": 0, "max_tokens": 256,
            "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": False}}).encode()
        req = urllib.request.Request(f"{self.llm_url}/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        if self.meter is not None:
            with self.meter.call("llm"):
                with urllib.request.urlopen(req, timeout=300) as r:
                    data = json.loads(r.read())
        else:
            with urllib.request.urlopen(req, timeout=300) as r:
                data = json.loads(r.read())
        return data["choices"][0]["message"]["content"] or "", data.get("usage") or {}

    def queries(self, title):
        """k search queries; falls back to the title if the LLM gives too few."""
        prompt = self.template.format(survey_title=title, question=title)
        self.last_llm = {"prompt_file": os.path.basename(self.prompt_file), "calls": 0,
                         "prompt_tokens": 0, "completion_tokens": 0, "raw": [],
                         "parse_failed": False, "padded_with_title": False}
        best = []
        for _ in range(3):
            text, usage = self._ask(prompt)
            self.last_llm["calls"] += 1
            self.last_llm["prompt_tokens"] += usage.get("prompt_tokens", 0)
            self.last_llm["completion_tokens"] += usage.get("completion_tokens", 0)
            self.last_llm["raw"].append(text)
            got = parse_lines(text)
            if len(got) > len(best):
                best = got
            if len(best) >= K_QUERIES:
                break
        best = best[:K_QUERIES]
        if len(best) < K_QUERIES and title.lower() not in {q.lower() for q in best}:
            best.append(title)
            self.last_llm["padded_with_title"] = True
        self.last_llm["parse_failed"] = not best
        self.last_llm["queries"] = best
        return [(f"llm_{i}", q) for i, q in enumerate(best, 1)]

    def search(self, title, *, cutoff=None):
        """Sorted seed keys; results after the cutoff are dropped."""
        union = {}
        counts = {"returned": 0, self.no_key_label: 0, "dropped_after_cutoff": 0,
                  "api_fail": 0}
        self.last_queries = {}
        for label, query in self.queries(title):
            before = self.cache.counts.get("api_fail", 0)
            hits = self.cache.search(query, limit=TOP_K, date_until=cutoff)
            failed = self.cache.counts.get("api_fail", 0) - before
            counts["api_fail"] += failed
            counts["returned"] += len(hits)
            kept = 0
            for hit in hits:
                key = self.key_of(hit)
                if key is None:
                    counts[self.no_key_label] += 1
                    continue
                if past_cutoff(hit, cutoff):
                    counts["dropped_after_cutoff"] += 1
                    continue
                kept += 1
                union.setdefault(key, label)
            self.last_queries[label] = {"query": query, "returned": len(hits),
                                        "kept": kept, "api_fail": failed}
        counts["llm_parse_failed"] = int(self.last_llm.get("parse_failed", False))
        self.last_counts = counts
        return sorted(union)
