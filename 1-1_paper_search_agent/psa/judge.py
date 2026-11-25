"""Judging and output: the LLM rates each candidate 0-9; rating and citation links pick the final list.

Output rule: rating >= 8 and links >= the 75th percentile among judged candidates.
"""
from __future__ import annotations

import json
import math
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

PROMPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
MISSING_ABSTRACT = "(no abstract available)"
TOP_LOGPROBS = 20
LINK_QUANTILE = 0.75
RATING_MIN = 8.0


class VLLMJudge:
    """Scores a candidate in [0, 1] from the first-token digit probabilities (vLLM /completions)."""

    def __init__(self, *, url, model, prompt_file, parallel=32, timeout=600, meter=None):
        self.url = url.rstrip("/")
        self.model = model
        self.parallel = parallel
        self.timeout = timeout
        self.meter = meter
        self.prompt_file = prompt_file
        self.template = open(os.path.join(PROMPT_DIR, prompt_file), encoding="utf-8").read()
        self.name = f"vllm:{model}:{os.path.splitext(prompt_file)[0]}"
        self.last_usage = None
        self.true_not_in_top = 0

    def prompt(self, question, paper):
        abstract = (paper.get("abstract") or "").strip() or MISSING_ABSTRACT
        user = self.template.format(survey_title=question, question=question,
                                    candidate_title=paper.get("title") or "",
                                    candidate_abstract=abstract)
        return ("<|im_start|>user\n" + user.strip() + "<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\nRating: ")

    def _one(self, prompt):
        body = json.dumps({"model": self.model, "prompt": prompt, "max_tokens": 1,
                           "temperature": 0, "logprobs": TOP_LOGPROBS}).encode()
        request = urllib.request.Request(f"{self.url}/completions", data=body,
                                         headers={"Content-Type": "application/json"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as r:
                    data = json.loads(r.read())
                break
            except Exception:  # noqa: BLE001 - retry
                if attempt == 2:
                    raise
        top = (data["choices"][0].get("logprobs") or {}).get("top_logprobs") or [{}]
        digits = {str(i): 0.0 for i in range(10)}
        for tok, lp in (top[0] or {}).items():
            word = tok.replace("Ġ", " ").strip()
            if word in digits:
                digits[word] += math.exp(lp)
        total = sum(digits.values())
        usage = data.get("usage") or {}
        if not total:
            return 0.0, True, usage
        return sum(int(k) * v for k, v in digits.items()) / total / 9.0, False, usage

    def score(self, question, papers):
        """One score per paper, in order."""
        if not papers:
            self.last_usage = None
            return []
        prompts = [self.prompt(question, p) for p in papers]
        if self.meter is not None:
            with self.meter.call("llm", n=len(papers)):
                results = self._map(prompts)
        else:
            results = self._map(prompts)
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0,
                           "llm_calls": len(results), "true_not_in_top": 0}
        out = []
        for p, missing, usage in results:
            out.append(p)
            self.last_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
            self.last_usage["completion_tokens"] += usage.get("completion_tokens", 0)
            if missing:
                self.last_usage["true_not_in_top"] += 1
                self.true_not_in_top += 1
        return out

    def _map(self, prompts):
        with ThreadPoolExecutor(max_workers=self.parallel) as pool:
            return list(pool.map(self._one, prompts))


def judge_shortlist(question, papers, cosine, judge, *, threshold, batch=200):
    """Judge papers with cosine >= threshold, in cosine order."""
    ranked = sorted(papers, key=lambda p: (-cosine[p["corpusid"]], p["corpusid"]))
    shortlist = [p for p in ranked if cosine[p["corpusid"]] >= threshold]
    scores = []
    for start in range(0, len(shortlist), batch):
        chunk = shortlist[start:start + batch]
        got = judge.score(question, chunk)
        if len(got) != len(chunk):
            raise ValueError(f"{judge.name}가 {len(chunk)}편에 점수 {len(got)}개를 줬다")
        scores.extend(got)
    return [{"corpusid": p["corpusid"], "cosine": cosine[p["corpusid"]], "score": s}
            for p, s in zip(shortlist, scores)]


def select_output(judged, links, *, link_quantile=LINK_QUANTILE, rating_min=RATING_MIN):
    """Apply the output rule; returns (selected rows, link threshold)."""
    for r in judged:
        r["rating"] = round((r["score"] or 0.0) * 9, 4)
        r["links"] = links.get(r["corpusid"], 0)
    k_links = (float(np.quantile([r["links"] for r in judged], link_quantile))
               if judged else 0.0)
    keep = [r for r in judged if r["links"] >= k_links and r["rating"] >= rating_min]
    keep.sort(key=lambda r: (-(r["score"] or 0.0), -r["cosine"], r["corpusid"]))
    return [dict(r, rank=i) for i, r in enumerate(keep, 1)], k_links
