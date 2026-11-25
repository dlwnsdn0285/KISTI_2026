"""Expansion building blocks: embedding relevance, frontier, neighbour harvest, and the bandit.

Relevance is all-MiniLM-L6-v2 cosine (query: title only; paper: title + abstract).
The score is a Poisson model: expected reward per new pair of one (node, direction) = exp(w.x).
The bandit samples w from the prior (priors/*.json, Thompson sampling) and refits after each pull.
"""
from __future__ import annotations

import json
import math
import random

import numpy as np

DIRECTIONS = ("references", "cited_by")
MAX_TOKENS = 256

# Feature names; the prior file selects a subset.
FEATURE_NAMES = ["age", "log_citations", "log_indegree_before", "is_cited_by",
                 "log_expected_pairs", "relevance"]


# --------------------------------------------------------------------------- embedding

def paper_text(paper):
    """Paper text for embedding: title, blank line, abstract."""
    title = (paper.get("title") or "").strip()
    body = (paper.get("abstract") or "").strip()
    return f"{title}\n\n{body}".strip() if body else title


class MiniLM:
    """Sentence embeddings (mean pooling, L2-normalised)."""

    def __init__(self, model="sentence-transformers/all-MiniLM-L6-v2", device="cpu",
                 batch_size=64):
        self.name, self.device, self.batch_size = model, device, batch_size
        self.model = self.tokenizer = None

    def load(self):
        if self.model is None:
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.name)
            self.model = AutoModel.from_pretrained(self.name).to(self.device).eval()
        return self

    def embed(self, texts):
        """One unit vector per text."""
        import torch
        self.load()
        out = []
        for start in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(texts[start:start + self.batch_size], padding=True,
                                   truncation=True, max_length=MAX_TOKENS,
                                   return_tensors="pt")
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                hidden = self.model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            out.append(torch.nn.functional.normalize(pooled, dim=-1).cpu())
        return torch.cat(out) if out else torch.empty(0, 384)


class Relevance:
    """Cosine between one query and papers, cached per paper."""

    def __init__(self, query_text, model):
        self.model = model
        self.query_text = (query_text or "").strip()
        self.query = None
        self.cache = {}

    def cosine(self, papers):
        """{key: cosine}."""
        todo = [p for p in papers if p["corpusid"] not in self.cache]
        if todo:
            if self.query is None:
                self.query = self.model.embed([self.query_text])[0]
            vecs = self.model.embed([paper_text(p) for p in todo])
            for p, v in zip(todo, (vecs @ self.query).tolist()):
                self.cache[p["corpusid"]] = float(v)
        return {p["corpusid"]: self.cache[p["corpusid"]] for p in papers}


# --------------------------------------------------------------------------- frontier / harvest

def frontier(nodes, top=200, sample=100, seed=0):
    """Top `top` nodes by cosine plus `sample` random ones from the rest."""
    if len(nodes) <= top + sample:
        return list(nodes), len(nodes), 0
    head = nodes[:top]
    rest = nodes[top:]
    sampled = random.Random(seed).sample(rest, sample)
    return head + sorted(sampled, key=lambda n: n["rank"]), len(head), len(sampled)


def harvest(ids, direction, cutoff, store, block, *, cap, window_years, floor, batch):
    """Neighbours of each node in one direction, after the cutoff / window / citation filters."""
    out = {}
    floor = floor if direction == "cited_by" else None
    for start in range(0, len(ids), batch):
        chunk = ids[start:start + batch]
        hop = store.neighbours(chunk, direction, cap=cap, cutoff=cutoff,
                               window_years=window_years, min_indegree_before=floor)
        for cid in chunk:
            out[cid] = [c for c in (hop.edges.get(cid) or []) if c not in block]
    return out


# --------------------------------------------------------------------------- score model

def vector(row, direction):
    """Raw features of one (node, direction), in FEATURE_NAMES order."""
    relevance = row.get("embed_cosine")
    return [
        float(row["age"] if row.get("age") is not None else 0.0),
        math.log1p(max(0, row.get("citationcount") or 0)),
        math.log1p(max(0, row.get("indegree_before") or 0)),
        1.0 if direction == "cited_by" else 0.0,
        math.log1p(max(0, row["expected_pairs"][direction])),
        float(relevance if relevance is not None else 0.0),
    ]


def design(raw, scale):
    """Intercept plus standardised features."""
    mu, sd = scale
    return [1.0] + ((np.asarray(raw, dtype=float) - mu) / sd).tolist()


def fit_poisson(X, y, prior_mean, prior_precision, offset=None, steps=200, tol=1e-8):
    """MAP fit of a log-link Poisson regression under a Gaussian prior."""
    w = np.array(prior_mean, dtype=float)
    P0 = np.asarray(prior_precision, dtype=float)
    m0 = np.asarray(prior_mean, dtype=float)
    if len(X) == 0:
        return w, P0
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    off = np.zeros(len(y)) if offset is None else np.asarray(offset, dtype=float)
    for _ in range(steps):
        mu = np.exp(np.clip(X @ w + off, -30, 30))
        grad = X.T @ (mu - y) + P0 @ (w - m0)
        H = X.T @ (X * mu[:, None]) + P0
        step = np.linalg.solve(H, grad)
        w = w - step
        if np.max(np.abs(step)) < tol:
            break
    mu = np.exp(np.clip(X @ w + off, -30, 30))
    return w, X.T @ (X * mu[:, None]) + P0


def load_prior(path, intercept_precision=0.01):
    """Load a prior file. Only the intercept gets a wide prior; other coefficients stay as trained."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    w = np.array(data["w"], dtype=float)
    precision = np.array(data["precision"], dtype=float)
    precision[0, :] = 0.0
    precision[:, 0] = 0.0
    precision[0, 0] = intercept_precision
    scale = (np.array(data["scale"]["mean"]), np.array(data["scale"]["sd"]))
    columns = [FEATURE_NAMES.index(n) for n in data["features"][1:]]
    return w, precision, scale, data, columns


class Bandit:
    """Thompson-sampling bandit over (node, direction) arms."""

    def __init__(self, arms, x, w, precision, rng):
        self.arms = list(arms)
        self.rank = {k: i for i, k in enumerate(sorted({a[0] for a in self.arms}))}
        self.x = x
        self.prior_mean = np.array(w, dtype=float)
        self.prior_precision = np.array(precision, dtype=float)
        self.w = self.prior_mean.copy()
        self.precision = self.prior_precision.copy()
        self.rng = rng
        self.seen_x, self.seen_y, self.seen_off = [], [], []

    def set_arms(self, arms, x):
        """Replace the arm list (when the frontier is re-drawn); learned weights are kept."""
        self.arms = sorted(arms)
        self.rank = {k: i for i, k in enumerate(sorted({a[0] for a in self.arms}))}
        self.x = x

    def _draw(self):
        try:
            cov = np.linalg.inv(self.precision)
            return self.rng.multivariate_normal(self.w, (cov + cov.T) / 2)
        except np.linalg.LinAlgError:
            return self.w

    def pick(self):
        w = self._draw()
        best = max(self.arms, key=lambda a: (float(np.dot(w, self.x[a])),
                                             -self.rank[a[0]], a[1] == "references"))
        self.arms.remove(best)
        return best

    def observe(self, arm, count, exposure):
        self.seen_x.append(self.x[arm])
        self.seen_y.append(float(count))
        self.seen_off.append(math.log(max(1, exposure)))
        self.w, self.precision = fit_poisson(self.seen_x, self.seen_y, self.prior_mean,
                                             self.prior_precision, offset=self.seen_off)
