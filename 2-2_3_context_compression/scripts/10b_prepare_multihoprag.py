"""Step 10b: MultiHop-RAG 2-evidence fixed-context manifest (multi-evidence, not multi-hop).

    ~/anaconda3/envs/pisco-ref/bin/python scripts/10b_prepare_multihoprag.py

MultiHop-RAG (Tang & Yang 2024, yixuantt/MultiHopRAG, ODC-By): news articles
(Sep-Dec 2023) and GPT-4-written queries whose answer needs facts from several
articles. Query types: inference (which entity do both articles describe),
comparison (do the two articles agree / differ), temporal (did coverage
change between two dated reports), null (unanswerable; excluded here).

Selection: question_type != null, exactly 2 evidence items, both facts found
verbatim in their article body, answer non-empty. No model output is used.

Context (k = 5, same policy as Step 09c):
  * 2 gold chunks: for each evidence item, the article's fact sentence with
    surrounding words so that the whole document fits PISCO's 128-token
    window. Header "{title} ({source}, {date}): " because queries name sources
    and dates. Articles are ~1300 words, so a chunk is necessary: PISCO reads
    only the first 128 tokens of a document.
  * 3 distractor chunks: BM25 (query tokens) over the pool of all evidence
    chunks of the whole dataset, excluding chunks from the query's own
    evidence articles. Same-domain hard negatives.
  * seeded per-sample shuffle: random.Random(f"{SEED}:{sample_id}").

Outputs: data/multihoprag/multihoprag_2ev_gold2_dist3_fixture.json,
         data/manifests/multihoprag_2ev_gold2_dist3_n<N>.json
"""

import datetime
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pisco_research import eval_inputs  # noqa: E402
from pisco_research.scoring import normalize_answer  # noqa: E402

SEED, N_DISTRACTORS = 0, 3
REPO, REVISION = "yixuantt/MultiHopRAG", "71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82"
OUT_DIR = PROJECT_ROOT / "data" / "multihoprag"
REFERENCE_MANIFEST = PROJECT_ROOT / "reference" / "manifest.json"
DOC_TOKEN_BUDGET = eval_inputs.DOC_MAX_TOKENS   # 128: the chunk must not be truncated by PISCO
WORD_RE = re.compile(r"[a-z0-9]+")


def fetch(filename):
    from huggingface_hub import HfApi, hf_hub_download
    path = hf_hub_download(REPO, filename, repo_type="dataset", revision=REVISION, token=False)
    info = HfApi().dataset_info(REPO, revision=REVISION, files_metadata=True, token=False)
    remote = next(s for s in info.siblings if s.rfilename == filename)
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if remote.lfs is not None and digest != remote.lfs.sha256:
        raise RuntimeError(f"{filename}: sha256 mismatch")
    return path, {"repo_id": REPO, "revision": REVISION, "file": filename, "sha256": digest}


def fmt_date(iso):
    d = datetime.datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return f"{d.strftime('%B')} {d.day}, {d.year}"


class ChunkBuilder:
    """Fact-centred chunk of an article that PISCO reads without truncation."""

    def __init__(self, tokenizer):
        self.tok = tokenizer

    def n_tokens(self, text):
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def fits(self, text):
        _, _, truncated = eval_inputs.pisco_encoder_span(self.tok, text)
        return not truncated

    def build(self, article, fact):
        body, fact = article["body"], fact.strip()
        pos = body.find(fact)
        if pos < 0:
            return None
        header = f"{article['title']} ({article['source']}, {fmt_date(article['published_at'])}): "
        before, after = body[:pos].split(), body[pos + len(fact):].split()
        if not self.fits(header + fact):
            return None

        def render(i, j):
            return header + " ".join(before[len(before) - i:] + [fact] + after[:j])
        i = j = 0
        step = 8
        # grow symmetrically in blocks, then back off one word at a time
        while True:
            ni, nj = min(i + step, len(before)), min(j + step, len(after))
            if (ni, nj) == (i, j) or not self.fits(render(ni, nj)):
                break
            i, j = ni, nj
        grew = True
        while grew:
            grew = False
            for di, dj in ((1, 0), (0, 1)):
                ni, nj = min(i + di, len(before)), min(j + dj, len(after))
                if (ni, nj) != (i, j) and self.fits(render(ni, nj)):
                    i, j, grew = ni, nj, True
        text = render(i, j)
        if not self.fits(text):
            raise RuntimeError("chunk does not fit after back-off")
        return {"text": text, "words_before": i, "words_after": j,
                "words_total_before": len(before), "words_total_after": len(after)}


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(WORD_RE.findall(d.lower())) for d in docs]
        self.len = [sum(c.values()) for c in self.docs]
        self.avg = sum(self.len) / len(self.len)
        df = Counter(t for c in self.docs for t in c)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query, idx):
        q = WORD_RE.findall(query.lower())
        c, dl = self.docs[idx], self.len[idx]
        s = 0.0
        for t in q:
            if t in c:
                tf = c[t]
                s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avg))
        return s


def answer_kind(answer, qtype):
    a = normalize_answer(answer)
    if a in ("yes", "no"):
        return "yes_no"
    return "entity" if qtype == "inference" else "other"


def main():
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    q_path, q_source = fetch("MultiHopRAG.json")
    c_path, c_source = fetch("corpus.json")
    queries = json.load(open(q_path))
    corpus = json.load(open(c_path))
    by_url = {a["url"]: a for a in corpus}
    if len(by_url) != len(corpus):
        raise RuntimeError("duplicate article urls")

    os.environ["HF_HUB_OFFLINE"] = "1"
    ref = json.loads(REFERENCE_MANIFEST.read_text())
    tok = eval_inputs.load_pisco_tokenizer(ref["snapshots"]["backbone"]["local_dir"])
    tok_source = {"repo_id": ref["backbone"]["repo_id"], "revision": ref["backbone"]["revision_sha"],
                  "added_special_tokens": eval_inputs.PISCO_ADDED_TOKENS}
    builder = ChunkBuilder(tok)

    # chunk pool: every evidence item of the whole dataset (also 3/4-evidence and other queries)
    pool, pool_key = [], {}
    chunk_cache = {}
    for qi, q in enumerate(queries):
        for e in q["evidence_list"]:
            key = (e["url"], e["fact"].strip())
            if key in chunk_cache:
                continue
            chunk_cache[key] = builder.build(by_url[e["url"]], e["fact"])
            if chunk_cache[key] is not None and chunk_cache[key]["text"] not in pool_key:
                pool_key[chunk_cache[key]["text"]] = len(pool)
                pool.append({"url": e["url"], "text": chunk_cache[key]["text"]})
    bm25 = BM25([p["text"] for p in pool])

    records, reasons = [], Counter()
    for qi, q in enumerate(queries):
        qtype = q["question_type"].replace("_query", "")
        ev = q["evidence_list"]
        if qtype == "null":
            reasons["null_query"] += 1
            continue
        if len(ev) != 2:
            reasons[f"evidence_count_{len(ev)}"] += 1
            continue
        chunks = [chunk_cache[(e["url"], e["fact"].strip())] for e in ev]
        if any(c is None for c in chunks):
            reasons["fact_not_found_or_too_long"] += 1
            continue
        if ev[0]["url"] == ev[1]["url"]:
            reasons["both_evidence_same_article"] += 1
            continue
        if not normalize_answer(q["answer"]):
            reasons["empty_answer"] += 1
            continue
        gold_urls = {e["url"] for e in ev}
        cand = [(bm25.score(q["query"], i), i) for i, p in enumerate(pool) if p["url"] not in gold_urls]
        cand.sort(key=lambda x: (-x[0], x[1]))
        distractors = [pool[i] for _, i in cand[:N_DISTRACTORS]]
        sid = f"mhrag-{qi:04d}"
        docs = [{"doc_id": f"{sid}#e{k}", "text": c["text"], "gold": True, "fact": e["fact"].strip(),
                 "source": e["source"], "date": e["published_at"], "url": e["url"]} for k, (e, c) in enumerate(zip(ev, chunks))]
        docs += [{"doc_id": f"{sid}#d{k}", "text": d["text"], "gold": False, "url": d["url"], "bm25": round(s, 3)}
                 for k, (d, (s, _)) in enumerate(zip(distractors, cand[:N_DISTRACTORS]))]
        random.Random(f"{SEED}:{sid}").shuffle(docs)
        dates = sorted(datetime.datetime.fromisoformat(e["published_at"].replace("Z", "+00:00")) for e in ev)
        records.append({
            "sample_id": sid, "question": q["query"], "reference_answers": [q["answer"]],
            "gold_doc_ids": [d["doc_id"] for d in docs if d["gold"]],
            "documents": [{"doc_id": d["doc_id"], "text": d["text"]} for d in docs],
            "evidence_sentences": {d["doc_id"]: [d["fact"]] for d in docs if d["gold"]},
            "annotation": {
                "question_type": qtype, "answer_kind": answer_kind(q["answer"], qtype),
                "answer_is_yes_no": normalize_answer(q["answer"]) in ("yes", "no"),
                "sources": [e["source"] for e in ev], "same_source": ev[0]["source"] == ev[1]["source"],
                "date_gap_days": (dates[1] - dates[0]).days,
                "evidence": [{"doc_id": d["doc_id"], "source": d["source"], "date": d["date"], "url": d["url"]}
                             for d in docs if d["gold"]],
                "distractor_bm25": [d["bm25"] for d in docs if not d["gold"]],
                "chunk_words": [[chunk_cache[(e["url"], e["fact"].strip())][k] for k in ("words_before", "words_after")]
                                for e in ev],
            },
        })

    manifest_id = f"multihoprag_2ev_gold2_dist3_n{len(records)}"
    fixture = {
        "fixture_id": "multihoprag_2ev",
        "source": {"dataset": REPO, "split": "all (single file)", "queries": q_source, "corpus": c_source,
                   "license": "ODC-By",
                   "selection": {"rule": "question_type != null, exactly 2 evidence, facts verbatim, distinct articles",
                                 "rows": len(queries), "eligible": len(records), "excluded": dict(reasons),
                                 "selected_ids": [r["sample_id"] for r in records]}},
        "preprocessing": {"doc_text": "'{title} ({source}, {Month D, YYYY}): ' + fact-centred window of the article body "
                                      "grown symmetrically (words) until the 128-token PISCO window is full",
                          "distractors": f"BM25(k1=1.5,b=0.75) top-{N_DISTRACTORS} over {len(pool)} evidence chunks of the "
                                         "whole dataset, excluding the query's evidence articles"},
        "context_policy": {"protocol": "fixed_context_research", "context_mode": "gold2_plus_distractors",
                           "gold_policy": "both evidence chunks included",
                           "distractor_rule": f"BM25 top-{N_DISTRACTORS} same-domain chunks",
                           "order": "seeded per-sample shuffle: random.Random(f'{seed}:{sample_id}').shuffle"},
        "samples": records,
    }
    fixture_path = OUT_DIR / "multihoprag_2ev_gold2_dist3_fixture.json"
    fixture_path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False))
    manifest = eval_inputs.build_manifest(fixture_path, tok, tok_source, manifest_id)
    manifest_path = PROJECT_ROOT / "data" / "manifests" / f"{manifest_id}.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if eval_inputs.sha256_json(old["samples"]) != eval_inputs.sha256_json(manifest["samples"]):
            sys.exit(f"{manifest_path} exists with different samples; refusing to overwrite")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    docs = [p for s in manifest["samples"] for p in s["provenance"]["documents"]]
    ann = [s["annotation"] for s in manifest["samples"]]
    print(json.dumps({
        "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "rows": len(queries), "eligible": len(records), "excluded": dict(reasons), "pool_chunks": len(pool),
        "types": dict(Counter(a["question_type"] for a in ann)),
        "answer_kind_by_type": {t: dict(Counter(a["answer_kind"] for a in ann if a["question_type"] == t))
                                for t in sorted({a["question_type"] for a in ann})},
        "same_source": sum(a["same_source"] for a in ann),
        "gold_slot_counts": sorted(Counter(p for a in ann for p in a["gold_positions"]).items()),
        "docs_truncated": f"{sum(d['truncated'] for d in docs)}/{len(docs)}",
        "samples_all_evidence_preserved": sum(a["all_evidence_preserved"] for a in ann),
        "doc_tokens_median": sorted(d["n_tokens_used_by_pisco"] for d in docs)[len(docs) // 2],
        "example": manifest["samples"][0]["model_input"],
    }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
