"""Step 09d: NQ fixed-context manifest analogous to Step 09c (gold in the top-5).

    ~/anaconda3/envs/pisco-ref/bin/python scripts/09d_prepare_nq_gold_fixed_context.py

Source: official DPR NQ dev (biencoder-nq-dev.json.gz, downloaded to data/nq_dpr/).
  * gold = positive_ctxs[0] (the annotated NQ passage; DPR score 1000)
  * distractors = the first 4 hard_negative_ctxs (BM25-ranked passages that do
    not contain the answer)
  * k = 5, shuffled with random.Random(f"{SEED}:{sample_id}"); text "{title}: {text}"
  * eligibility: >= 1 positive, >= 4 hard negatives, >= 1 answer non-empty after
    normalization; N=1000 via random.Random(SEED).sample, sorted
Not a fair counterpart of the 2-hop sets: 1 gold + 4 distractors vs 2 + 3.

`all_evidence_preserved` here means: some reference answer string survives in
the gold passage's 128-token used_text (sentence-level evidence is not available).

Output: data/manifests/nq_gold1_dist4_n1000.json
"""

import gzip
import hashlib
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pisco_research import eval_inputs  # noqa: E402
from pisco_research.scoring import normalize_answer  # noqa: E402

SOURCE_URL = "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz"
SOURCE_PATH = PROJECT_ROOT / "data" / "nq_dpr" / "biencoder-nq-dev.json.gz"
SOURCE_SHA256 = "b583e78adcad00fbd4b421f691fac60e477c30a1b142fbcba1cf24b3a782f621"  # recorded at first download
SEED, N_SAMPLES, N_DISTRACTORS = 0, 1000, 4
MANIFEST_ID = f"nq_gold1_dist{N_DISTRACTORS}_n{N_SAMPLES}"
REFERENCE_MANIFEST = PROJECT_ROOT / "reference" / "manifest.json"


def eligibility(row):
    if not row["positive_ctxs"]:
        return "no_positive"
    if len(row["hard_negative_ctxs"]) < N_DISTRACTORS:
        return "fewer_than_4_hard_negatives"
    if not any(normalize_answer(a) for a in row["answers"]):
        return "empty_answer_after_normalization"
    return None


def to_sample(i, row):
    sample_id = f"nq-dev-{i}"
    ctxs = [("gold", row["positive_ctxs"][0])] + [("hn", c) for c in row["hard_negative_ctxs"][:N_DISTRACTORS]]
    order = list(range(len(ctxs)))
    random.Random(f"{SEED}:{sample_id}").shuffle(order)
    docs = [{"doc_id": f"psg{ctxs[j][1]['passage_id']}", "text": f"{ctxs[j][1]['title']}: {ctxs[j][1]['text']}"} for j in order]
    return {
        "sample_id": sample_id, "question": row["question"],
        "reference_answers": [a for a in row["answers"] if normalize_answer(a)],
        "gold_doc_ids": [f"psg{row['positive_ctxs'][0]['passage_id']}"],
        "documents": docs,
        "annotation": {"question_type": "nq_single_hop", "gold_dpr_score": row["positive_ctxs"][0]["score"],
                       "answer_is_yes_no": False, "source_row": i},
    }


def main():
    if not SOURCE_PATH.exists():
        sys.exit(f"download {SOURCE_URL} to {SOURCE_PATH} first")
    digest = hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        sys.exit(f"source sha256 changed: {digest}")
    rows = json.load(gzip.open(SOURCE_PATH))
    reasons = [eligibility(r) for r in rows]
    eligible = [i for i, why in enumerate(reasons) if why is None]
    chosen = sorted(random.Random(SEED).sample(eligible, N_SAMPLES))
    samples = [to_sample(i, rows[i]) for i in chosen]

    fixture = {
        "fixture_id": "nq_dpr_dev",
        "source": {"dataset": "DPR Natural Questions (biencoder-nq-dev)", "split": "dev", "url": SOURCE_URL,
                   "sha256": digest,
                   "selection": {"seed": SEED, "n": N_SAMPLES, "rows": len(rows), "eligible": len(eligible),
                                 "excluded": dict(Counter(r for r in reasons if r)),
                                 "rule": "random.Random(seed).sample over eligible row indices, sorted",
                                 "selected_rows": chosen}},
        "preprocessing": {"doc_text": "'{title}: {text}' (DPR 100-word passage)"},
        "context_policy": {"protocol": "fixed_context_research", "context_mode": "gold1_plus_hard_negatives",
                           "gold_policy": "positive_ctxs[0] included",
                           "distractor_rule": f"first {N_DISTRACTORS} hard_negative_ctxs (BM25 order)",
                           "order": "seeded per-sample shuffle: random.Random(f'{seed}:{sample_id}').shuffle"},
        "samples": samples,
    }
    fixture_path = SOURCE_PATH.parent / f"{MANIFEST_ID}_fixture.json"
    fixture_path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False))

    os.environ["HF_HUB_OFFLINE"] = "1"
    ref = json.loads(REFERENCE_MANIFEST.read_text())
    tok = eval_inputs.load_pisco_tokenizer(ref["snapshots"]["backbone"]["local_dir"])
    tok_source = {"repo_id": ref["backbone"]["repo_id"], "revision": ref["backbone"]["revision_sha"],
                  "added_special_tokens": eval_inputs.PISCO_ADDED_TOKENS}
    manifest = eval_inputs.build_manifest(fixture_path, tok, tok_source, MANIFEST_ID)
    # Answer-level evidence check on the gold passage (what PISCO / Raw actually read).
    for s in manifest["samples"]:
        gold = next(d for d in s["model_input"]["documents"] if d["doc_id"] in s["evaluation"]["gold_doc_ids"])
        s["annotation"]["all_evidence_preserved"] = any(
            normalize_answer(a) in normalize_answer(gold["used_text"]) for a in s["evaluation"]["reference_answers"])
    manifest_path = PROJECT_ROOT / "data" / "manifests" / f"{MANIFEST_ID}.json"
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
        "rows": len(rows), "eligible": len(eligible), "excluded": fixture["source"]["selection"]["excluded"],
        "gold_slot_counts": sorted(Counter(p for a in ann for p in a["gold_positions"]).items()),
        "gold_dpr_score_1000": sum(a["gold_dpr_score"] == 1000 for a in ann),
        "docs_truncated": f"{sum(d['truncated'] for d in docs)}/{len(docs)}",
        "samples_gold_not_truncated": sum(a["gold_not_truncated"] for a in ann),
        "samples_answer_in_gold_used_text": sum(a["all_evidence_preserved"] for a in ann),
        "used_text_not_prefix": sum(not d["used_text_is_prefix_of_original"] for d in docs),
    }))


if __name__ == "__main__":
    main()
