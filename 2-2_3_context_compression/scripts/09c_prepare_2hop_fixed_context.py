"""Step 09c: 2-hop fixed-context manifests for HotpotQA, 2WikiMultihopQA, MuSiQue.

    ~/anaconda3/envs/pisco-ref/bin/python scripts/09c_prepare_2hop_fixed_context.py

Same context policy for all three datasets:
  * 2-hop = exactly 2 supporting paragraphs (HotpotQA: all; 2Wiki: compositional,
    comparison, inference; MuSiQue: "2hop" ids, answerable)
  * k = 5: both gold paragraphs + the first 3 non-gold paragraphs in the
    dataset's paragraph order, shuffled with random.Random(f"{SEED}:{sample_id}")
  * document text: "{title}: {paragraph}" (HotpotQA/2Wiki sentences stripped and
    joined with single spaces)
  * N=1000 per dataset: random.Random(SEED).sample over eligible rows, sorted.
    Eligibility is data integrity only; no model output is used.

Outputs: data/multihop_2hop/<id>_fixture.json, data/manifests/<id>.json
"""

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

SEED, N_SAMPLES, N_DISTRACTORS = 0, 1000, 3
OUT_DIR = PROJECT_ROOT / "data" / "multihop_2hop"
REFERENCE_MANIFEST = PROJECT_ROOT / "reference" / "manifest.json"
SOURCES = {
    "hotpotqa": ("hotpotqa/hotpot_qa", "1908d6afbbead072334abe2965f91bd2709910ab",
                 "distractor/validation-00000-of-00001.parquet"),
    "2wiki": ("framolfese/2WikiMultihopQA", "fe713bfbd1afbca1a65246741a75890405d56a3a",
              "data/validation-00000-of-00001.parquet"),
    "musique": ("dgslibisey/MuSiQue", "c8f4f8c9465fb69d31a8eae894c3fd509c4ca321", "musique_ans_v1.0_dev.jsonl"),
}


def fetch(name):
    from huggingface_hub import HfApi, hf_hub_download
    repo, revision, filename = SOURCES[name]
    path = hf_hub_download(repo, filename, repo_type="dataset", revision=revision, token=False)
    info = HfApi().dataset_info(repo, revision=revision, files_metadata=True, token=False)
    remote = next(s for s in info.siblings if s.rfilename == filename)
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if remote.lfs is None or digest != remote.lfs.sha256:
        raise RuntimeError(f"{repo}/{filename}: sha256 mismatch")
    return path, {"repo_id": repo, "revision": revision, "file": filename, "sha256": digest}


# ------------------------------------------------- per-dataset normalization
# Each loader yields a common record:
#   id, question, answers, type, paragraphs [(title, text)], gold_idx [i, j],
#   evidence {paragraph index: [supporting sentence, ...]} (empty for MuSiQue), reason (None if eligible)

def from_hotpot_style(row, qtype):
    """HotpotQA and 2Wiki share the {context: {title, sentences}, supporting_facts} layout."""
    titles, sentences = row["context"]["title"], row["context"]["sentences"]
    gold_titles = set(row["supporting_facts"]["title"])
    gold_idx = [i for i, t in enumerate(titles) if t in gold_titles]
    rec = {"id": row["id"], "question": row["question"], "answers": [row["answer"]], "type": qtype,
           "paragraphs": [(t, " ".join(s.strip() for s in sents)) for t, sents in zip(titles, sentences)],
           "gold_idx": gold_idx, "evidence": {}, "reason": None}
    if len(gold_titles) != 2 or len(gold_idx) != 2:
        rec["reason"] = "not_exactly_2_gold_paragraphs"
        return rec
    for title, sid in zip(row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"]):
        i = titles.index(title)
        if sid >= len(sentences[i]):
            rec["reason"] = "supporting_sentence_index_out_of_range"
            return rec
        rec["evidence"].setdefault(i, []).append(sentences[i][sid].strip())
    return rec


def from_musique(row):
    paras = row["paragraphs"]
    gold_idx = [i for i, p in enumerate(paras) if p["is_supporting"]]
    rec = {"id": row["id"], "question": row["question"], "answers": [row["answer"], *row["answer_aliases"]],
           "type": row["id"].split("__")[0], "paragraphs": [(p["title"], p["paragraph_text"].strip()) for p in paras],
           "gold_idx": gold_idx, "evidence": {}, "reason": None}
    if not row["id"].startswith("2hop__"):
        rec["reason"] = "not_2hop"
    elif not row["answerable"]:
        rec["reason"] = "not_answerable"
    elif len(gold_idx) != 2:
        rec["reason"] = "not_exactly_2_gold_paragraphs"
    return rec


def load_records(name, path):
    if name == "musique":
        return [from_musique(json.loads(line)) for line in open(path)]
    import pyarrow.parquet as pq
    return [from_hotpot_style(r, r["type"]) for r in pq.read_table(path).to_pylist()]


def finalize_reason(rec):
    if rec["reason"] is None and len(rec["paragraphs"]) - 2 < N_DISTRACTORS:
        rec["reason"] = "fewer_than_3_distractors"
    if rec["reason"] is None and not any(normalize_answer(a) for a in rec["answers"]):
        rec["reason"] = "empty_answer_after_normalization"
    return rec["reason"]


def to_sample(rec):
    gold = rec["gold_idx"]
    distractors = [i for i in range(len(rec["paragraphs"])) if i not in gold][:N_DISTRACTORS]
    chosen = sorted(gold + distractors)
    random.Random(f"{SEED}:{rec['id']}").shuffle(chosen)

    def doc_id(i):
        return f"{rec['id']}#p{i}"
    sample = {
        "sample_id": rec["id"], "question": rec["question"],
        "reference_answers": [a for a in rec["answers"] if normalize_answer(a)],
        "gold_doc_ids": [doc_id(i) for i in gold],
        "documents": [{"doc_id": doc_id(i), "text": f"{rec['paragraphs'][i][0]}: {rec['paragraphs'][i][1]}"} for i in chosen],
        "annotation": {"question_type": rec["type"],
                       "answer_is_yes_no": any(normalize_answer(a) in ("yes", "no") for a in rec["answers"]),
                       "source_paragraph_indices": chosen},
    }
    if rec["evidence"]:
        sample["evidence_sentences"] = {doc_id(i): sents for i, sents in rec["evidence"].items()}
    return sample


def main():
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    built = {}
    for name in SOURCES:
        path, source = fetch(name)
        records = load_records(name, path)
        reasons = [finalize_reason(r) for r in records]
        eligible = [r for r, why in zip(records, reasons) if why is None]
        chosen = sorted(random.Random(SEED).sample(range(len(eligible)), N_SAMPLES))
        built[name] = {"source": source, "records": [eligible[i] for i in chosen],
                       "counts": {"rows": len(records), "eligible": len(eligible),
                                  "excluded": dict(Counter(r for r in reasons if r))}}

    os.environ["HF_HUB_OFFLINE"] = "1"
    ref = json.loads(REFERENCE_MANIFEST.read_text())
    tok = eval_inputs.load_pisco_tokenizer(ref["snapshots"]["backbone"]["local_dir"])
    tok_source = {"repo_id": ref["backbone"]["repo_id"], "revision": ref["backbone"]["revision_sha"],
                  "added_special_tokens": eval_inputs.PISCO_ADDED_TOKENS}

    for name, b in built.items():
        manifest_id = f"{name}_2hop_gold2_dist3_n{N_SAMPLES}"
        fixture = {
            "fixture_id": f"{name}_dev_2hop",
            "source": {"dataset": b["source"]["repo_id"], "split": "dev/validation", **b["source"],
                       "selection": {"seed": SEED, "n": N_SAMPLES, **b["counts"],
                                     "rule": "random.Random(seed).sample over eligible rows in file order, then sorted",
                                     "selected_ids": [r["id"] for r in b["records"]]}},
            "preprocessing": {"doc_text": "'{title}: {paragraph}' (sentences stripped, joined by single spaces)"},
            "context_policy": {"protocol": "fixed_context_research", "context_mode": "gold2_plus_distractors",
                               "gold_policy": "both supporting paragraphs included",
                               "distractor_rule": f"first {N_DISTRACTORS} non-gold paragraphs in dataset order",
                               "order": "seeded per-sample shuffle: random.Random(f'{seed}:{sample_id}').shuffle"},
            "samples": [to_sample(r) for r in b["records"]],
        }
        fixture_path = OUT_DIR / f"{manifest_id}_fixture.json"
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
            **b["counts"], "types": dict(Counter(a["question_type"] for a in ann)),
            "yes_no": sum(a["answer_is_yes_no"] for a in ann),
            "gold_slot_counts": sorted(Counter(p for a in ann for p in a["gold_positions"]).items()),
            "docs_truncated": f"{sum(d['truncated'] for d in docs)}/{len(docs)}",
            "samples_gold_not_truncated": sum(a["gold_not_truncated"] for a in ann),
            "samples_all_evidence_preserved": sum(a.get("all_evidence_preserved", False) for a in ann)
            if "all_evidence_preserved" in ann[0] else "n/a (no sentence-level evidence)",
            "used_text_not_prefix": sum(not d["used_text_is_prefix_of_original"] for d in docs),
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
