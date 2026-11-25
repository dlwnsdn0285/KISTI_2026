"""Step 11a: training manifest for the fusion module (HotpotQA train, fixed context).

    ~/anaconda3/envs/pisco-ref/bin/python scripts/11_prepare_fusion_train.py

Same context policy as the Step 09c evaluation sets (2 gold + 3 distractors,
seeded per-sample shuffle, "{title}: {paragraph}" documents), applied to the
HotpotQA *train* split so that no evaluation question is seen. N_TRAIN rows
are sampled with SEED_TRAIN from the eligible rows. Eligibility is data
integrity only (exactly 2 gold paragraphs, >= 3 distractors, non-empty answer).

The manifest is the same contract the evaluation runner uses (used_text for
the teacher, PISCO encoder ids for the student).

Outputs: data/fusion_train/hotpotqa_train_gold2_dist3_n<N>_fixture.json,
         data/manifests/hotpotqa_train_gold2_dist3_n<N>.json
"""

import hashlib
import importlib.util
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pisco_research import eval_inputs  # noqa: E402

SEED_TRAIN, N_TRAIN = 1, 8000
REPO, REVISION = "hotpotqa/hotpot_qa", "1908d6afbbead072334abe2965f91bd2709910ab"
FILES = ["distractor/train-00000-of-00002.parquet", "distractor/train-00001-of-00002.parquet"]
OUT_DIR = PROJECT_ROOT / "data" / "fusion_train"
REFERENCE_MANIFEST = PROJECT_ROOT / "reference" / "manifest.json"

spec = importlib.util.spec_from_file_location("step09c", PROJECT_ROOT / "scripts" / "09c_prepare_2hop_fixed_context.py")
step09c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step09c)


def fetch(filename):
    from huggingface_hub import HfApi, hf_hub_download
    path = hf_hub_download(REPO, filename, repo_type="dataset", revision=REVISION, token=False)
    info = HfApi().dataset_info(REPO, revision=REVISION, files_metadata=True, token=False)
    remote = next(s for s in info.siblings if s.rfilename == filename)
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if remote.lfs is None or digest != remote.lfs.sha256:
        raise RuntimeError(f"{filename}: sha256 mismatch")
    return path, {"repo_id": REPO, "revision": REVISION, "file": filename, "sha256": digest}


def main():
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import pyarrow.parquet as pq
    records, sources = [], []
    for f in FILES:
        path, src = fetch(f)
        sources.append(src)
        records += [step09c.from_hotpot_style(r, r["type"]) for r in pq.read_table(path).to_pylist()]
    reasons = [step09c.finalize_reason(r) for r in records]
    eligible = [r for r, why in zip(records, reasons) if why is None]
    chosen = sorted(random.Random(SEED_TRAIN).sample(range(len(eligible)), N_TRAIN))
    selected = [eligible[i] for i in chosen]

    # no overlap with any evaluation manifest
    eval_ids = set()
    for m in (PROJECT_ROOT / "data" / "manifests").glob("*.json"):
        eval_ids |= {s["sample_id"] for s in json.loads(m.read_text())["samples"]}
    overlap = [r["id"] for r in selected if r["id"] in eval_ids]
    if overlap:
        raise RuntimeError(f"{len(overlap)} training ids overlap evaluation manifests")

    os.environ["HF_HUB_OFFLINE"] = "1"
    ref = json.loads(REFERENCE_MANIFEST.read_text())
    tok = eval_inputs.load_pisco_tokenizer(ref["snapshots"]["backbone"]["local_dir"])
    tok_source = {"repo_id": ref["backbone"]["repo_id"], "revision": ref["backbone"]["revision_sha"],
                  "added_special_tokens": eval_inputs.PISCO_ADDED_TOKENS}
    manifest_id = f"hotpotqa_train_gold2_dist3_n{N_TRAIN}"
    fixture = {
        "fixture_id": "hotpotqa_train_2hop",
        "source": {"dataset": REPO, "split": "train (distractor)", "files": sources,
                   "selection": {"seed": SEED_TRAIN, "n": N_TRAIN, "rows": len(records), "eligible": len(eligible),
                                 "excluded": dict(Counter(r for r in reasons if r)),
                                 "rule": "random.Random(seed).sample over eligible rows in file order, then sorted",
                                 "selected_ids": [r["id"] for r in selected]}},
        "preprocessing": {"doc_text": "'{title}: {paragraph}' (sentences stripped, joined by single spaces)"},
        "context_policy": {"protocol": "fixed_context_research", "context_mode": "gold2_plus_distractors",
                           "gold_policy": "both supporting paragraphs included",
                           "distractor_rule": f"first {step09c.N_DISTRACTORS} non-gold paragraphs in dataset order",
                           "order": "seeded per-sample shuffle: random.Random(f'{seed}:{sample_id}').shuffle",
                           "purpose": "fusion-module training (Step 11); never used for evaluation"},
        "samples": [step09c.to_sample(r) for r in selected],
    }
    fixture_path = OUT_DIR / f"{manifest_id}_fixture.json"
    fixture_path.write_text(json.dumps(fixture, ensure_ascii=False))
    manifest = eval_inputs.build_manifest(fixture_path, tok, tok_source, manifest_id)
    manifest_path = PROJECT_ROOT / "data" / "manifests" / f"{manifest_id}.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if eval_inputs.sha256_json(old["samples"]) != eval_inputs.sha256_json(manifest["samples"]):
            sys.exit(f"{manifest_path} exists with different samples; refusing to overwrite")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False))
    docs = [p for s in manifest["samples"] for p in s["provenance"]["documents"]]
    ann = [s["annotation"] for s in manifest["samples"]]
    print(json.dumps({"manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
                      "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                      "rows": len(records), "eligible": len(eligible), "n": len(ann),
                      "types": dict(Counter(a["question_type"] for a in ann)),
                      "docs_truncated": f"{sum(d['truncated'] for d in docs)}/{len(docs)}",
                      "samples_all_evidence_preserved": sum(a["all_evidence_preserved"] for a in ann)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
