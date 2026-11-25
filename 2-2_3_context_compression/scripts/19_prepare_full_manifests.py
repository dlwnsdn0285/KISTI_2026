"""Step 19: manifests whose sixth document = the five documents exactly as PISCO read them
(used_text, <= 128 tokens each, joined by newlines, in manifest order), for the full-document
residual generator (capacity pressure: <= 640 tokens -> one 8-vector block).

    ~/anaconda3/envs/pisco-ref/bin/python scripts/19_prepare_full_manifests.py

The sixth document's own PISCO truncation flags are irrelevant: the generator reads its `text`
field (see 08_run_eval.py, input_max_len > 128). Documents 1-5 are unchanged.
Outputs: data/manifests/<id>_full.json for the training manifest and four evaluation manifests.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pisco_research import eval_inputs  # noqa: E402

SOURCES = ["hotpotqa_train_gold2_dist3_n8000", "musique_2hop_gold2_dist3_n1000", "multihoprag_2ev_yesno_forced_n882",
           "multihoprag_2ev_gold2_dist3_n1061", "nq_gold1_dist4_n1000"]
SEP = "\n"


def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    ref = json.loads((PROJECT_ROOT / "reference" / "manifest.json").read_text())
    tok = eval_inputs.load_pisco_tokenizer(ref["snapshots"]["backbone"]["local_dir"])
    tok_source = {"repo_id": ref["backbone"]["repo_id"], "revision": ref["backbone"]["revision_sha"],
                  "added_special_tokens": eval_inputs.PISCO_ADDED_TOKENS}
    for mid in SOURCES:
        src = json.loads((PROJECT_ROOT / "data" / "manifests" / f"{mid}.json").read_text())
        samples, lengths = [], []
        for s in src["samples"]:
            docs = s["model_input"]["documents"]
            full = SEP.join(d["used_text"] for d in docs)
            lengths.append(len(tok(full, add_special_tokens=False)["input_ids"]))
            ann = {k: v for k, v in s["annotation"].items() if k not in ("gold_positions", "gold_not_truncated",
                                                                         "evidence_preserved_per_doc", "all_evidence_preserved")}
            samples.append({"sample_id": s["sample_id"], "question": s["model_input"]["question"],
                            "reference_answers": s["evaluation"]["reference_answers"], "gold_doc_ids": s["evaluation"]["gold_doc_ids"],
                            "documents": [{"doc_id": d["doc_id"], "text": d["text"]} for d in docs] +
                                         [{"doc_id": f"{s['sample_id']}#full", "text": full}],
                            "annotation": {**ann, "full_doc_order": [d["doc_id"] for d in docs]}})
        fixture = {"fixture_id": f"{mid}_full", "source": {**src["source"], "derived_from": mid},
                   "preprocessing": {**src["preprocessing"], "sixth_document": "five used_texts joined by newline"},
                   "context_policy": {**src["context_policy"], "k_final": 6,
                                      "note": "Step 19: sixth document = all five documents as PISCO read them"},
                   "samples": samples}
        fp = PROJECT_ROOT / "data" / "fusion_train" / f"{mid}_full_fixture.json"
        fp.write_text(json.dumps(fixture, ensure_ascii=False))
        manifest = eval_inputs.build_manifest(fp, tok, tok_source, f"{mid}_full")
        out = PROJECT_ROOT / "data" / "manifests" / f"{mid}_full.json"
        if out.exists():
            old = json.loads(out.read_text())
            if eval_inputs.sha256_json(old["samples"]) != eval_inputs.sha256_json(manifest["samples"]):
                sys.exit(f"{out} exists with different samples; refusing to overwrite")
        out.write_text(json.dumps(manifest, ensure_ascii=False))
        src_docs = {s["sample_id"]: s["model_input"]["documents"] for s in src["samples"]}
        for s in manifest["samples"]:
            if s["model_input"]["documents"][:-1] != src_docs[s["sample_id"]]:
                raise RuntimeError(f"{s['sample_id']}: documents 1-5 changed")
        lengths.sort()
        print(json.dumps({"manifest": out.name, "sha256": hashlib.sha256(out.read_bytes()).hexdigest()[:12], "n": len(samples),
                          "full_tokens_median": lengths[len(lengths) // 2], "full_tokens_max": lengths[-1]}))


if __name__ == "__main__":
    main()
