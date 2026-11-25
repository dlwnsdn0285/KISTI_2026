"""Step 10b-forced: the yes/no questions of the MultiHop-RAG manifest with a forced-choice suffix.

    ~/anaconda3/envs/pisco-ref/bin/python scripts/10b_prepare_multihoprag_forced.py

In the free-form run PISCO often answers a yes/no question without any yes/no
word (explains, or says the documents do not say). This control keeps the
SAME five documents (same order, same text; PISCO memory identical) and
appends one sentence to the question so that both systems must commit:

    "<question> Answer yes or no first."

Only samples whose reference is yes/no are kept. Both systems receive the
same question text. This deviates from the paper protocol and is reported
as a control, not as the main number.

Outputs: data/multihoprag/multihoprag_2ev_yesno_forced_fixture.json,
         data/manifests/multihoprag_2ev_yesno_forced_n<N>.json
"""

import hashlib
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pisco_research import eval_inputs  # noqa: E402

BASE_FIXTURE = PROJECT_ROOT / "data" / "multihoprag" / "multihoprag_2ev_gold2_dist3_fixture.json"
BASE_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "multihoprag_2ev_gold2_dist3_n1061.json"
REFERENCE_MANIFEST = PROJECT_ROOT / "reference" / "manifest.json"
SUFFIX = " Answer yes or no first."


def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    base = json.loads(BASE_FIXTURE.read_text())
    base_manifest = {s["sample_id"]: s for s in json.loads(BASE_MANIFEST.read_text())["samples"]}
    samples = []
    for s in base["samples"]:
        if not s["annotation"]["answer_is_yes_no"]:
            continue
        samples.append({**s, "question": s["question"].rstrip() + SUFFIX,
                        "annotation": {**s["annotation"], "original_question": s["question"], "forced_choice": True}})

    ref = json.loads(REFERENCE_MANIFEST.read_text())
    tok = eval_inputs.load_pisco_tokenizer(ref["snapshots"]["backbone"]["local_dir"])
    tok_source = {"repo_id": ref["backbone"]["repo_id"], "revision": ref["backbone"]["revision_sha"],
                  "added_special_tokens": eval_inputs.PISCO_ADDED_TOKENS}
    manifest_id = f"multihoprag_2ev_yesno_forced_n{len(samples)}"
    fixture = {"fixture_id": "multihoprag_2ev_yesno_forced",
               "source": {**base["source"], "derived_from": {"fixture": BASE_FIXTURE.name,
                                                             "sha256": hashlib.sha256(BASE_FIXTURE.read_bytes()).hexdigest()},
                          "selection": {**base["source"]["selection"], "subset": "answer is yes/no",
                                        "selected_ids": [s["sample_id"] for s in samples]}},
               "preprocessing": {**base["preprocessing"], "question_suffix": SUFFIX},
               "context_policy": base["context_policy"], "samples": samples}
    fixture_path = BASE_FIXTURE.parent / "multihoprag_2ev_yesno_forced_fixture.json"
    fixture_path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False))
    manifest = eval_inputs.build_manifest(fixture_path, tok, tok_source, manifest_id)
    for s in manifest["samples"]:
        b = base_manifest[s["sample_id"]]
        if s["model_input"]["documents"] != b["model_input"]["documents"] or s["provenance"] != b["provenance"]:
            raise RuntimeError(f"{s['sample_id']}: documents differ from the base manifest")
    manifest_path = PROJECT_ROOT / "data" / "manifests" / f"{manifest_id}.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if eval_inputs.sha256_json(old["samples"]) != eval_inputs.sha256_json(manifest["samples"]):
            sys.exit(f"{manifest_path} exists with different samples; refusing to overwrite")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps({"manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
                      "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(), "n": len(samples),
                      "documents_identical_to_base": True,
                      "example_question": manifest["samples"][0]["model_input"]["question"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
