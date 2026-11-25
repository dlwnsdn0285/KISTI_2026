"""Step 10a: MuSiQue hop-decomposition probes on the Step 09c contexts.

    ~/anaconda3/envs/pisco-ref/bin/python scripts/10a_prepare_musique_hops.py

Question: when PISCO fails a 2-hop question with both gold paragraphs present,
is the bridge entity lost in compression (hop 1 fails), or is each hop
answerable alone and only the composition fails?

For each of the 1000 MuSiQue 2-hop samples of Step 09c, the SAME five documents
(same order, same text) are kept and only the question changes:

  hop1  first sub-question                  -> reference: hop-1 answer (bridge entity)
  hop2  second sub-question, "#1" replaced  -> reference: final answer + aliases
        by the gold hop-1 answer
  (composed = original question = the Step 09c runs, reused)

PISCO compresses documents without the question, so the memory is identical in
all three conditions; only the decoder question differs.

KB-style sub-questions "<subject> >> <relation>" are rendered with one fixed
template: "What is the <relation> of <subject>?". NL sub-questions are used
verbatim. Both systems receive the same question text.

Outputs: data/multihop_2hop/musique_2hop_{hop1,hop2}_n1000_fixture.json,
         data/manifests/musique_2hop_{hop1,hop2}_n1000.json
"""

import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pisco_research import eval_inputs  # noqa: E402
from pisco_research.scoring import normalize_answer  # noqa: E402

BASE_ID = "musique_2hop_gold2_dist3_n1000"
BASE_FIXTURE = PROJECT_ROOT / "data" / "multihop_2hop" / f"{BASE_ID}_fixture.json"
BASE_MANIFEST = PROJECT_ROOT / "data" / "manifests" / f"{BASE_ID}.json"
OUT_DIR = PROJECT_ROOT / "data" / "multihop_2hop"
REFERENCE_MANIFEST = PROJECT_ROOT / "reference" / "manifest.json"
KB_TEMPLATE = "What is the {relation} of {subject}?"


def source_path(source):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(source["repo_id"], source["file"], repo_type="dataset",
                           revision=source["revision"], token=False)
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != source["sha256"]:
        raise RuntimeError("MuSiQue source file differs from the one Step 09c used")
    return path


def render(sub_question):
    """KB triple '<subject> >> <relation>' -> template; NL question -> verbatim."""
    if ">>" in sub_question:
        subject, relation = (x.strip() for x in sub_question.split(">>", 1))
        return KB_TEMPLATE.format(relation=relation, subject=subject), "KB"
    return sub_question.strip(), "NL"


def answer_in(text, answers):
    t = normalize_answer(text)
    return any(normalize_answer(a) and normalize_answer(a) in t for a in answers)


def main():
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ["HF_HUB_OFFLINE"] = "1"
    base = json.loads(BASE_FIXTURE.read_text())
    rows = {(r := json.loads(line))["id"]: r for line in open(source_path(base["source"]))}
    base_manifest = {s["sample_id"]: s for s in json.loads(BASE_MANIFEST.read_text())["samples"]}

    probes = {"hop1": [], "hop2": []}
    for s in base["samples"]:
        row = rows[s["sample_id"]]
        d1, d2 = row["question_decomposition"]
        if "#1" not in d2["question"] or "#1" in d1["question"] or d2["answer"] != row["answer"]:
            raise ValueError(f"{s['sample_id']}: unexpected decomposition layout")
        q1, kind1 = render(d1["question"])
        q2, kind2 = render(d2["question"].replace("#1", d1["answer"]))
        hop_doc = {"hop1": f"{s['sample_id']}#p{d1['paragraph_support_idx']}",
                   "hop2": f"{s['sample_id']}#p{d2['paragraph_support_idx']}"}
        if set(hop_doc.values()) != set(s["gold_doc_ids"]):
            raise ValueError(f"{s['sample_id']}: hop paragraphs != gold paragraphs")
        used = {d["doc_id"]: d["used_text"] for d in base_manifest[s["sample_id"]]["model_input"]["documents"]}
        refs = {"hop1": [d1["answer"]], "hop2": s["reference_answers"]}
        common_ann = {
            **s["annotation"], "hop_kinds": f"{kind1}->{kind2}",
            "original_question": s["question"],
            "sub_questions_raw": [d1["question"], d2["question"]], "sub_answers": [d1["answer"], d2["answer"]],
            "hop_doc_ids": hop_doc,
            # does the hop answer survive the 128-token cut of its own supporting paragraph?
            "hop1_answer_in_used_text": answer_in(used[hop_doc["hop1"]], refs["hop1"]),
            "hop2_answer_in_used_text": answer_in(used[hop_doc["hop2"]], refs["hop2"]),
        }
        for probe, question in (("hop1", q1), ("hop2", q2)):
            if not any(normalize_answer(a) for a in refs[probe]):
                raise ValueError(f"{s['sample_id']}: empty {probe} reference")
            probes[probe].append({
                "sample_id": s["sample_id"], "question": question,
                "reference_answers": [a for a in refs[probe] if normalize_answer(a)],
                "gold_doc_ids": s["gold_doc_ids"], "documents": s["documents"],
                "annotation": {**common_ann, "probe": probe, "probe_kind": kind1 if probe == "hop1" else kind2,
                               "probe_doc_id": hop_doc[probe]},
            })

    ref = json.loads(REFERENCE_MANIFEST.read_text())
    tok = eval_inputs.load_pisco_tokenizer(ref["snapshots"]["backbone"]["local_dir"])
    tok_source = {"repo_id": ref["backbone"]["repo_id"], "revision": ref["backbone"]["revision_sha"],
                  "added_special_tokens": eval_inputs.PISCO_ADDED_TOKENS}

    for probe, samples in probes.items():
        manifest_id = f"musique_2hop_{probe}_n{len(samples)}"
        fixture = {
            "fixture_id": f"musique_dev_2hop_{probe}",
            "source": {**base["source"], "derived_from": {"fixture": BASE_FIXTURE.name,
                                                          "sha256": hashlib.sha256(BASE_FIXTURE.read_bytes()).hexdigest()}},
            "preprocessing": {**base["preprocessing"],
                              "question": {"hop1": "first sub-question",
                                           "hop2": "second sub-question with '#1' -> gold hop-1 answer"}[probe],
                              "kb_template": KB_TEMPLATE},
            "context_policy": base["context_policy"],
            "samples": samples,
        }
        fixture_path = OUT_DIR / f"{manifest_id}_fixture.json"
        fixture_path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False))
        manifest = eval_inputs.build_manifest(fixture_path, tok, tok_source, manifest_id)

        # documents (text, used_text, PISCO encoder ids) must be identical to Step 09c
        for s in manifest["samples"]:
            b = base_manifest[s["sample_id"]]
            if s["model_input"]["documents"] != b["model_input"]["documents"] or \
                    s["provenance"] != b["provenance"]:
                raise RuntimeError(f"{s['sample_id']}: documents differ from Step 09c manifest")

        manifest_path = PROJECT_ROOT / "data" / "manifests" / f"{manifest_id}.json"
        if manifest_path.exists():
            old = json.loads(manifest_path.read_text())
            if eval_inputs.sha256_json(old["samples"]) != eval_inputs.sha256_json(manifest["samples"]):
                sys.exit(f"{manifest_path} exists with different samples; refusing to overwrite")
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        ann = [s["annotation"] for s in manifest["samples"]]
        print(json.dumps({
            "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
            "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "n": len(ann), "documents_identical_to_09c": True,
            "probe_kind": dict(Counter(a["probe_kind"] for a in ann)),
            "answer_in_used_text": sum(a[f"{probe}_answer_in_used_text"] for a in ann),
            "examples": [s["model_input"]["question"] + " => " + " | ".join(s["evaluation"]["reference_answers"])
                         for s in manifest["samples"][:4]],
        }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()