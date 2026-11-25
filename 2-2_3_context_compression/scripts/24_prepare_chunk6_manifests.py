"""Step 24: budget-matched control — the same five used_texts re-split into SIX compression units.

    ~/anaconda3/envs/pisco-ref/bin/python scripts/24_prepare_chunk6_manifests.py [manifest_id ...]
    (no arguments: the six Step 24 manifests below; Step 31 passes the KISTI multi-evidence manifests)

The extra-block methods (Step 19 QA-only, Step 21 residual) give the decoder 5 + 1 = 6 units (48 vectors)
built from the five used_texts (<= 640 tokens). This control gives the decoder the same 6 units / 48 vectors,
produced only by the unchanged PISCO compressor from the same text:

    the five used_texts (manifest order) are tokenised into words (whitespace); the total token count T is
    cut at the 5 targets i*T/6. Each cut goes to the sentence end (Step 21 splitter) closest to the target
    within +-8 tokens if one exists, otherwise to the nearest word boundary. Chunks may cross document
    boundaries. Each chunk must fit PISCO's 128-token window (verified: no truncation).

Documents = the 6 chunks; question, references and annotations unchanged.
Outputs: data/manifests/<id>_chunk6.json for MuSiQue (original, hop1, hop2), MultiHop-RAG (forced, free), NQ.
"""

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pisco_research import eval_inputs  # noqa: E402

spec = importlib.util.spec_from_file_location("imp21", PROJECT_ROOT / "scripts" / "21_teacher_importance.py")
imp21 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imp21)

SOURCES = ["musique_2hop_gold2_dist3_n1000", "musique_2hop_hop1_n1000", "musique_2hop_hop2_n1000",
           "multihoprag_2ev_yesno_forced_n882", "multihoprag_2ev_gold2_dist3_n1061", "nq_gold1_dist4_n1000"]
N_CHUNKS = 6
LIMIT = eval_inputs.DOC_MAX_TOKENS - 4      # leave room for tokenisation differences at chunk joins


SNAP = 8


def chunk_texts(used_texts, ntok):
    """Six contiguous chunks of the word sequence, cut near equal token targets (prefer sentence ends)."""
    words, sent_end = [], []
    for di, text in enumerate(used_texts):
        for a, b in imp21.split_sentences(text):
            ws = text[a:b].split()
            for wi, w in enumerate(ws):
                words.append(w)
                sent_end.append(wi == len(ws) - 1)
    costs = [ntok(" " + w) for w in words]
    cum = [0]
    for c in costs:
        cum.append(cum[-1] + c)
    total = cum[-1]
    cuts, prev = [0], 0
    for i in range(1, N_CHUNKS):
        target = total * i / N_CHUNKS
        lo = prev + 1
        hi = len(words) - (N_CHUNKS - i)
        cands = range(lo, hi + 1)
        near = [j for j in cands if sent_end[j - 1] and abs(cum[j] - target) <= SNAP]
        j = min(near, key=lambda j: abs(cum[j] - target)) if near else min(cands, key=lambda j: abs(cum[j] - target))
        cuts.append(j)
        prev = j
    cuts.append(len(words))
    return [" ".join(words[a:b]) for a, b in zip(cuts, cuts[1:])], cuts


def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    ref = json.loads((PROJECT_ROOT / "reference" / "manifest.json").read_text())
    tok = eval_inputs.load_pisco_tokenizer(ref["snapshots"]["backbone"]["local_dir"])
    tok_source = {"repo_id": ref["backbone"]["repo_id"], "revision": ref["backbone"]["revision_sha"],
                  "added_special_tokens": eval_inputs.PISCO_ADDED_TOKENS}

    def ntok(s):
        return len(tok(s, add_special_tokens=False)["input_ids"])

    for mid in sys.argv[1:] or SOURCES:
        src = json.loads((PROJECT_ROOT / "data" / "manifests" / f"{mid}.json").read_text())
        samples, stats = [], {"max_chunk_tokens": 0, "chunks_crossing_docs": 0, "total_tokens": []}
        for s in src["samples"]:
            used = [d["used_text"] for d in s["model_input"]["documents"]]
            texts, cuts = chunk_texts(used, ntok)
            doc_of_word = [di for di, t in enumerate(used) for a, b in imp21.split_sentences(t) for _ in t[a:b].split()]
            chunks = []
            for ci, (text, a, b) in enumerate(zip(texts, cuts, cuts[1:])):
                docs_in = sorted(set(doc_of_word[a:b]))
                stats["chunks_crossing_docs"] += len(docs_in) > 1
                chunks.append({"doc_id": f"{s['sample_id']}#c{ci}", "text": text, "source_docs": docs_in})
            stats["total_tokens"].append(sum(ntok(c["text"]) for c in chunks))
            gold = set(s["evaluation"]["gold_doc_ids"])
            gold_idx = [i for i, d in enumerate(s["model_input"]["documents"]) if d["doc_id"] in gold]
            gold_chunks = [c["doc_id"] for c in chunks if set(c["source_docs"]) & set(gold_idx)]
            ann = {k: v for k, v in s["annotation"].items() if k not in ("gold_positions", "gold_not_truncated",
                                                                         "evidence_preserved_per_doc", "all_evidence_preserved")}
            samples.append({"sample_id": s["sample_id"], "question": s["model_input"]["question"],
                            "reference_answers": s["evaluation"]["reference_answers"], "gold_doc_ids": gold_chunks,
                            "documents": [{"doc_id": c["doc_id"], "text": c["text"]} for c in chunks],
                            "annotation": {**ann, "chunk_source_docs": [c["source_docs"] for c in chunks],
                                           "original_doc_ids": [d["doc_id"] for d in s["model_input"]["documents"]]}})
        fixture = {"fixture_id": f"{mid}_chunk6", "source": {**src["source"], "derived_from": mid},
                   "preprocessing": {**src["preprocessing"], "chunking": "five used_texts -> words -> 6 contiguous chunks cut near i*T/6 (sentence end within +-8 tokens, else word boundary)"},
                   "context_policy": {**src["context_policy"], "k_final": N_CHUNKS,
                                      "note": "Step 24: budget-matched control (6 units / 48 vectors from the same text)"},
                   "samples": samples}
        fp = PROJECT_ROOT / "data" / "fusion_train" / f"{mid}_chunk6_fixture.json"
        fp.write_text(json.dumps(fixture, ensure_ascii=False))
        manifest = eval_inputs.build_manifest(fp, tok, tok_source, f"{mid}_chunk6")
        trunc = sum(p["truncated"] for x in manifest["samples"] for p in x["provenance"]["documents"])
        if trunc:
            raise RuntimeError(f"{mid}: {trunc} chunks truncated by the 128-token window")
        mx = max(p["n_tokens_used_by_pisco"] for x in manifest["samples"] for p in x["provenance"]["documents"])
        out = PROJECT_ROOT / "data" / "manifests" / f"{mid}_chunk6.json"
        if out.exists():
            old = json.loads(out.read_text())
            if eval_inputs.sha256_json(old["samples"]) != eval_inputs.sha256_json(manifest["samples"]):
                sys.exit(f"{out} exists with different samples; refusing to overwrite")
        out.write_text(json.dumps(manifest, ensure_ascii=False))
        tt = sorted(stats["total_tokens"])
        print(json.dumps({"manifest": out.name, "sha256": hashlib.sha256(out.read_bytes()).hexdigest()[:12], "n": len(samples),
                          "max_chunk_tokens": mx, "truncated": trunc, "chunks_crossing_docs_per_q": round(stats["chunks_crossing_docs"] / len(samples), 2),
                          "text_tokens_median": tt[len(tt) // 2]}))


if __name__ == "__main__":
    main()
