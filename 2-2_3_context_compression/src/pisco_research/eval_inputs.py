"""Common evaluation input contract: fixture -> fixed manifest consumed by every system.

Each manifest sample has three parts:
  model_input   question + ordered documents. The ONLY part a system may read.
                documents[].text       original text (PISCO input; PISCO truncates it itself)
                documents[].used_text  exactly the content PISCO reads after its
                                       128-token truncation (Raw input)
  evaluation    reference answers, gold doc ids (scoring only)
  provenance    hashes, token counts, truncation flags, expected PISCO encoder ids

`used_text` exists so Raw never reads text that PISCO cut off. Decode ->
re-tokenize is not assumed to be lossless: the PISCO encoder ids are stored
and re-checked against the real model at prediction time.
"""

import hashlib
import json
from pathlib import Path

DOC_MAX_TOKENS = 128                   # official compress path hardcodes max_length=128 (L983, L1031)
ENCODER_PADDED_LENGTH = DOC_MAX_TOKENS + 3   # "<ENC>" + "<s>" + text + "</s>", padded/truncated (L705)
PISCO_ADDED_TOKENS = [f"<MEM{i}>" for i in range(8)] + ["<AE>", "<ENC>", "<SEP>"]  # order of L576
MEM_PER_DOC = 8


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(obj):
    return sha256_text(json.dumps(obj, sort_keys=True, ensure_ascii=False))


def load_pisco_tokenizer(backbone_dir):
    """Backbone tokenizer extended exactly like COCOM.create_decoder_tokenizer (L568-599)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(backbone_dir, use_fast=True, padding_side="left")
    tok.add_special_tokens({"additional_special_tokens": PISCO_ADDED_TOKENS})
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.bos_token_id
    return tok


def pisco_encoder_span(tokenizer, text):
    """One document through the official encoder formatting (L704-712).

    Returns:
      encoder_ids  [139] ids the official path should produce (padding + text + 8 MEM ids)
      text_ids     the document tokens PISCO actually reads (between <s> and </s>, or to the cut)
      truncated    True if the 131-token limit cut the document (then </s> is cut too)
    """
    enc = tokenizer(["<ENC>" + tokenizer.bos_token + text + tokenizer.eos_token],
                    padding="max_length", max_length=ENCODER_PADDED_LENGTH,
                    truncation=True, add_special_tokens=False)
    ids, mask = enc["input_ids"][0], enc["attention_mask"][0]
    first = mask.index(1)
    if ids[first] != tokenizer.convert_tokens_to_ids("<ENC>") or ids[first + 1] != tokenizer.bos_token_id:
        raise ValueError("unexpected encoder prefix")
    body = ids[first + 2:]
    truncated = body[-1] != tokenizer.eos_token_id
    text_ids = body if truncated else body[:-1]
    mem_ids = tokenizer.convert_tokens_to_ids(PISCO_ADDED_TOKENS[:MEM_PER_DOC])
    return ids + mem_ids, text_ids, truncated


def build_manifest(fixture_path, tokenizer, tokenizer_source, manifest_id):
    fixture_path = Path(fixture_path)
    fixture = json.loads(fixture_path.read_text())
    samples = []
    for s in fixture["samples"]:
        model_docs, provenance = [], []
        for d in s["documents"]:
            encoder_ids, text_ids, truncated = pisco_encoder_span(tokenizer, d["text"])
            used_text = tokenizer.decode(text_ids)
            model_docs.append({"doc_id": d["doc_id"], "chunk_id": f"{d['doc_id']}#0",
                               "text": d["text"], "used_text": used_text})
            provenance.append({
                "doc_id": d["doc_id"],
                "original_text_sha256": sha256_text(d["text"]), "used_text_sha256": sha256_text(used_text),
                "n_tokens_standalone": len(tokenizer(d["text"], add_special_tokens=False)["input_ids"]),
                "n_tokens_used_by_pisco": len(text_ids),
                "truncated": truncated,
                # decode is not guaranteed lossless; record how used_text relates to the original
                "used_text_equals_original": used_text == d["text"],
                "used_text_is_prefix_of_original": d["text"].startswith(used_text),
                "pisco_encoder_ids_sha256": sha256_json(encoder_ids),
            })
        model_input = {"question": s["question"], "documents": model_docs}
        gold = s["gold_doc_ids"]
        doc_ids = [d["doc_id"] for d in s["documents"]]
        annotation = {
            "gold_positions": [doc_ids.index(g) for g in gold],
            "gold_not_truncated": all(not p["truncated"] for p in provenance if p["doc_id"] in gold),
            **s.get("annotation", {}),
        }
        # Optional evidence sentences per doc: is each one still fully inside what PISCO reads?
        if "evidence_sentences" in s:
            used = {d["doc_id"]: d["used_text"] for d in model_docs}
            preserved = {doc_id: all(sent.strip() in used[doc_id] for sent in sents)
                         for doc_id, sents in s["evidence_sentences"].items()}
            annotation["evidence_preserved_per_doc"] = preserved
            annotation["all_evidence_preserved"] = all(preserved.values())
        samples.append({
            "sample_id": s["sample_id"],
            "model_input": model_input,
            "model_input_sha256": sha256_json(model_input),
            "evaluation": {"reference_answers": s["reference_answers"], "gold_doc_ids": gold},
            "provenance": {"documents": provenance},
            "annotation": annotation,
        })

    ks = {len(s["model_input"]["documents"]) for s in samples}
    if len(ks) != 1:
        raise ValueError(f"k_final differs across samples: {sorted(ks)}")
    k = ks.pop()
    return {
        "manifest_id": manifest_id,
        "source": {"dataset": fixture["fixture_id"], "split": "fixture", "path": str(fixture_path.name),
                   "revision_sha256": sha256_text(fixture_path.read_text()), **fixture.get("source", {})},
        "tokenizer": tokenizer_source,
        "preprocessing": {"doc_max_tokens": DOC_MAX_TOKENS,
                          "pisco_encoder_format": "<ENC><s>{text}</s> + <MEM0..7>, left pad to 131, right truncation",
                          "raw_reads": "used_text (decoded PISCO-read span)", **fixture.get("preprocessing", {})},
        "context_policy": {**fixture["context_policy"], "k_final": k} if "context_policy" in fixture else {
            "context_mode": "fixture_fixed", "k_final": k, "order": "fixture order",
            "gold_policy": "gold included by fixture construction (no insertion/replacement)"},
        "budget": {"memory_per_doc": MEM_PER_DOC, "total_memory": MEM_PER_DOC * k,
                   "note": "decoder total length (instruction + separators + question) is recorded per prediction"},
        "samples": samples,
    }
