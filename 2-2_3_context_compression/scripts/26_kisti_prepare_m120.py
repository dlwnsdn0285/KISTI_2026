"""Step 26 (KISTI): sample-500 top-5 retrieval over the PISCO-sized chunk index -> PISCO manifest.

Restored from this server's lost "Step 24/25 (KISTI)" (see STEP_LOG Step 26); behaviour unchanged.
Same queries, references, query encoding, dedup and manifest format as the child125 index
(scripts/26_kisti_prepare_child125.py); only the retrieval unit changes (125-char Chroma child
chunks -> <=120 Mistral-token chunks from scripts/26_kisti_embed.py).

  # 1) dense retrieval (sentence-transformers + GPU; conda env `refrag`)
  CUDA_VISIBLE_DEVICES=0 ~/anaconda3/envs/refrag/bin/python scripts/26_kisti_prepare_m120.py retrieve
  # 2) manifests (PISCO tokenizer; conda env `pisco-ref`)
  ~/anaconda3/envs/pisco-ref/bin/python scripts/26_kisti_prepare_m120.py manifest [--ko]
  # 3) retrieval comparison with the child125 index (any env)
  python scripts/26_kisti_prepare_m120.py compare
  # add `--index v2` to any command for the KISTI_CHUNKER=v2 index (the chosen baseline index)

Index      /data3/jwlee/kisti_sample_e5_m120[_v2]/blocks/*.{npy,parquet}: KISTI_sample 20,252 papers,
           e5-large-instruct (no instruction, normalized, fp16).
Retrieval  exact inner product on GPU, query embedded without instruction (kisti-rag1
           embed_query), top-5 after removing identical texts.
Gold       a chunk is "gold" when its paper_id equals the question's source paper
           (query_id prefix). The child125 index could only match by title; both are recorded.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("child125", PROJECT_ROOT / "scripts" / "26_kisti_prepare_child125.py")
child125 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(child125)

INDEX_TAG = "v2" if "--index" in sys.argv and sys.argv[sys.argv.index("--index") + 1] == "v2" else ""
INDEX_DIR = Path("/data3/jwlee/kisti_sample_e5_m120" + ("_v2" if INDEX_TAG else ""))
TOP_K = 5
SEARCH_K = 64
OUT_DIR = PROJECT_ROOT / "data" / "kisti"
RETRIEVED = OUT_DIR / f"sample500_e5m120{INDEX_TAG}_top5_retrieved.json"
MANIFEST_ID = f"kisti_sample500_e5m120{INDEX_TAG}_top5"


def load_index():
    import numpy as np
    import pyarrow.parquet as pq
    files = sorted((INDEX_DIR / "blocks").glob("*.parquet"))
    vecs, rows = [], []
    for f in files:
        t = pq.read_table(f, columns=["chunk_id", "paper_id", "section", "n_tokens", "text"]).to_pylist()
        v = np.load(str(f).replace(".parquet", ".npy"))
        if v.shape[0] != len(t):
            sys.exit(f"{f.name}: {v.shape[0]} vectors vs {len(t)} rows")
        vecs.append(v)
        rows.extend(t)
    return np.concatenate(vecs), rows


def cmd_retrieve():
    import pyarrow.parquet as pq
    import torch
    from sentence_transformers import SentenceTransformer

    queries = json.loads(child125.QUERY_FILE.read_text())
    qa = {}
    for p in sorted(child125.QA_DIR.glob("*/*/*.json")):
        for item in json.loads(p.read_text()).get("qas", []):
            qa.setdefault(item["question"].strip(), []).append(
                {"answer": item["answer"]["answer_text"], "level": item.get("level"), "paper": p.stem})
    papers = {r["paper_id"]: r["title"] for r in pq.read_table(INDEX_DIR / "papers.parquet", columns=["paper_id", "title"]).to_pylist()}

    vecs, rows = load_index()
    print(f"index: {len(rows)} chunks", flush=True)
    mat = torch.from_numpy(vecs).to("cuda")
    del vecs
    model = SentenceTransformer(child125.E5_DIR, device="cuda").half()   # same precision as the document side
    qvec = model.encode([q["query"] for q in queries], normalize_embeddings=True, batch_size=32, convert_to_tensor=True)
    top_scores, top_idx = (qvec.to(mat.dtype) @ mat.T).topk(SEARCH_K, dim=1)
    top_scores, top_idx = top_scores.float().cpu().numpy(), top_idx.cpu().numpy()

    samples = []
    for qi, q in enumerate(queries):
        refs = qa.get(q["query"].strip())
        if not refs:
            sys.exit(f"no reference answer for {q['query_id']}")
        paper_id = q["query_id"].rsplit("-", 1)[0]
        gold_title = papers.get(paper_id)
        picked, seen = [], set()
        for j, sc in zip(top_idx[qi], top_scores[qi]):
            r = rows[j]
            if r["text"] in seen:
                continue
            seen.add(r["text"])
            picked.append({"doc_id": r["chunk_id"], "text": r["text"], "score": float(sc), "paper_id": r["paper_id"],
                           "title": papers.get(r["paper_id"]), "section": r["section"], "n_tokens": r["n_tokens"],
                           "from_gold_paper": r["paper_id"] == paper_id,
                           "gold_title_match": gold_title is not None and papers.get(r["paper_id"]) == gold_title})
            if len(picked) == TOP_K:
                break
        if len(picked) < TOP_K:
            sys.exit(f"{q['query_id']}: fewer than {TOP_K} unique chunks in top-{SEARCH_K}")
        samples.append({"sample_id": q["query_id"], "question": q["query"],
                        "reference_answers": [r["answer"] for r in refs][:1],
                        "qa_level": refs[0]["level"], "gold_paper": paper_id, "gold_title": gold_title,
                        "gold_paper_in_index": paper_id in papers,
                        "need_retrieval_koni": q["need_retrieval"], "documents": picked})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {"retrieval": {"index": str(INDEX_DIR), "chunks": len(rows), "query_encoding": "e5 no-instruction (kisti-rag1 embed_query), fp16",
                         "search": "exact inner product, fp16, GPU", "dedup": "identical text", "top_k": TOP_K},
           "samples": samples}
    RETRIEVED.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    gold_any = sum(any(d["from_gold_paper"] for d in s["documents"]) for s in samples)
    print(f"wrote {RETRIEVED}: {len(samples)} samples, gold paper in top-5: {gold_any}, "
          f"gold paper in index: {sum(s['gold_paper_in_index'] for s in samples)}")


def cmd_manifest(ko=False):
    os.environ["HF_HUB_OFFLINE"] = "1"
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from pisco_research import eval_inputs
    retrieved = json.loads(RETRIEVED.read_text())
    ref = json.loads(child125.REFERENCE_MANIFEST.read_text())
    tok = eval_inputs.load_pisco_tokenizer(ref["snapshots"]["backbone"]["local_dir"])
    tok_source = {"repo_id": ref["backbone"]["repo_id"], "revision": ref["backbone"]["revision_sha"],
                  "added_special_tokens": eval_inputs.PISCO_ADDED_TOKENS}
    samples = []
    for s in retrieved["samples"]:
        samples.append({
            "sample_id": s["sample_id"], "question": s["question"] + (child125.KO_SUFFIX if ko else ""),
            "reference_answers": s["reference_answers"],
            "gold_doc_ids": [], "documents": [{"doc_id": d["doc_id"], "text": d["text"]} for d in s["documents"]],
            "annotation": {"gold_not_truncated": None, "qa_level": s["qa_level"], "gold_paper": s["gold_paper"],
                           "gold_paper_in_top5": any(d["from_gold_paper"] for d in s["documents"]),
                           "gold_paper_positions": [i for i, d in enumerate(s["documents"]) if d["from_gold_paper"]],
                           "need_retrieval_koni": s["need_retrieval_koni"]},
        })
    fixture = {
        "fixture_id": f"kisti_sample500_m120{INDEX_TAG}",
        "source": {"dataset": "KISTI 국내 논문 QA (kisti-rag1 500-sample)", "query_file": str(child125.QUERY_FILE),
                   "retrieved_file": str(RETRIEVED.relative_to(PROJECT_ROOT)), "retrieval": retrieved["retrieval"]},
        "preprocessing": {"doc_text": f"Step 23 chunk text, chunker {INDEX_TAG or 'v1'} (<=120 Mistral tokens, sentence-packed, exact section substring)"},
        "context_policy": {"protocol": "kisti_retrieved", "context_mode": "retrieved_only",
                           "retrieval": "e5-large-instruct dense, exact search, top-5 unique",
                           "gold_policy": "none (no gold insertion)", "order": "retrieval score, most relevant first"},
        "samples": samples,
    }
    manifest_id = MANIFEST_ID + ("_ko" if ko else "")
    if ko:
        fixture["preprocessing"]["question"] = f"original question + {child125.KO_SUFFIX!r}"
    fixture_path = OUT_DIR / f"{manifest_id}_fixture.json"
    fixture_path.write_text(json.dumps(fixture, indent=1, ensure_ascii=False))
    manifest = eval_inputs.build_manifest(fixture_path, tok, tok_source, manifest_id)
    manifest_path = PROJECT_ROOT / "data" / "manifests" / f"{manifest_id}.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if eval_inputs.sha256_json(old["samples"]) != eval_inputs.sha256_json(manifest["samples"]):
            sys.exit(f"{manifest_path} exists with different samples; refusing to overwrite")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    docs = [p for s in manifest["samples"] for p in s["provenance"]["documents"]]
    n_tok = sorted(p["n_tokens_standalone"] for p in docs)
    print(f"wrote {manifest_path}: {len(manifest['samples'])} samples, {len(docs)} docs, "
          f"truncated {sum(p['truncated'] for p in docs)}, tokens median {n_tok[len(n_tok)//2]}, "
          f"used_text==original {sum(p['used_text_equals_original'] for p in docs)}")


def cmd_compare():
    """Gold-paper recall of this index and the child125 index on the same 500 questions."""
    new = {s["sample_id"]: s for s in json.loads(RETRIEVED.read_text())["samples"]}
    old = {s["sample_id"]: s for s in json.loads(child125.RETRIEVED.read_text())["samples"]}
    if new.keys() != old.keys():
        sys.exit("sample ids differ between the child125 and m120 retrievals")
    out = {"n": len(new)}
    for name, key in (("paper_id", "from_gold_paper"), ("title", "gold_title_match")):
        n_hit = [any(d[key] for d in s["documents"]) for s in new.values()]
        out[f"m120_gold_in_top5_by_{name}"] = sum(n_hit)
    o = [any(d["from_gold_paper"] for d in s["documents"]) for s in old.values()]   # child125 = title match
    n = [any(d["gold_title_match"] for d in new[k]["documents"]) for k in old]
    out["child125_gold_in_top5_by_title"] = sum(o)
    out["both"], out["only_child125"], out["only_m120"] = (sum(a and b for a, b in zip(o, n)),
                                                           sum(a and not b for a, b in zip(o, n)), sum(b and not a for a, b in zip(o, n)))
    for tag, ss, key in (("m120", new.values(), "from_gold_paper"), ("child125", old.values(), "from_gold_paper")):
        out[f"{tag}_gold_chunks_per_q_mean"] = round(sum(sum(d[key] for d in s["documents"]) for s in ss) / len(new), 2)
        firsts = [next((i for i, d in enumerate(s["documents"]) if d[key]), None) for s in ss]
        out[f"{tag}_gold_at_rank1"] = sum(f == 0 for f in firsts)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "retrieve":
        cmd_retrieve()
    elif cmd == "manifest":
        cmd_manifest(ko="--ko" in sys.argv[2:])
    else:
        cmd_compare()
