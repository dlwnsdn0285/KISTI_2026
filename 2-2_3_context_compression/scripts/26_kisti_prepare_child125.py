"""Step 26 (KISTI): kisti-rag1 Korean science-paper RAG, top-5 retrieved 125-char chunks -> PISCO manifest.

Restored from this server's lost "Step 22 (KISTI)" (see STEP_LOG Step 26); behaviour unchanged.

Two stages, two environments:

  # 1) dense retrieval (needs sentence-transformers + GPU; conda env `refrag`)
  CUDA_VISIBLE_DEVICES=0 ~/anaconda3/envs/refrag/bin/python scripts/26_kisti_prepare_child125.py retrieve
  # 2) manifest (PISCO tokenizer; conda env `pisco-ref`)
  ~/anaconda3/envs/pisco-ref/bin/python scripts/26_kisti_prepare_child125.py manifest [--ko]

Queries     kisti-rag1 500-sample set (adaptive_retrieval_format_KONI_4B.json), all 500
            (the need_retrieval flag is ignored: this step measures compression loss).
References  answer_text of the same question in kisti-rag1 data/KISTI_sample QA files.
Collection  /data2/jwlee/past_embed, Chroma collection kisti-sentence-child-500-125:
            KISTI_sample corpus (20,252 papers), RecursiveCharacterTextSplitter
            chunk_size=125 characters, separators ["\\n\\n", ".", "\\n"], small chunks
            (<20 chars) removed. Embeddings: intfloat/multilingual-e5-large-instruct,
            normalized, document side without instruction.
Retrieval   Exact inner-product search over the stored vectors (read directly from the
            Chroma HNSW files; chromadb is not installed on this server and a different
            chromadb version could migrate the sqlite schema). Query embedded exactly as
            kisti-rag1 does (langchain HuggingFaceEmbeddings.embed_query: no instruction).
            Top-5 after removing duplicate texts (kisti-rag1 remove_duplicate_docs).
Gold        the source paper of each question = query_id prefix; a retrieved chunk is
            "gold" when its title equals that paper's title (child metadata doc_id was
            overwritten by ParentDocumentRetriever with the parent uuid).
"""

import json
import os
import pickle
import sqlite3
import struct
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KISTI_ROOT = Path("/home/jwlee/kisti-rag1")
QUERY_FILE = KISTI_ROOT / "adaptive_retrieval_format_KONI_4B.json"
QA_DIR = KISTI_ROOT / "data/KISTI_sample/(분류) 국내 논문 QA 데이터셋"
PAPER_DIR = KISTI_ROOT / "data/KISTI_sample/(분류) 국내 논문 전문 텍스트 데이터셋"
CHROMA_DIR = Path("/data2/jwlee/past_embed")
COLLECTION = "kisti-sentence-child-500-125"
E5_DIR = "/data2/jwlee/huggingface/hub/models--intfloat--multilingual-e5-large-instruct/snapshots/274baa43b0e13e37fafa6428dbc7938e62e5c439"
TOP_K = 5
OUT_DIR = PROJECT_ROOT / "data" / "kisti"
RETRIEVED = OUT_DIR / "sample500_e5child125_top5_retrieved.json"
MANIFEST_ID = "kisti_sample500_e5child125_top5"
KO_SUFFIX = " 한국어로 간단히 답하세요."   # variant `manifest --ko`: same docs, answer-language instruction appended
REFERENCE_MANIFEST = PROJECT_ROOT / "reference" / "manifest.json"


def choose_language(d):
    return d.get("ko") or d.get("en") or ""


class _PersistentDataStub:
    """Stands in for chromadb's PersistentData when unpickling index_metadata.pickle."""
    def __setstate__(self, state):
        self.__dict__.update(state)


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("chromadb") and name == "PersistentData":
            return _PersistentDataStub
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")


def segment_dirs(con):
    rows = con.execute("select s.id, s.scope from segments s join collections c on s.collection=c.id "
                       "where c.name=?", (COLLECTION,)).fetchall()
    return {scope: sid for sid, scope in rows}


def load_vectors(vector_dir):
    """All live vectors of the HNSW segment -> (np.float32 [N, dim], [embedding_id])."""
    import numpy as np
    header = (vector_dir / "header.bin").read_bytes()
    (version,) = struct.unpack_from("<i", header, 0)
    (off_l0, max_el, cur, size_el, label_off, data_off, maxlevel, enter, max_m, max_m0, m, mult, efc) = \
        struct.unpack_from("<QQQQQQiIQQQdQ", header, 4)
    with open(vector_dir / "index_metadata.pickle", "rb") as f:
        meta = _Unpickler(f).load()
    dim = meta.dimensionality
    if data_off + dim * 4 != label_off or label_off + 8 != size_el:
        raise ValueError(f"unexpected HNSW layout: {(data_off, label_off, size_el, dim)}")
    raw = np.memmap(vector_dir / "data_level0.bin", dtype=np.uint8, mode="r", shape=(cur, size_el))
    labels = raw[:, label_off:label_off + 8].copy().view("<u8").ravel()
    label_to_id = meta.label_to_id
    keep = np.array([int(l) in label_to_id for l in labels])
    vecs = np.ascontiguousarray(raw[keep, data_off:data_off + dim * 4]).view("<f4").reshape(-1, dim)
    ids = [label_to_id[int(l)] for l in labels[keep]]
    info = {"hnsw_header_version": version, "elements_in_file": int(cur), "live": int(keep.sum()),
            "id_to_label": len(meta.id_to_label), "dim": dim}
    return vecs, ids, info


def fetch_docs(con, metadata_segment, embedding_ids):
    out = {}
    q = ("select e.embedding_id, m.key, m.string_value from embeddings e join embedding_metadata m on m.id=e.id "
         "where e.segment_id=? and e.embedding_id=? and m.key in ('chroma:document','title','journal','year','section')")
    for eid in embedding_ids:
        row = {}
        for _, key, val in con.execute(q, (metadata_segment, eid)):
            row[key] = val
        out[eid] = row
    return out


def cmd_retrieve():
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    queries = json.loads(QUERY_FILE.read_text())
    qa = {}
    for p in sorted(QA_DIR.glob("*/*/*.json")):
        for item in json.loads(p.read_text()).get("qas", []):
            qa.setdefault(item["question"].strip(), []).append(
                {"answer": item["answer"]["answer_text"], "level": item.get("level"), "paper": p.stem})
    titles = {}
    for q in queries:
        paper_id = q["query_id"].rsplit("-", 1)[0]
        if paper_id not in titles:
            hits = list(PAPER_DIR.glob(f"*/*/{paper_id}.json"))
            titles[paper_id] = choose_language(json.loads(hits[0].read_text()).get("title", {})) if hits else None

    con = sqlite3.connect(f"file:{CHROMA_DIR / 'chroma.sqlite3'}?mode=ro", uri=True)
    segs = segment_dirs(con)
    vecs, ids, info = load_vectors(CHROMA_DIR / segs["VECTOR"])
    n_sql = con.execute("select count(*) from embeddings where segment_id=?", (segs["METADATA"],)).fetchone()[0]
    info["sqlite_rows"] = n_sql
    print("vectors", info, flush=True)
    if info["live"] != n_sql:
        print(f"WARNING: live HNSW vectors {info['live']} != sqlite rows {n_sql}", flush=True)

    model = SentenceTransformer(E5_DIR, device="cuda")
    # sanity: re-embedding stored documents must reproduce the stored vectors
    probe_ids = ids[:: max(1, len(ids) // 20)][:20]
    probe_docs = fetch_docs(con, segs["METADATA"], probe_ids)
    probe_vec = model.encode([probe_docs[i]["chroma:document"] for i in probe_ids], normalize_embeddings=False)
    id_pos = {eid: k for k, eid in enumerate(ids)}
    cos = [float(np.dot(probe_vec[k], vecs[id_pos[eid]]) / np.linalg.norm(probe_vec[k]) / np.linalg.norm(vecs[id_pos[eid]]))
           for k, eid in enumerate(probe_ids)]
    info["reembed_cosine_min"] = min(cos)
    info["stored_norm_mean"] = float(np.linalg.norm(vecs[:1000], axis=1).mean())
    print("re-embed cosine min", min(cos), "stored norm", info["stored_norm_mean"], flush=True)
    if min(cos) < 0.99:
        sys.exit("stored vectors do not match the e5 model; aborting")

    mat = torch.from_numpy(vecs).to("cuda", dtype=torch.float16)
    qtexts = [q["query"] for q in queries]
    qvec = torch.from_numpy(model.encode(qtexts, normalize_embeddings=False, batch_size=32)).to("cuda", dtype=torch.float16)
    # langchain_chroma default space is l2 on normalized vectors -> same order as inner product
    scores = qvec @ mat.T
    top_scores, top_idx = scores.topk(64, dim=1)
    top_scores, top_idx = top_scores.float().cpu().numpy(), top_idx.cpu().numpy()
    del mat, scores
    torch.cuda.empty_cache()

    needed = {ids[j] for row in top_idx for j in row}
    docs = fetch_docs(con, segs["METADATA"], sorted(needed))
    samples = []
    for qi, q in enumerate(queries):
        refs = qa.get(q["query"].strip())
        if not refs:
            sys.exit(f"no reference answer for {q['query_id']}")
        paper_id = q["query_id"].rsplit("-", 1)[0]
        picked, seen = [], set()
        for j, sc in zip(top_idx[qi], top_scores[qi]):
            d = docs[ids[j]]
            text = d["chroma:document"]
            if text in seen:
                continue
            seen.add(text)
            picked.append({"doc_id": ids[j], "text": text, "score": float(sc), "title": d.get("title"),
                           "journal": d.get("journal"), "year": d.get("year"), "section": d.get("section"),
                           "from_gold_paper": titles[paper_id] is not None and d.get("title") == titles[paper_id]})
            if len(picked) == TOP_K:
                break
        if len(picked) < TOP_K:
            sys.exit(f"{q['query_id']}: fewer than {TOP_K} unique chunks in top-64")
        samples.append({"sample_id": q["query_id"], "question": q["query"],
                        "reference_answers": [r["answer"] for r in refs][:1],
                        "qa_level": refs[0]["level"], "gold_paper": paper_id, "gold_title": titles[paper_id],
                        "need_retrieval_koni": q["need_retrieval"], "documents": picked})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {"retrieval": {"collection": f"{CHROMA_DIR}:{COLLECTION}", "query_encoding": "e5 no-instruction (kisti-rag1 embed_query)",
                         "search": "exact inner product, fp16, GPU", "dedup": "identical text", "top_k": TOP_K, **info},
           "samples": samples}
    RETRIEVED.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    gold_any = sum(any(d["from_gold_paper"] for d in s["documents"]) for s in samples)
    print(f"wrote {RETRIEVED}: {len(samples)} samples, gold paper in top-5: {gold_any}")


def cmd_manifest(ko=False):
    os.environ["HF_HUB_OFFLINE"] = "1"
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from pisco_research import eval_inputs
    retrieved = json.loads(RETRIEVED.read_text())
    ref = json.loads(REFERENCE_MANIFEST.read_text())
    tok = eval_inputs.load_pisco_tokenizer(ref["snapshots"]["backbone"]["local_dir"])
    tok_source = {"repo_id": ref["backbone"]["repo_id"], "revision": ref["backbone"]["revision_sha"],
                  "added_special_tokens": eval_inputs.PISCO_ADDED_TOKENS}
    samples = []
    for s in retrieved["samples"]:
        samples.append({
            "sample_id": s["sample_id"], "question": s["question"] + (KO_SUFFIX if ko else ""), "reference_answers": s["reference_answers"],
            "gold_doc_ids": [], "documents": [{"doc_id": d["doc_id"], "text": d["text"]} for d in s["documents"]],
            "annotation": {"gold_not_truncated": None, "qa_level": s["qa_level"], "gold_paper": s["gold_paper"],
                           "gold_paper_in_top5": any(d["from_gold_paper"] for d in s["documents"]),
                           "gold_paper_positions": [i for i, d in enumerate(s["documents"]) if d["from_gold_paper"]],
                           "need_retrieval_koni": s["need_retrieval_koni"]},
        })
    fixture = {
        "fixture_id": "kisti_sample500",
        "source": {"dataset": "KISTI 국내 논문 QA (kisti-rag1 500-sample)", "query_file": str(QUERY_FILE),
                   "retrieved_file": str(RETRIEVED.relative_to(PROJECT_ROOT)), "retrieval": retrieved["retrieval"]},
        "preprocessing": {"doc_text": "Chroma child chunk text as stored (125-char sentence-aware chunks)"},
        "context_policy": {"protocol": "kisti_retrieved", "context_mode": "retrieved_only",
                           "retrieval": "e5-large-instruct dense, exact search, top-5 unique",
                           "gold_policy": "none (no gold insertion)", "order": "retrieval score, most relevant first"},
        "samples": samples,
    }
    manifest_id = MANIFEST_ID + ("_ko" if ko else "")
    if ko:
        fixture["preprocessing"]["question"] = f"original question + {KO_SUFFIX!r}"
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


if __name__ == "__main__":
    if sys.argv[1] == "retrieve":
        cmd_retrieve()
    else:
        cmd_manifest(ko="--ko" in sys.argv[2:])
