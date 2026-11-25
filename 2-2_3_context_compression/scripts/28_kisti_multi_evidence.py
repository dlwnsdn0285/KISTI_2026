"""Step 28: synthetic two-evidence KISTI QA from the single-evidence KISTI_sample QA, and its manifests.

  # 1) QA pairs (CPU; conda env `refrag`): data/kisti/me_pairs.json
  ~/anaconda3/envs/refrag/bin/python scripts/28_kisti_multi_evidence.py pairs
  # 2) merge each pair's two questions into one question + verify (Qwen2.5-32B-Instruct over GPUs 0-3; `refrag`)
  CUDA_VISIBLE_DEVICES=0,1,2,3 ~/anaconda3/envs/refrag/bin/python scripts/28_kisti_multi_evidence.py compose
  # 3) e5 retrieval for the merged questions (one GPU; `refrag`)
  CUDA_VISIBLE_DEVICES=0 ~/anaconda3/envs/refrag/bin/python scripts/28_kisti_multi_evidence.py retrieve
  # 4) manifests (`pisco-ref`)
  ~/anaconda3/envs/pisco-ref/bin/python scripts/28_kisti_multi_evidence.py manifest

Source QA   kisti-rag1 data/KISTI_sample QA (4,680 questions, 1,583 papers). Answers are extractive spans.
Evidence    a QA is usable when its answer (whitespace-insensitive) occurs in exactly one chunk of its own paper in
            the Step 26 v2 index (<= 120 Mistral tokens per chunk): that chunk is the QA's gold chunk.
Type A      two QAs of the SAME paper whose gold chunks differ (one pair per paper).
Type B      two QAs of DIFFERENT papers on a similar topic: greedy matching by cosine of the gold-chunk e5 vectors
            (each QA and each paper pair used once).
Question    an LLM only rewrites the two questions into one question that asks both; answers stay the two
            original extractive spans (no generated answers). A second LLM pass keeps a pair only if the merged
            question still asks both sub-questions with their terms and adds nothing.
Settings    gold2_dist3: the two gold chunks + 3 distractors (top e5 chunks for the merged question, excluding
            the gold papers and any chunk containing either answer), order shuffled per sample (seed 0).
            ret_top5: plain top-5 e5 retrieval for the merged question (no gold insertion).
            Both with the answer-in-Korean suffix (Step 26 ko). Each also gets a `_full` twin whose last document
            = the five chunk texts joined by newlines (input of the Step 19 rep_full extra-block generator).
"""

import importlib.util
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QA_DIR = Path("/home/jwlee/kisti-rag1/data/KISTI_sample/(분류) 국내 논문 QA 데이터셋")
INDEX_DIR = Path("/data3/jwlee/kisti_sample_e5_m120_v2")
E5_DIR = "/data2/jwlee/huggingface/hub/models--intfloat--multilingual-e5-large-instruct/snapshots/274baa43b0e13e37fafa6428dbc7938e62e5c439"
LLM_DIR = "/data2/jwlee/huggingface/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"
OUT = PROJECT_ROOT / "data" / "kisti"
PAIRS, COMPOSED, RETRIEVED = OUT / "me_pairs.json", OUT / "me_composed.json", OUT / "me_retrieved.json"
KO_SUFFIX = " 한국어로 간단히 답하세요."
N_CANDIDATES, N_FINAL, SEED = 650, 500, 0     # pairs composed per type, pairs kept per type after verification
SEARCH_K, N_DIST = 64, 3
SEP = "\n"

COMPOSE_PROMPT = """아래 두 질문을 하나의 자연스러운 한국어 질문으로 합치세요.
규칙:
1. 합친 질문 하나에 답하려면 두 질문의 답이 모두 필요해야 합니다. 두 질문이 묻는 내용을 빠짐없이 물으세요.
2. 원래 질문에 있는 용어, 고유명사, 조건, 인용(예: 저자·연도)을 바꾸거나 빼지 마세요.
3. 원래 질문에 없는 정보, 추측, 답을 추가하지 마세요.
4. 한두 문장의 질문만 출력하세요. 설명이나 따옴표는 쓰지 마세요.

질문 1: {q1}
질문 2: {q2}

합친 질문:"""

VERIFY_PROMPT = """원래 질문 두 개와 이를 합친 질문이 있습니다.
합친 질문이 (1) 질문 1과 질문 2가 묻는 내용을 모두 묻고, (2) 원래 질문의 용어·조건을 빠뜨리거나 바꾸지 않았고, (3) 원래 질문에 없는 정보를 추가하지 않았으면 "예", 하나라도 어기면 "아니오"라고만 답하세요.

질문 1: {q1}
질문 2: {q2}
합친 질문: {merged}

판정:"""


def norm(s):
    return "".join(s.split())


def load_index_rows(columns=("chunk_id", "paper_id", "text")):
    import pyarrow.parquet as pq
    rows = []
    for f in sorted((INDEX_DIR / "blocks").glob("*.parquet")):
        rows.extend(pq.read_table(f, columns=list(columns)).to_pylist())
    return rows


def build_pairs():
    """-> (usable QAs, type A pairs, type B (a, b, cosine) triples), all candidates, seed-SEED order."""
    import numpy as np
    qas = []
    for p in sorted(QA_DIR.glob("*/*/*.json")):
        d = json.loads(p.read_text())
        for q in d["qas"]:
            qas.append({"qa_id": q["id"], "paper_id": d["doc_id"], "title": d["title"], "question": q["question"].strip(),
                        "answer": q["answer"]["answer_text"].strip(), "level": q["level"]})
    papers = {q["paper_id"] for q in qas}
    rows = load_index_rows()
    by_paper = defaultdict(list)
    for i, r in enumerate(rows):
        if r["paper_id"] in papers:
            by_paper[r["paper_id"]].append(i)
    usable = []
    for q in qas:
        hits = [i for i in by_paper[q["paper_id"]] if norm(q["answer"]) in norm(rows[i]["text"])]
        if len(hits) == 1 and len(norm(q["answer"])) >= 2:
            usable.append({**q, "gold_row": hits[0], "gold_chunk": rows[hits[0]]["chunk_id"]})
    print(f"QA {len(qas)}, usable (answer in exactly one chunk) {len(usable)}")
    rng = random.Random(SEED)

    # Type A: same paper, different gold chunks, one pair per paper
    per = defaultdict(list)
    for q in usable:
        per[q["paper_id"]].append(q)
    type_a = []
    for pid in sorted(per):
        cands = [(a, b) for i, a in enumerate(per[pid]) for b in per[pid][i + 1:] if a["gold_chunk"] != b["gold_chunk"]]
        if cands:
            a, b = rng.choice(cands)
            type_a.append([a, b] if rng.random() < 0.5 else [b, a])
    rng.shuffle(type_a)

    # Type B: different papers, most similar gold chunks first (greedy, each QA and paper pair once)
    vecs = np.concatenate([np.load(f) for f in sorted((INDEX_DIR / "blocks").glob("*.npy"))])
    g = np.stack([vecs[q["gold_row"]].astype(np.float32) for q in usable])
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    sim = g @ g.T
    pid = np.array([q["paper_id"] for q in usable])
    sim[pid[:, None] == pid[None, :]] = -1
    sim[sim > 0.97] = -1                              # near-identical chunks (duplicated papers / boilerplate)
    order = np.dstack(np.unravel_index(np.argsort(-np.triu(sim, 1), axis=None), sim.shape))[0]
    used_q, used_pp, type_b = set(), set(), []
    for i, j in order:
        if sim[i, j] <= 0:
            break
        pp = tuple(sorted((pid[i], pid[j])))
        if i in used_q or j in used_q or pp in used_pp:
            continue
        used_q.update((i, j))
        used_pp.add(pp)
        first, second = (usable[i], usable[j]) if rng.random() < 0.5 else (usable[j], usable[i])
        type_b.append((first, second, float(sim[i, j])))
    print(f"type A candidates {len(type_a)}, type B candidates {len(type_b)} "
          f"(cosine median {sorted(t[2] for t in type_b)[len(type_b) // 2]:.3f})")
    rng.shuffle(type_b)
    return usable, type_a, type_b


def cmd_pairs():
    usable, type_a, type_b = build_pairs()
    strip = lambda parts: [{k: v for k, v in q.items() if k != "gold_row"} for q in parts]   # a QA may sit in A and B
    out = {"A": [{"pair_id": f"A{k:04d}", "parts": strip(p)} for k, p in enumerate(type_a[:N_CANDIDATES])],
           "B": [{"pair_id": f"B{k:04d}", "parts": strip(p[:2]), "gold_cosine": round(p[2], 4)}
                 for k, p in enumerate(type_b[:N_CANDIDATES])]}
    OUT.mkdir(parents=True, exist_ok=True)
    PAIRS.write_text(json.dumps({"seed": SEED, "n_usable_qa": len(usable), "n_candidates": {"A": len(type_a), "B": len(type_b)},
                                 **out}, ensure_ascii=False, indent=1))
    print(f"wrote {PAIRS}: A {len(out['A'])}, B {len(out['B'])}")


def chat_generate(model, tok, prompts, max_new_tokens, batch_size=16):
    """Greedy chat completions; prompts are batched in length order (less padding), results in input order."""
    import torch
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    outs = [None] * len(prompts)
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]
        texts = [tok.apply_chat_template([{"role": "user", "content": prompts[k]}], tokenize=False, add_generation_prompt=True)
                 for k in idx]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        for k, t in zip(idx, tok.batch_decode(gen[:, enc["input_ids"].size(1):], skip_special_tokens=True)):
            outs[k] = t.strip()
        print(f"  {min(i + batch_size, len(prompts))}/{len(prompts)}", flush=True)
    return outs


def cmd_compose():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    pairs = json.loads(PAIRS.read_text())
    tok = AutoTokenizer.from_pretrained(LLM_DIR, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(LLM_DIR, torch_dtype=torch.bfloat16, device_map="auto").eval()
    out = {"llm": LLM_DIR, "compose_prompt": COMPOSE_PROMPT, "verify_prompt": VERIFY_PROMPT}
    for t in ("A", "B"):
        items = pairs[t]
        merged = chat_generate(model, tok, [COMPOSE_PROMPT.format(q1=i["parts"][0]["question"], q2=i["parts"][1]["question"])
                                            for i in items], 160)
        merged = [m.splitlines()[0].strip().strip('"“”') if m else "" for m in merged]
        verdict = chat_generate(model, tok, [VERIFY_PROMPT.format(q1=i["parts"][0]["question"], q2=i["parts"][1]["question"],
                                                                  merged=m) for i, m in zip(items, merged)], 4)
        for i, m, v in zip(items, merged, verdict):
            i["question"] = m
            i["verify"] = v
            i["kept"] = v.startswith("예") and bool(m) and m.endswith("?")
        kept = [i for i in items if i["kept"]]
        out[t] = items
        out[f"{t}_final_ids"] = [i["pair_id"] for i in kept[:N_FINAL]]
        print(f"type {t}: composed {len(items)}, verified {len(kept)}, final {len(out[f'{t}_final_ids'])}", flush=True)
    COMPOSED.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("wrote", COMPOSED)


def cmd_retrieve():
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    comp = json.loads(COMPOSED.read_text())
    rows = load_index_rows()
    vecs = np.concatenate([np.load(f) for f in sorted((INDEX_DIR / "blocks").glob("*.npy"))])
    mat = torch.from_numpy(vecs).to("cuda")
    del vecs
    model = SentenceTransformer(E5_DIR, device="cuda").half()
    out = {}
    for t in ("A", "B"):
        items = {i["pair_id"]: i for i in comp[t]}
        final = [items[k] for k in comp[f"{t}_final_ids"]]
        qv = model.encode([i["question"] for i in final], normalize_embeddings=True, convert_to_tensor=True, batch_size=32)
        _, idx = (qv.to(mat.dtype) @ mat.T).topk(SEARCH_K, dim=1)
        out[t] = []
        for i, top in zip(final, idx.cpu().tolist()):
            ranked, seen = [], set()
            for j in top:
                if rows[j]["text"] not in seen:
                    seen.add(rows[j]["text"])
                    ranked.append(rows[j])
            out[t].append({"pair_id": i["pair_id"], "ranked": ranked})
    RETRIEVED.write_text(json.dumps(out, ensure_ascii=False))
    print("wrote", RETRIEVED)


def cmd_manifest():
    os.environ["HF_HUB_OFFLINE"] = "1"
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from pisco_research import eval_inputs
    ref = json.loads((PROJECT_ROOT / "reference" / "manifest.json").read_text())
    tok = eval_inputs.load_pisco_tokenizer(ref["snapshots"]["backbone"]["local_dir"])
    tok_source = {"repo_id": ref["backbone"]["repo_id"], "revision": ref["backbone"]["revision_sha"],
                  "added_special_tokens": eval_inputs.PISCO_ADDED_TOKENS}
    comp = json.loads(COMPOSED.read_text())
    ret = json.loads(RETRIEVED.read_text())
    chunk_text = {r["chunk_id"]: r["text"] for r in load_index_rows(("chunk_id", "text"))}

    def write(manifest_id, samples, policy):
        fixture = {"fixture_id": manifest_id,
                   "source": {"dataset": "synthetic two-evidence KISTI QA (Step 28)", "pairs": str(PAIRS.relative_to(PROJECT_ROOT)),
                              "composed": str(COMPOSED.relative_to(PROJECT_ROOT)), "llm": comp["llm"]},
                   "preprocessing": {"question": f"merged question + {KO_SUFFIX!r}",
                                     "doc_text": "Step 26 v2 chunk text (<=120 Mistral tokens)"},
                   "context_policy": policy, "samples": samples}
        fp = OUT / f"{manifest_id}_fixture.json"
        fp.write_text(json.dumps(fixture, ensure_ascii=False))
        manifest = eval_inputs.build_manifest(fp, tok, tok_source, manifest_id)
        path = PROJECT_ROOT / "data" / "manifests" / f"{manifest_id}.json"
        if path.exists():
            old = json.loads(path.read_text())
            if eval_inputs.sha256_json(old["samples"]) != eval_inputs.sha256_json(manifest["samples"]):
                sys.exit(f"{path} exists with different samples; refusing to overwrite")
        path.write_text(json.dumps(manifest, ensure_ascii=False))
        docs = [p for s in manifest["samples"] for p in s["provenance"]["documents"]]
        print(f"wrote {path.name}: {len(samples)} samples, truncated docs {sum(p['truncated'] for p in docs)}")

    def with_full(samples):
        return [{**s, "documents": s["documents"] + [{"doc_id": f"{s['sample_id']}#full",
                                                     "text": SEP.join(d["text"] for d in s["documents"])}],
                 "annotation": {**s["annotation"], "full_doc_order": [d["doc_id"] for d in s["documents"]]}} for s in samples]

    for t in ("A", "B"):
        items = {i["pair_id"]: i for i in comp[t]}
        rets = {r["pair_id"]: r["ranked"] for r in ret[t]}
        gold_s, ret_s, stats = [], [], {"both_gold_chunks_top5": 0, "one_gold_chunk_top5": 0, "both_gold_papers_top5": 0}
        for pid in comp[f"{t}_final_ids"]:
            it = items[pid]
            parts = it["parts"]
            golds = [p["gold_chunk"] for p in parts]
            gold_papers = {p["paper_id"] for p in parts}
            answers = [norm(p["answer"]) for p in parts]
            base = {"sample_id": pid, "question": it["question"] + KO_SUFFIX,
                    "reference_answers": [p["answer"] for p in parts], "gold_doc_ids": golds}
            ann = {"type": t, "sub_questions": [p["question"] for p in parts], "sub_answers": [p["answer"] for p in parts],
                   "qa_ids": [p["qa_id"] for p in parts], "levels": [p["level"] for p in parts],
                   "gold_papers": [p["paper_id"] for p in parts], "gold_cosine": it.get("gold_cosine")}
            dist = [r for r in rets[pid] if r["paper_id"] not in gold_papers
                    and not any(a in norm(r["text"]) for a in answers)][:N_DIST]
            if len(dist) < N_DIST:
                sys.exit(f"{pid}: fewer than {N_DIST} distractors")
            docs = [{"doc_id": g, "text": chunk_text[g]} for g in golds] + [{"doc_id": r["chunk_id"], "text": r["text"]} for r in dist]
            random.Random(f"{SEED}-{pid}").shuffle(docs)
            gold_s.append({**base, "documents": docs,
                           "annotation": {**ann, "gold_positions": [i for i, d in enumerate(docs) if d["doc_id"] in golds]}})
            top5 = rets[pid][:5]
            hit = [g in {r["chunk_id"] for r in top5} for g in golds]
            stats["both_gold_chunks_top5"] += all(hit)
            stats["one_gold_chunk_top5"] += sum(hit) == 1
            stats["both_gold_papers_top5"] += gold_papers <= {r["paper_id"] for r in top5}
            ret_s.append({**base, "gold_doc_ids": [g for g, h in zip(golds, hit) if h],   # only gold chunks retrieved
                          "documents": [{"doc_id": r["chunk_id"], "text": r["text"]} for r in top5],
                          "annotation": {**ann, "gold_chunks_in_top5": hit,
                                         "gold_papers_in_top5": [p in {r["paper_id"] for r in top5} for p in ann["gold_papers"]]}})
        print(t, "retrieval top-5:", stats, "of", len(ret_s))
        gpol = {"protocol": "kisti_me_gold2_dist3", "gold_policy": "both gold chunks inserted",
                "distractors": "top e5 chunks for the merged question, gold papers and answer-containing chunks excluded",
                "order": "shuffled per sample (seed 0)"}
        rpol = {"protocol": "kisti_me_retrieved", "gold_policy": "none", "retrieval": "e5 top-5 unique", "order": "retrieval score"}
        write(f"kisti_me{t}_gold2_dist3_ko", gold_s, gpol)
        write(f"kisti_me{t}_ret_top5_ko", ret_s, rpol)
        write(f"kisti_me{t}_gold2_dist3_ko_full", with_full(gold_s), {**gpol, "k_final": 6, "note": "last doc = five texts joined"})
        write(f"kisti_me{t}_ret_top5_ko_full", with_full(ret_s), {**rpol, "k_final": 6, "note": "last doc = five texts joined"})


def cmd_single_full():
    """`_full` twin of the Step 26 single-evidence manifest (last document = the five chunk texts joined)."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from pisco_research import eval_inputs
    ref = json.loads((PROJECT_ROOT / "reference" / "manifest.json").read_text())
    tok = eval_inputs.load_pisco_tokenizer(ref["snapshots"]["backbone"]["local_dir"])
    tok_source = {"repo_id": ref["backbone"]["repo_id"], "revision": ref["backbone"]["revision_sha"],
                  "added_special_tokens": eval_inputs.PISCO_ADDED_TOKENS}
    mid = "kisti_sample500_e5m120v2_top5_ko"
    src = json.loads((PROJECT_ROOT / "data" / "manifests" / f"{mid}.json").read_text())
    samples = []
    for s in src["samples"]:
        docs = s["model_input"]["documents"]
        samples.append({"sample_id": s["sample_id"], "question": s["model_input"]["question"],
                        "reference_answers": s["evaluation"]["reference_answers"], "gold_doc_ids": s["evaluation"]["gold_doc_ids"],
                        "documents": [{"doc_id": d["doc_id"], "text": d["text"]} for d in docs] +
                                     [{"doc_id": f"{s['sample_id']}#full", "text": SEP.join(d["text"] for d in docs)}],
                        "annotation": {**s["annotation"], "full_doc_order": [d["doc_id"] for d in docs]}})
    fixture = {"fixture_id": f"{mid}_full", "source": {**src["source"], "derived_from": mid},
               "preprocessing": {**src["preprocessing"], "sixth_document": "five chunk texts joined by newline"},
               "context_policy": {**src["context_policy"], "k_final": 6}, "samples": samples}
    fp = OUT / f"{mid}_full_fixture.json"
    fp.write_text(json.dumps(fixture, ensure_ascii=False))
    manifest = eval_inputs.build_manifest(fp, tok, tok_source, f"{mid}_full")
    path = PROJECT_ROOT / "data" / "manifests" / f"{mid}_full.json"
    if path.exists() and eval_inputs.sha256_json(json.loads(path.read_text())["samples"]) != eval_inputs.sha256_json(manifest["samples"]):
        sys.exit(f"{path} exists with different samples; refusing to overwrite")
    path.write_text(json.dumps(manifest, ensure_ascii=False))
    for s, o in zip(manifest["samples"], src["samples"]):
        if s["model_input"]["documents"][:-1] != o["model_input"]["documents"]:
            raise RuntimeError(f"{s['sample_id']}: documents 1-5 changed")
    print("wrote", path.name)


# ============================================================ compositional set (one answer from two evidences)

CGEN, CRETRIEVED = OUT / "me_comp_generated.json", OUT / "me_comp_retrieved.json"
QTYPES = ["연결형", "비교형", "결합형"]

CGEN_PROMPT = """다음은 과학기술 논문에서 가져온 두 문서 조각입니다.

[문서 1]
{c1}

[문서 2]
{c2}

두 문서의 정보를 모두 사용해야만 답할 수 있는 {qtype} 질문 하나와 그 정답을 만드세요.
- 연결형: 문서 1의 내용으로 어떤 대상(개념·방법·물질·현상 등)을 가리키고, 그 대상에 대해 문서 2에만 있는 정보를 묻습니다. 질문에 그 대상의 이름을 직접 쓰지 말고 문서 1의 설명으로 지칭하세요.
- 비교형: 두 문서에 공통으로 나오는 성질(수치, 시기, 방법, 결과의 방향 등)을 비교하는 질문입니다. 정답은 둘 중 하나이거나 '예'/'아니오'입니다.
- 결합형: 문서 1의 조건과 문서 2의 조건을 모두 만족하는 것, 또는 두 문서의 정보를 합쳐야 나오는 값·결론을 묻습니다.
규칙:
1. 문서 하나만 읽어서는 답할 수 없어야 합니다.
2. 질문은 문서를 보지 않은 사람도 이해할 수 있게 자기완결적으로 쓰세요. '문서 1', '위 글', '이 연구' 같은 표현을 쓰지 마세요.
3. 정답은 문서 내용에 근거한 짧은 단어·구·수치 또는 '예'/'아니오'로, 20자 이내로 쓰세요. 정답을 질문에 드러내지 마세요.
4. 두 문서로 이런 질문을 만들 수 없으면 question을 null로 하세요.
JSON 한 줄로만 출력하세요: {{"question": "...", "answer": "...", "reasoning": "문서 1에서 ..., 문서 2에서 ..."}}"""

CANSWER_PROMPT = """다음 문서만 근거로 질문에 답하세요. 답은 짧은 단어·구·수치 또는 '예'/'아니오'로만 쓰고, 문서로 알 수 없으면 '알 수 없음'이라고만 쓰세요.

{docs}

질문: {question}
답:"""

CJUDGE_PROMPT = """질문, 정답, 예측이 있습니다. 예측이 정답과 같은 내용이면 "예", 다르거나 '알 수 없음'이면 "아니오"라고만 답하세요.

질문: {question}
정답: {answer}
예측: {pred}

판정:"""


def parse_json_line(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def cmd_cgen():
    """Generate one compositional QA per chunk pair (question types cycled), then keep only QAs the LLM answers
    correctly from both chunks and fails from either chunk alone."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    usable, type_a, type_b = build_pairs()
    rows = {r["chunk_id"]: r for r in load_index_rows(("chunk_id", "paper_id", "text"))}
    cands = []
    for t, pairs in (("A", type_a), ("B", [p[:2] for p in type_b])):
        for k, (a, b) in enumerate(pairs):
            cands.append({"pair_id": f"C{t}{k:04d}", "type": t, "qtype": QTYPES[k % 3],
                          "chunks": [a["gold_chunk"], b["gold_chunk"]], "papers": [a["paper_id"], b["paper_id"]],
                          "texts": [rows[a["gold_chunk"]]["text"], rows[b["gold_chunk"]]["text"]]})
    tok = AutoTokenizer.from_pretrained(LLM_DIR, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(LLM_DIR, torch_dtype=torch.bfloat16, device_map="auto").eval()
    bs = int(os.environ.get("CGEN_BATCH", "32"))
    gen = chat_generate(model, tok, [CGEN_PROMPT.format(c1=c["texts"][0], c2=c["texts"][1], qtype=c["qtype"]) for c in cands], 200, bs)
    ok = []
    for c, g in zip(cands, gen):
        j = parse_json_line(g) or {}
        q, ans = (j.get("question") or "").strip(), str(j.get("answer") or "").strip()
        c.update({"raw": g, "question": q, "answer": ans, "reasoning": j.get("reasoning")})
        c["stage"] = ("no_question" if not q else "answer_len" if not ans or len(ans) > 30 else
                      "no_qmark" if not q.endswith("?") else
                      "answer_in_question" if len(norm(ans)) >= 2 and norm(ans) in norm(q) else "generated")
        if c["stage"] == "generated":
            ok.append(c)
    print(f"generated {len(cands)}, well-formed {len(ok)}", flush=True)

    def answer_and_judge(items, docs_fn, tag):
        preds = chat_generate(model, tok, [CANSWER_PROMPT.format(docs=docs_fn(c), question=c["question"]) for c in items], 40, bs)
        verdict = chat_generate(model, tok, [CJUDGE_PROMPT.format(question=c["question"], answer=c["answer"], pred=p)
                                             for c, p in zip(items, preds)], 4, bs)
        for c, p, v in zip(items, preds, verdict):
            c[f"pred_{tag}"], c[f"correct_{tag}"] = p, v.startswith("예")

    answer_and_judge(ok, lambda c: f"[문서 1]\n{c['texts'][0]}\n\n[문서 2]\n{c['texts'][1]}", "both")
    both = [c for c in ok if c["correct_both"]]
    for c in ok:
        if not c["correct_both"]:
            c["stage"] = "both_wrong"
    answer_and_judge(both, lambda c: f"[문서]\n{c['texts'][0]}", "only1")
    answer_and_judge(both, lambda c: f"[문서]\n{c['texts'][1]}", "only2")
    for c in both:
        c["stage"] = "single_answerable" if (c["correct_only1"] or c["correct_only2"]) else "kept"
    out = {"llm": LLM_DIR, "gen_prompt": CGEN_PROMPT, "answer_prompt": CANSWER_PROMPT, "judge_prompt": CJUDGE_PROMPT,
           "candidates": [{k: v for k, v in c.items() if k != "texts"} for c in cands]}
    for t in ("A", "B"):
        kept = [c for c in cands if c["type"] == t and c["stage"] == "kept"]
        by_q = {q: [c for c in kept if c["qtype"] == q] for q in QTYPES}
        final = []
        while len(final) < N_FINAL and any(by_q.values()):          # round-robin over question types
            for q in QTYPES:
                if by_q[q] and len(final) < N_FINAL:
                    final.append(by_q[q].pop(0)["pair_id"])
        out[f"{t}_final_ids"] = final
        stages = defaultdict(int)
        for c in cands:
            if c["type"] == t:
                stages[c["stage"]] += 1
        print(f"type {t}: {dict(stages)}; kept by qtype "
              f"{ {q: sum(1 for c in kept if c['qtype'] == q) for q in QTYPES} }; final {len(final)}", flush=True)
    CGEN.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("wrote", CGEN)


def e5_rank(questions):
    """-> list of ranked unique chunk rows (top SEARCH_K) per question."""
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    rows = load_index_rows()
    vecs = np.concatenate([np.load(f) for f in sorted((INDEX_DIR / "blocks").glob("*.npy"))])
    mat = torch.from_numpy(vecs).to("cuda")
    del vecs
    model = SentenceTransformer(E5_DIR, device="cuda").half()
    qv = model.encode(questions, normalize_embeddings=True, convert_to_tensor=True, batch_size=32)
    _, idx = (qv.to(mat.dtype) @ mat.T).topk(SEARCH_K, dim=1)
    out = []
    for top in idx.cpu().tolist():
        ranked, seen = [], set()
        for j in top:
            if rows[j]["text"] not in seen:
                seen.add(rows[j]["text"])
                ranked.append(rows[j])
        out.append(ranked)
    return out


def cmd_cretrieve():
    gen = json.loads(CGEN.read_text())
    cands = {c["pair_id"]: c for c in gen["candidates"]}
    out = {}
    for t in ("A", "B"):
        final = [cands[k] for k in gen[f"{t}_final_ids"]]
        out[t] = [{"pair_id": c["pair_id"], "ranked": r} for c, r in zip(final, e5_rank([c["question"] for c in final]))]
    CRETRIEVED.write_text(json.dumps(out, ensure_ascii=False))
    print("wrote", CRETRIEVED)


def cmd_cmanifest():
    os.environ["HF_HUB_OFFLINE"] = "1"
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from pisco_research import eval_inputs
    ref = json.loads((PROJECT_ROOT / "reference" / "manifest.json").read_text())
    tok = eval_inputs.load_pisco_tokenizer(ref["snapshots"]["backbone"]["local_dir"])
    tok_source = {"repo_id": ref["backbone"]["repo_id"], "revision": ref["backbone"]["revision_sha"],
                  "added_special_tokens": eval_inputs.PISCO_ADDED_TOKENS}
    gen = json.loads(CGEN.read_text())
    ret = json.loads(CRETRIEVED.read_text())
    cands = {c["pair_id"]: c for c in gen["candidates"]}
    chunk_text = {r["chunk_id"]: r["text"] for r in load_index_rows(("chunk_id", "text"))}

    def write(manifest_id, samples, policy):
        fixture = {"fixture_id": manifest_id,
                   "source": {"dataset": "synthetic compositional two-evidence KISTI QA (Step 28)",
                              "generated": str(CGEN.relative_to(PROJECT_ROOT)), "llm": gen["llm"]},
                   "preprocessing": {"question": f"generated question + {KO_SUFFIX!r}",
                                     "doc_text": "Step 26 v2 chunk text (<=120 Mistral tokens)"},
                   "context_policy": policy, "samples": samples}
        fp = OUT / f"{manifest_id}_fixture.json"
        fp.write_text(json.dumps(fixture, ensure_ascii=False))
        manifest = eval_inputs.build_manifest(fp, tok, tok_source, manifest_id)
        path = PROJECT_ROOT / "data" / "manifests" / f"{manifest_id}.json"
        if path.exists() and eval_inputs.sha256_json(json.loads(path.read_text())["samples"]) != eval_inputs.sha256_json(manifest["samples"]):
            sys.exit(f"{path} exists with different samples; refusing to overwrite")
        path.write_text(json.dumps(manifest, ensure_ascii=False))
        print(f"wrote {path.name}: {len(samples)} samples")

    def with_full(samples):
        return [{**s, "documents": s["documents"] + [{"doc_id": f"{s['sample_id']}#full",
                                                     "text": SEP.join(d["text"] for d in s["documents"])}],
                 "annotation": {**s["annotation"], "full_doc_order": [d["doc_id"] for d in s["documents"]]}} for s in samples]

    for t in ("A", "B"):
        rets = {r["pair_id"]: r["ranked"] for r in ret[t]}
        gold_s, ret_s = [], []
        stats = {"both_gold_chunks_top5": 0, "one_gold_chunk_top5": 0, "both_gold_papers_top5": 0}
        for pid in gen[f"{t}_final_ids"]:
            c = cands[pid]
            golds, gold_papers, ans = c["chunks"], set(c["papers"]), norm(c["answer"])
            base = {"sample_id": pid, "question": c["question"] + KO_SUFFIX, "reference_answers": [c["answer"]]}
            ann = {"type": t, "qtype": c["qtype"], "gold_chunks": golds, "gold_papers": c["papers"], "reasoning": c["reasoning"]}
            excl_ans = len(ans) >= 4 and ans not in ("예", "아니오")
            dist = [r for r in rets[pid] if r["paper_id"] not in gold_papers and not (excl_ans and ans in norm(r["text"]))][:N_DIST]
            if len(dist) < N_DIST:
                sys.exit(f"{pid}: fewer than {N_DIST} distractors")
            docs = [{"doc_id": g, "text": chunk_text[g]} for g in golds] + [{"doc_id": r["chunk_id"], "text": r["text"]} for r in dist]
            random.Random(f"{SEED}-{pid}").shuffle(docs)
            gold_s.append({**base, "gold_doc_ids": golds, "documents": docs,
                           "annotation": {**ann, "gold_positions": [i for i, d in enumerate(docs) if d["doc_id"] in golds]}})
            top5 = rets[pid][:5]
            hit = [g in {r["chunk_id"] for r in top5} for g in golds]
            stats["both_gold_chunks_top5"] += all(hit)
            stats["one_gold_chunk_top5"] += sum(hit) == 1
            stats["both_gold_papers_top5"] += gold_papers <= {r["paper_id"] for r in top5}
            ret_s.append({**base, "gold_doc_ids": [g for g, h in zip(golds, hit) if h],
                          "documents": [{"doc_id": r["chunk_id"], "text": r["text"]} for r in top5],
                          "annotation": {**ann, "gold_chunks_in_top5": hit}})
        print(t, "retrieval top-5:", stats, "of", len(ret_s))
        gpol = {"protocol": "kisti_comp_gold2_dist3", "gold_policy": "both evidence chunks inserted",
                "distractors": "top e5 chunks for the question, gold papers (and answer-containing chunks if the answer has >= 4 chars) excluded",
                "order": "shuffled per sample (seed 0)"}
        rpol = {"protocol": "kisti_comp_retrieved", "gold_policy": "none", "retrieval": "e5 top-5 unique", "order": "retrieval score"}
        write(f"kisti_comp{t}_gold2_dist3_ko", gold_s, gpol)
        write(f"kisti_comp{t}_ret_top5_ko", ret_s, rpol)
        write(f"kisti_comp{t}_gold2_dist3_ko_full", with_full(gold_s), {**gpol, "k_final": 6})
        write(f"kisti_comp{t}_ret_top5_ko_full", with_full(ret_s), {**rpol, "k_final": 6})


if __name__ == "__main__":
    {"pairs": cmd_pairs, "compose": cmd_compose, "retrieve": cmd_retrieve, "manifest": cmd_manifest,
     "single_full": cmd_single_full, "cgen": cmd_cgen, "cretrieve": cmd_cretrieve, "cmanifest": cmd_cmanifest}[sys.argv[1]]()
