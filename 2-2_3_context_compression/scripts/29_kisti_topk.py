"""Step 29: KISTI accuracy vs number of documents k (5, 10, 20, 40): raw vs PISCO (baseline) vs PISCO + Step 19
repetition block (ours; block always built from the k=5 documents, as in the Step 27 latency runs).

  # 1) deeper e5 rankings (one GPU; conda env `refrag`)
  CUDA_VISIBLE_DEVICES=0 ~/anaconda3/envs/refrag/bin/python scripts/29_kisti_topk.py retrieve
  # 2) manifests (`pisco-ref`); k=5 must reproduce the Step 26 / Step 28 manifests' documents
  ~/anaconda3/envs/pisco-ref/bin/python scripts/29_kisti_topk.py manifest
  # 3) judge + score (`refrag` for judge)
  CUDA_VISIBLE_DEVICES=0 ~/anaconda3/envs/refrag/bin/python scripts/29_kisti_topk.py judge --set single_k10
  python scripts/29_kisti_topk.py score

Sets
  single_k{k}       kisti_single500_v2_top{k}_ko: e5 top-k unique chunks (v2 index, retrieval order), Step 26 questions
                    (k=5: the Step 26 manifest kisti_sample500_e5m120v2_top5_ko)
  comp{A,B}_k{k}    kisti_comp{A,B}_gold2_dist{k-2}_ko: the two evidence chunks + k-2 distractors = the three Step 28
                    distractors followed by the next chunks of a deeper e5 ranking (gold papers and answer-containing
                    chunks excluded), all k shuffled per sample; the k=5 documents are a subset of every larger k
  <manifest>_full   same documents + a last document = the k=5 documents joined (input of the block generator)
Judging and statistics: Step 28 (Qwen2.5-7B, Step 26 judge prompt; paired bootstrap 95% CI).
"""

import importlib.util
import json
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "data" / "kisti"
MAN = PROJECT_ROOT / "data" / "manifests"
ART = PROJECT_ROOT / "artifacts" / "step29"
RANKED = OUT / "topk_ranked.json"
KS = (5, 10, 20, 40)
SEARCH_K = 256


def load(name, file):
    argv, sys.argv = sys.argv, [sys.argv[0]]
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.argv = argv
    return mod


s28 = load("s28", "28_kisti_multi_evidence.py")


def manifest_ids():
    ids = {}
    for k in KS:
        ids[f"single_k{k}"] = "kisti_sample500_e5m120v2_top5_ko" if k == 5 else f"kisti_single500_v2_top{k}_ko"
        for t in "AB":
            ids[f"comp{t}_k{k}"] = f"kisti_comp{t}_gold2_dist{k - 2}_ko"
    return ids


def cmd_retrieve():
    s26 = load("s26c", "26_kisti_prepare_child125.py")
    queries = json.loads(s26.QUERY_FILE.read_text())
    gen = json.loads(s28.CGEN.read_text())
    cands = {c["pair_id"]: c for c in gen["candidates"]}
    comp_ids = [pid for t in "AB" for pid in gen[f"{t}_final_ids"]]
    s28.SEARCH_K = SEARCH_K
    single = s28.e5_rank([q["query"] for q in queries])
    comp = s28.e5_rank([cands[pid]["question"] for pid in comp_ids])
    keep = lambda r: {k: r[k] for k in ("chunk_id", "paper_id", "text")}
    RANKED.write_text(json.dumps({"search_k": SEARCH_K,
                                  "single": {q["query_id"]: [keep(r) for r in rk[:60]] for q, rk in zip(queries, single)},
                                  "comp": {pid: [keep(r) for r in rk] for pid, rk in zip(comp_ids, comp)}}, ensure_ascii=False))
    print("wrote", RANKED)


def cmd_manifest():
    os.environ["HF_HUB_OFFLINE"] = "1"
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from pisco_research import eval_inputs
    ref = json.loads((PROJECT_ROOT / "reference" / "manifest.json").read_text())
    tok = eval_inputs.load_pisco_tokenizer(ref["snapshots"]["backbone"]["local_dir"])
    tok_source = {"repo_id": ref["backbone"]["repo_id"], "revision": ref["backbone"]["revision_sha"],
                  "added_special_tokens": eval_inputs.PISCO_ADDED_TOKENS}
    ranked = json.loads(RANKED.read_text())
    gen = json.loads(s28.CGEN.read_text())
    cands = {c["pair_id"]: c for c in gen["candidates"]}
    chunk_text = {r["chunk_id"]: r["text"] for r in s28.load_index_rows(("chunk_id", "text"))}

    def write(manifest_id, samples, source, policy):
        fixture = {"fixture_id": manifest_id, "source": source,
                   "preprocessing": {"question": f"question + {s28.KO_SUFFIX!r}", "doc_text": "Step 26 v2 chunk text"},
                   "context_policy": policy, "samples": samples}
        fp = OUT / f"{manifest_id}_fixture.json"
        fp.write_text(json.dumps(fixture, ensure_ascii=False))
        manifest = eval_inputs.build_manifest(fp, tok, tok_source, manifest_id)
        path = MAN / f"{manifest_id}.json"
        if path.exists() and eval_inputs.sha256_json(json.loads(path.read_text())["samples"]) != eval_inputs.sha256_json(manifest["samples"]):
            sys.exit(f"{path} exists with different samples; refusing to overwrite")
        path.write_text(json.dumps(manifest, ensure_ascii=False))
        print(f"wrote {path.name}: {len(samples)} samples")

    def with_full(samples, block_docs):
        return [{**s, "documents": s["documents"] + [{"doc_id": f"{s['sample_id']}#full",
                                                     "text": s28.SEP.join(d["text"] for d in block_docs[s["sample_id"]])}],
                 "annotation": {**s["annotation"], "block_doc_ids": [d["doc_id"] for d in block_docs[s["sample_id"]]]}}
                for s in samples]

    def docs_of(manifest_id):
        return {s["sample_id"]: [{"doc_id": d["doc_id"], "text": d["text"]} for d in s["model_input"]["documents"]]
                for s in json.loads((MAN / f"{manifest_id}.json").read_text())["samples"]}

    # ---- single evidence: top-k in retrieval order; k=5 must equal Step 26
    step26 = json.loads((MAN / "kisti_sample500_e5m120v2_top5_ko.json").read_text())
    s26_docs = docs_of("kisti_sample500_e5m120v2_top5_ko")
    same = sum([d["doc_id"] for d in s26_docs[s["sample_id"]]] == [r["chunk_id"] for r in ranked["single"][s["sample_id"]][:5]]
               for s in step26["samples"])
    print(f"single: top-5 identical to Step 26 for {same}/{len(step26['samples'])} questions")
    if same != len(step26["samples"]):
        sys.exit("re-ranked top-5 differs from Step 26; aborting")
    for k in KS[1:]:
        samples = []
        for s in step26["samples"]:
            top = ranked["single"][s["sample_id"]][:k]
            gold = s["annotation"]["gold_paper"]
            samples.append({"sample_id": s["sample_id"], "question": s["model_input"]["question"],
                            "reference_answers": s["evaluation"]["reference_answers"], "gold_doc_ids": [],
                            "documents": [{"doc_id": r["chunk_id"], "text": r["text"]} for r in top],
                            "annotation": {**s["annotation"], "gold_paper_in_topk": any(r["paper_id"] == gold for r in top),
                                           "k": k}})
        mid = f"kisti_single500_v2_top{k}_ko"
        src = {"dataset": "KISTI 500 (Step 26)", "ranked": str(RANKED.relative_to(PROJECT_ROOT))}
        pol = {"protocol": "kisti_retrieved_topk", "k": k, "order": "retrieval score", "block_input": "top-5 chunks"}
        write(mid, samples, src, pol)
        write(f"{mid}_full", with_full(samples, s26_docs), src, {**pol, "k_final": k + 1})

    # ---- compositional: gold 2 + first k-2 distractors of one list, shuffled; k=5 must equal Step 28
    for t in "AB":
        base5 = f"kisti_comp{t}_gold2_dist3_ko"
        m5 = json.loads((MAN / f"{base5}.json").read_text())
        d5 = docs_of(base5)
        bad = 0
        lists = {}
        for s in m5["samples"]:
            pid = s["sample_id"]
            c = cands[pid]
            ans = s28.norm(c["answer"])
            excl = len(ans) >= 4 and ans not in ("예", "아니오")
            dist = [r for r in ranked["comp"][pid] if r["paper_id"] not in set(c["papers"]) and not (excl and ans in s28.norm(r["text"]))]
            if {d["doc_id"] for d in d5[pid]} != set(c["chunks"]) | {r["chunk_id"] for r in dist[:3]}:
                bad += 1           # re-encoding in a different batch flips near-ties; the Step 28 distractors stay first
            first = [{"chunk_id": d["doc_id"], "text": d["text"]} for d in d5[pid] if d["doc_id"] not in c["chunks"]]
            used = {d["chunk_id"] for d in first}
            dist = first + [r for r in dist if r["chunk_id"] not in used]
            lists[pid] = dist
            if len(dist) < max(KS) - 2:
                sys.exit(f"{pid}: only {len(dist)} distractors")
        print(f"comp {t}: re-ranked top-3 distractors differing from Step 28: {bad}/{len(m5['samples'])} "
              f"(Step 28 distractors kept first, so every k contains the k=5 documents)")
        for k in KS[1:]:
            samples = []
            for s in m5["samples"]:
                pid = s["sample_id"]
                c = cands[pid]
                docs = [{"doc_id": g, "text": chunk_text[g]} for g in c["chunks"]] + \
                       [{"doc_id": r["chunk_id"], "text": r["text"]} for r in lists[pid][:k - 2]]
                random.Random(f"29-{k}-{pid}").shuffle(docs)
                samples.append({"sample_id": pid, "question": s["model_input"]["question"],
                                "reference_answers": s["evaluation"]["reference_answers"], "gold_doc_ids": c["chunks"],
                                "documents": docs,
                                "annotation": {**s["annotation"], "k": k,
                                               "gold_positions": [i for i, d in enumerate(docs) if d["doc_id"] in c["chunks"]]}})
            mid = f"kisti_comp{t}_gold2_dist{k - 2}_ko"
            src = {"dataset": "synthetic compositional KISTI QA (Step 28)", "ranked": str(RANKED.relative_to(PROJECT_ROOT))}
            pol = {"protocol": "kisti_comp_gold2_distk", "k": k, "order": "shuffled per sample",
                   "block_input": "the k=5 documents (Step 28 gold2_dist3 order)"}
            write(mid, samples, src, pol)
            write(f"{mid}_full", with_full(samples, d5), src, {**pol, "k_final": k + 1})


def cmd_judge(set_name):
    sc = load("sc28", "28_score.py")
    sc.SETS = manifest_ids()
    sc.OUT = ART
    sc.cmd_judge(set_name)


def cmd_score():
    sc = load("sc28", "28_score.py")
    ids = manifest_ids()
    rows_by = {}
    for set_name, base in ids.items():
        jp = ART / "judge" / f"{set_name}.json"
        if not jp.exists():
            continue
        j = json.loads(jp.read_text())
        preds = sc.predictions(base)
        if j["runs"] != {k: v["run"] for k, v in preds.items()}:
            sys.exit(f"{set_name}: judge labels were computed on different runs")
        man = {s["sample_id"]: s for s in json.loads((MAN / f"{base}.json").read_text())["samples"]}
        rows = []
        for sid, lab in j["labels"].items():
            ann = man[sid]["annotation"]
            r = {"sample_id": sid, "gold_in": ann.get("gold_paper_in_topk", ann.get("gold_paper_in_top5"))}
            for name in sc.SYSTEMS:
                r[name] = {"correct": float(all(lab[name]))}
            rows.append(r)
        rows_by[set_name] = rows
    result, lines = {}, []
    for k in KS:
        groups = {"single": rows_by.get(f"single_k{k}"),
                  "comp": (rows_by[f"compA_k{k}"] + rows_by[f"compB_k{k}"]) if f"compA_k{k}" in rows_by and f"compB_k{k}" in rows_by else None}
        for g, rows in groups.items():
            if not rows:
                continue
            res = {"all": sc.summarize(rows, "correct")}
            if g == "single":
                res["gold_paper_in_topk_rate"] = round(sum(r["gold_in"] for r in rows) / len(rows), 3)
            result[f"{g}_k{k}"] = res
            a = res["all"]
            lines.append(f"{g:6s} k={k:2d} n={a['n']} raw {a['raw']:.3f} pisco {a['pisco']:.3f} ours {a['block']:.3f} | "
                         f"pisco-raw {a['pisco-raw']} ours-pisco {a['block-pisco']} ours-raw {a['block-raw']}"
                         + (f" | gold paper in top-k {res['gold_paper_in_topk_rate']}" if g == "single" else ""))
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "scores.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print("\n".join(lines))


# ============================================================ Step 30: one block per five documents

GROUP = 5


def cmd_manifest_mblk():
    """<base>_mblk for k > 5: documents in manifest order, after every GROUP documents a generator input
    '<sid>#blk<j>' = those GROUP chunk texts joined (one repetition block per group, placed right after it)."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from pisco_research import eval_inputs
    ref = json.loads((PROJECT_ROOT / "reference" / "manifest.json").read_text())
    tok = eval_inputs.load_pisco_tokenizer(ref["snapshots"]["backbone"]["local_dir"])
    tok_source = {"repo_id": ref["backbone"]["repo_id"], "revision": ref["backbone"]["revision_sha"],
                  "added_special_tokens": eval_inputs.PISCO_ADDED_TOKENS}
    for set_name, base in manifest_ids().items():
        if set_name.endswith("_k5"):
            continue                                   # one block: identical to the Step 29 `_full` layout
        src = json.loads((MAN / f"{base}.json").read_text())
        samples = []
        for s in src["samples"]:
            docs = [{"doc_id": d["doc_id"], "text": d["text"]} for d in s["model_input"]["documents"]]
            out = []
            for j in range(0, len(docs), GROUP):
                grp = docs[j:j + GROUP]
                out += grp + [{"doc_id": f"{s['sample_id']}#blk{j // GROUP}", "text": s28.SEP.join(d["text"] for d in grp)}]
            samples.append({"sample_id": s["sample_id"], "question": s["model_input"]["question"],
                            "reference_answers": s["evaluation"]["reference_answers"], "gold_doc_ids": s["evaluation"]["gold_doc_ids"],
                            "documents": out, "annotation": {**s["annotation"], "n_blocks": len(docs) // GROUP}})
        mid = f"{base}_mblk"
        fixture = {"fixture_id": mid, "source": {**src.get("source", {}), "derived_from": base},
                   "preprocessing": {"blocks": f"one generator input after every {GROUP} documents (their texts joined)"},
                   "context_policy": {"k": len(samples[0]["documents"]), "layout": "[m1..m5, b1, m6..m10, b2, ...]"},
                   "samples": samples}
        fp = OUT / f"{mid}_fixture.json"
        fp.write_text(json.dumps(fixture, ensure_ascii=False))
        manifest = eval_inputs.build_manifest(fp, tok, tok_source, mid)
        path = MAN / f"{mid}.json"
        if path.exists() and eval_inputs.sha256_json(json.loads(path.read_text())["samples"]) != eval_inputs.sha256_json(manifest["samples"]):
            sys.exit(f"{path} exists with different samples; refusing to overwrite")
        path.write_text(json.dumps(manifest, ensure_ascii=False))
        print(f"wrote {path.name}: {len(samples)} samples, {samples[0]['annotation']['n_blocks']} blocks each")


def cmd_judge_mblk(set_name):
    """Judge only the multi-block system (same judge as Step 28/29) -> artifacts/step29/judge_mblk/<set>.json."""
    sc = load("sc28", "28_score.py")
    base = manifest_ids()[set_name]
    sc.SYSTEMS = {"mblock": ("fusion_rep_full", "_full" if set_name.endswith("_k5") else "_mblk")}
    sc.SETS = {set_name: base}
    sc.OUT = ART / "mblk"
    sc.cmd_judge(set_name)


def cmd_score_mblk():
    sc = load("sc28", "28_score.py")
    result, lines = {}, []
    names = ("raw", "pisco", "block", "mblock")
    for k in KS:
        rows = []
        for g in (["single"], ["compA", "compB"]):
            grp_rows = []
            for set_name in [f"{x}_k{k}" for x in g]:
                j = json.loads((ART / "judge" / f"{set_name}.json").read_text())
                jm = json.loads((ART / "mblk" / "judge" / f"{set_name}.json").read_text())
                for sid, lab in j["labels"].items():
                    r = {n: {"correct": float(all(lab[n]))} for n in ("raw", "pisco", "block")}
                    r["mblock"] = {"correct": float(all(jm["labels"][sid]["mblock"]))}
                    grp_rows.append(r)
            key = f"{'single' if g == ['single'] else 'comp'}_k{k}"
            res = {"n": len(grp_rows)}
            for n in names:
                res[n] = round(sum(r[n]["correct"] for r in grp_rows) / len(grp_rows), 3)
            for a_, b_ in (("raw", "pisco"), ("pisco", "mblock"), ("raw", "mblock"), ("block", "mblock")):
                res[f"{b_}-{a_}"] = sc.paired_ci([r[a_]["correct"] for r in grp_rows], [r[b_]["correct"] for r in grp_rows])
            result[key] = res
            lines.append(f"{key:10s} n={res['n']} raw {res['raw']:.3f} pisco {res['pisco']:.3f} ours1 {res['block']:.3f} "
                         f"ours_per5 {res['mblock']:.3f} | per5-pisco {res['mblock-pisco']} per5-raw {res['mblock-raw']} "
                         f"per5-ours1 {res['mblock-block']}")
    (ART / "scores_mblk.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd in ("manifest_mblk", "score_mblk"):
        {"manifest_mblk": cmd_manifest_mblk, "score_mblk": cmd_score_mblk}[cmd]()
        sys.exit()
    if cmd == "judge_mblk":
        cmd_judge_mblk(sys.argv[sys.argv.index("--set") + 1])
        sys.exit()
    if cmd == "retrieve":
        cmd_retrieve()
    elif cmd == "manifest":
        cmd_manifest()
    elif cmd == "judge":
        cmd_judge(sys.argv[sys.argv.index("--set") + 1])
    else:
        cmd_score()
