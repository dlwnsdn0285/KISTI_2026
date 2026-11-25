"""Step 28 scoring: raw vs PISCO vs PISCO + Step 19 repetition block on the KISTI sets.

  # LLM judge per set (Qwen2.5-7B-Instruct, blind to the system; conda env `refrag`)
  CUDA_VISIBLE_DEVICES=0 ~/anaconda3/envs/refrag/bin/python scripts/28_score.py judge --set single
  # aggregate all judged sets, paired bootstrap (any env)
  python scripts/28_score.py score

Sets (manifest base; the block system reads <base>_full):
  single           kisti_sample500_e5m120v2_top5_ko (Step 26 single evidence, retrieved top-5)
  par{A,B}_{gold,ret}   kisti_me{A,B}_{gold2_dist3,ret_top5}_ko   parallel: two sub-questions, two extractive answers
  comp{A,B}_{gold,ret}  kisti_comp{A,B}_{gold2_dist3,ret_top5}_ko compositional: one short answer needing both chunks
Judging: single / comp = the Step 26 judge prompt (question, reference, prediction). parallel = each sub-answer
judged separately against its sub-question; correct = both sub-answers right (main), any = at least one.
Runs: latest finished run per system whose manifest sha256 matches (raw-, pisco-, fusion_rep_full-).
"""

import hashlib
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS = PROJECT_ROOT / "artifacts" / "step08" / "runs"
MAN = PROJECT_ROOT / "data" / "manifests"
OUT = PROJECT_ROOT / "artifacts" / "step28"
JUDGE_DIR = "/data2/jwlee/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct"
SYSTEMS = {"raw": ("raw", ""), "pisco": ("pisco", ""), "block": ("fusion_rep_full", "_full")}
N_BOOT, SEED = 2000, 0
SETS = {"single": "kisti_sample500_e5m120v2_top5_ko"}
for t in "AB":
    for short, st in (("gold", "gold2_dist3"), ("ret", "ret_top5")):
        SETS[f"par{t}_{short}"] = f"kisti_me{t}_{st}_ko"
        SETS[f"comp{t}_{short}"] = f"kisti_comp{t}_{st}_ko"

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import importlib.util
_spec = importlib.util.spec_from_file_location("s26", PROJECT_ROOT / "scripts" / "26_kisti_score.py")
s26 = importlib.util.module_from_spec(_spec)
_argv, sys.argv = sys.argv, [sys.argv[0]]          # 26_kisti_score parses sys.argv at import
_spec.loader.exec_module(s26)
sys.argv = _argv
JUDGE_PROMPT = s26.JUDGE_PROMPT
SUB_PROMPT = """다음은 과학기술 논문에 대한 질문(두 부분으로 구성), 그중 한 부분의 질문과 정답, 그리고 모델의 답변입니다.
모델의 답변이 이 부분 질문에 대해 정답의 핵심 내용을 맞게 담고 있으면 1, 틀리거나 이 부분에 대한 답이 없으면 0으로 판정하세요.
답변의 언어나 표현 방식은 상관없고, 의미가 정답과 맞는지만 보세요. 다른 부분에 대한 답이나 부가 설명이 있어도 이 부분의 핵심이 맞으면 1입니다.

전체 질문: {question}
이 부분의 질문: {sub_question}
이 부분의 정답: {reference}
모델 답변: {prediction}

판정 (0 또는 1 한 글자만 출력):"""


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def latest(prefix, manifest_path):
    sha = sha256(manifest_path)
    runs = [d for d in sorted(RUNS.glob(f"{prefix}-*")) if (d / "run.json").exists()
            and json.loads((d / "run.json").read_text())["manifest"]["sha256"] == sha]
    return runs[-1] if runs else None


def predictions(base):
    out = {}
    for name, (prefix, suffix) in SYSTEMS.items():
        run = latest(prefix, MAN / f"{base}{suffix}.json")
        if run is None:
            sys.exit(f"no finished {name} run for {base}{suffix}")
        rows = [json.loads(l) for l in (run / "predictions.jsonl").read_text().splitlines()]
        out[name] = {"run": run.name, "rows": {r["sample_id"]: (r["prediction"] if r["status"] == "ok" else "") for r in rows}}
    return out


def cmd_judge(set_name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = SETS[set_name]
    manifest = json.loads((MAN / f"{base}.json").read_text())
    preds = predictions(base)
    snap = sorted(Path(JUDGE_DIR, "snapshots").iterdir())[-1]
    tok = AutoTokenizer.from_pretrained(snap)
    model = AutoModelForCausalLM.from_pretrained(snap, torch_dtype=torch.bfloat16).to("cuda").eval()
    ids01 = [tok.convert_tokens_to_ids(t) for t in ("0", "1")]

    def judge(msg):
        text = tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = model(**enc).logits[0, -1]
        return int(torch.softmax(logits[ids01].float(), dim=0)[1] > 0.5)

    labels = {}
    for s in manifest["samples"]:
        sid, q = s["sample_id"], s["model_input"]["question"]
        refs = s["evaluation"]["reference_answers"]
        for name in SYSTEMS:
            pred = preds[name]["rows"].get(sid, "").strip() or "(빈 답변)"
            if set_name.startswith("par"):
                subs = s["annotation"]["sub_questions"]
                labels.setdefault(sid, {})[name] = [judge(SUB_PROMPT.format(question=q, sub_question=sq, reference=r, prediction=pred))
                                                    for sq, r in zip(subs, refs)]
            else:
                labels.setdefault(sid, {})[name] = [judge(JUDGE_PROMPT.format(question=q, reference=refs[0], prediction=pred))]
    OUT.joinpath("judge").mkdir(parents=True, exist_ok=True)
    (OUT / "judge" / f"{set_name}.json").write_text(json.dumps(
        {"set": set_name, "manifest": base, "judge_model": str(snap), "runs": {k: v["run"] for k, v in preds.items()},
         "labels": labels}, ensure_ascii=False, indent=1))
    print("wrote", OUT / "judge" / f"{set_name}.json")


def paired_ci(a, b):
    rng = random.Random(SEED)
    n = len(a)
    d = [y - x for x, y in zip(a, b)]
    boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(N_BOOT))
    return [round(100 * sum(d) / n, 1), round(100 * boots[int(0.025 * N_BOOT)], 1), round(100 * boots[int(0.975 * N_BOOT)], 1)]


def summarize(rows, metric):
    out = {"n": len(rows)}
    for name in SYSTEMS:
        out[name] = round(sum(r[name][metric] for r in rows) / len(rows), 3)
    for a, b in (("raw", "pisco"), ("pisco", "block"), ("raw", "block")):
        out[f"{b}-{a}"] = paired_ci([r[a][metric] for r in rows], [r[b][metric] for r in rows])
    return out


def cmd_score():
    result, all_rows = {}, {}
    for set_name, base in SETS.items():
        jp = OUT / "judge" / f"{set_name}.json"
        if not jp.exists():
            continue
        j = json.loads(jp.read_text())
        preds = predictions(base)
        if j["runs"] != {k: v["run"] for k, v in preds.items()}:
            sys.exit(f"{set_name}: judge labels were computed on different runs; rerun judge")
        manifest = {s["sample_id"]: s for s in json.loads((MAN / f"{base}.json").read_text())["samples"]}
        rows = []
        for sid, lab in j["labels"].items():
            ann = manifest[sid]["annotation"]
            r = {"sample_id": sid, "qtype": ann.get("qtype"),
                 "gold_hits": sum(ann["gold_chunks_in_top5"]) if "gold_chunks_in_top5" in ann else None}
            for name in SYSTEMS:
                v = lab[name]
                r[name] = {"correct": float(all(v)), "any": float(any(v)), "sub_mean": sum(v) / len(v)}
            rows.append(r)
        res = {"manifest": base, "runs": j["runs"], "all": summarize(rows, "correct")}
        if set_name.startswith("par"):
            res["any_correct"] = summarize(rows, "any")
            res["sub_mean"] = summarize(rows, "sub_mean")
        if set_name.startswith("comp"):
            for q in sorted({r["qtype"] for r in rows}):
                res[f"qtype_{q}"] = summarize([r for r in rows if r["qtype"] == q], "correct")
        if rows[0]["gold_hits"] is not None:
            for h in (2, 1, 0):
                sub = [r for r in rows if r["gold_hits"] == h]
                if sub:
                    res[f"gold_chunks_in_top5_{h}"] = summarize(sub, "correct")
        result[set_name] = res
        all_rows[set_name] = rows
        a = res["all"]
        print(f"{set_name:12s} n={a['n']} raw {a['raw']:.3f} pisco {a['pisco']:.3f} block {a['block']:.3f} | "
              f"pisco-raw {a['pisco-raw']} block-pisco {a['block-pisco']} block-raw {a['block-raw']}")
    # A + B pooled per family and setting (disjoint samples, same systems)
    for fam in ("par", "comp"):
        for st in ("gold", "ret"):
            parts = [f"{fam}{t}_{st}" for t in "AB"]
            if all(p in all_rows for p in parts):
                rows = all_rows[parts[0]] + all_rows[parts[1]]
                res = {"pooled_from": parts, "all": summarize(rows, "correct")}
                if fam == "par":
                    res["any_correct"] = summarize(rows, "any")
                    res["sub_mean"] = summarize(rows, "sub_mean")
                else:
                    for q in sorted({r["qtype"] for r in rows}):
                        res[f"qtype_{q}"] = summarize([r for r in rows if r["qtype"] == q], "correct")
                if st == "ret":
                    for h in (2, 1, 0):
                        res[f"gold_chunks_in_top5_{h}"] = summarize([r for r in rows if r["gold_hits"] == h], "correct")
                result[f"{fam}AB_{st}"] = res
                a = res["all"]
                print(f"{fam + 'AB_' + st:12s} n={a['n']} raw {a['raw']:.3f} pisco {a['pisco']:.3f} block {a['block']:.3f} | "
                      f"pisco-raw {a['pisco-raw']} block-pisco {a['block-pisco']} block-raw {a['block-raw']}")
    # all four multi-evidence sets pooled (parallel A/B + compositional A/B); "correct" = both sub-answers for parallel
    for st in ("gold", "ret"):
        parts = [f"{fam}{t}_{st}" for fam in ("par", "comp") for t in "AB"]
        if all(p in all_rows for p in parts):
            rows = [r for p in parts for r in all_rows[p]]
            res = {"pooled_from": parts, "all": summarize(rows, "correct")}
            for fam in ("par", "comp"):
                res[fam] = summarize([r for p in parts if p.startswith(fam) for r in all_rows[p]], "correct")
            result[f"allME_{st}"] = res
            a = res["all"]
            print(f"{'allME_' + st:12s} n={a['n']} raw {a['raw']:.3f} pisco {a['pisco']:.3f} block {a['block']:.3f} | "
                  f"pisco-raw {a['pisco-raw']} block-pisco {a['block-pisco']} block-raw {a['block-raw']}")
    OUT.joinpath("scores.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print("wrote", OUT / "scores.json")


if __name__ == "__main__":
    if sys.argv[1] == "judge":
        cmd_judge(sys.argv[sys.argv.index("--set") + 1])
    else:
        cmd_score()
