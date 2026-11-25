"""Step 26 (KISTI): score raw vs PISCO on the KISTI Korean manifests.

Restored from this server's lost "Step 22 (KISTI)" scorer (see STEP_LOG Step 26); behaviour unchanged.

  # LLM judge (Qwen2.5-7B-Instruct, blind to the system; conda env `refrag`)
  CUDA_VISIBLE_DEVICES=2 ~/anaconda3/envs/refrag/bin/python scripts/26_kisti_score.py judge [--ko] [--index …]
  # metrics + paired bootstrap (any env)
  ~/anaconda3/envs/pisco-ref/bin/python scripts/26_kisti_score.py score [--ko] [--index …]
--ko scores the variant manifest whose questions end with an answer-in-Korean instruction.
--index child125 | m120 | m120v2 (default m120v2, the chosen baseline index) picks the retrieval index;
outputs go to artifacts/step26/<index>/<base|ko>.

KISTI references are long extractive spans (median 64 chars), so the paper's
substring Match is reported but is not the main metric. Main metrics:
  char_f1   F1 over characters after removing whitespace and punctuation
            (Korean has no stable word boundary for SQuAD token F1)
  ref_recall  share of reference characters (multiset) present in the prediction
  judge     Qwen2.5-7B-Instruct: is the prediction correct given question + reference (1/0)
Latest run per system with the manifest's sha256 is used (08_run_eval convention).
"""

import hashlib
import json
import random
import string
import sys
import unicodedata
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VARIANT = "_ko" if "--ko" in sys.argv[2:] else ""
INDEXES = {"child125": "kisti_sample500_e5child125_top5", "m120": "kisti_sample500_e5m120_top5",
           "m120v2": "kisti_sample500_e5m120v2_top5"}
INDEX = sys.argv[sys.argv.index("--index") + 1] if "--index" in sys.argv else "m120v2"
MANIFEST = PROJECT_ROOT / "data" / "manifests" / f"{INDEXES[INDEX]}{VARIANT}.json"
RUNS_DIR = PROJECT_ROOT / "artifacts" / "step08" / "runs"
OUT_DIR = PROJECT_ROOT / "artifacts" / "step26" / INDEX / (VARIANT.strip("_") or "base")
SYSTEMS = ("raw", "pisco")
JUDGE_DIR = "/data2/jwlee/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct"
JUDGE_PROMPT = """다음은 과학기술 논문에 대한 질문, 정답, 그리고 모델의 답변입니다.
모델의 답변이 정답의 핵심 내용을 맞게 담고 있으면 1, 틀리거나 핵심 내용이 없거나 답을 하지 않았으면 0으로 판정하세요.
답변의 언어(한국어/영어)나 표현 방식은 상관없고, 의미가 정답과 맞는지만 보세요. 정답에 없는 부가 설명이 있어도 핵심이 맞으면 1입니다.

질문: {question}
정답: {reference}
모델 답변: {prediction}

판정 (0 또는 1 한 글자만 출력):"""
N_BOOT, SEED = 2000, 0


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def latest_run(system, manifest_sha):
    runs = sorted(d for d in RUNS_DIR.glob(f"{system}-kisti_*") if (d / "run.json").exists())
    runs = [d for d in runs if json.loads((d / "run.json").read_text())["manifest"]["sha256"] == manifest_sha]
    if not runs:
        sys.exit(f"no finished {system} run for {MANIFEST.name}")
    return runs[-1]


def load_predictions(manifest_sha):
    preds = {}
    for system in SYSTEMS:
        run = latest_run(system, manifest_sha)
        rows = [json.loads(l) for l in (run / "predictions.jsonl").read_text().splitlines()]
        preds[system] = {"run": run.name, "rows": {r["sample_id"]: r for r in rows}}
    return preds


_PUNCT = set(string.punctuation) | {"·", "․", "‧", "“", "”", "‘", "’", "「", "」", "『", "』", "…", "∼", "～"}


def chars(text):
    text = unicodedata.normalize("NFKC", text).lower()
    return [c for c in text if not c.isspace() and c not in _PUNCT and unicodedata.category(c)[0] != "P"]


def char_f1(pred, ref):
    p, r = Counter(chars(pred)), Counter(chars(ref))
    common = sum((p & r).values())
    if not common:
        return 0.0
    prec, rec = common / sum(p.values()), common / sum(r.values())
    return 2 * prec * rec / (prec + rec)


def ref_recall(pred, ref):
    p, r = Counter(chars(pred)), Counter(chars(ref))
    return sum((p & r).values()) / max(1, sum(r.values()))


def substring_match(pred, ref):
    return "".join(chars(ref)) in "".join(chars(pred))


def hangul_ratio(text):
    cs = [c for c in text if c.isalpha()]
    return sum("가" <= c <= "힣" for c in cs) / len(cs) if cs else 0.0


def cmd_judge():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    snap = sorted(Path(JUDGE_DIR, "snapshots").iterdir())[-1]
    manifest_sha = sha256_file(MANIFEST)
    manifest = json.loads(MANIFEST.read_text())
    preds = load_predictions(manifest_sha)
    tok = AutoTokenizer.from_pretrained(snap)
    model = AutoModelForCausalLM.from_pretrained(snap, torch_dtype=torch.bfloat16).to("cuda").eval()
    ids01 = [tok.convert_tokens_to_ids(t) for t in ("0", "1")]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {"judge_model": str(snap), "prompt": JUDGE_PROMPT, "manifest_sha256": manifest_sha,
           "runs": {s: preds[s]["run"] for s in SYSTEMS}, "labels": {}}
    for s in manifest["samples"]:
        sid = s["sample_id"]
        for system in SYSTEMS:
            row = preds[system]["rows"].get(sid)
            pred = row["prediction"] if row and row["status"] == "ok" else ""
            msg = JUDGE_PROMPT.format(question=s["model_input"]["question"],
                                      reference=s["evaluation"]["reference_answers"][0], prediction=pred.strip() or "(빈 답변)")
            text = tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True)
            enc = tok(text, return_tensors="pt").to("cuda")
            with torch.no_grad():
                logits = model(**enc).logits[0, -1]
            p01 = torch.softmax(logits[ids01].float(), dim=0)
            out["labels"].setdefault(sid, {})[system] = {"judge": int(p01[1] > 0.5), "p1": round(float(p01[1]), 4)}
    (OUT_DIR / "judge.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("wrote", OUT_DIR / "judge.json")


def paired_ci(a, b):
    """mean(b - a) with paired bootstrap 95% CI, in %p."""
    rng = random.Random(SEED)
    n = len(a)
    diffs = [y - x for x, y in zip(a, b)]
    boots = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(N_BOOT))
    return [round(100 * sum(diffs) / n, 1), round(100 * boots[int(0.025 * N_BOOT)], 1), round(100 * boots[int(0.975 * N_BOOT)], 1)]


def cmd_score():
    manifest_sha = sha256_file(MANIFEST)
    manifest = json.loads(MANIFEST.read_text())
    preds = load_predictions(manifest_sha)
    judge_path = OUT_DIR / "judge.json"
    judge = json.loads(judge_path.read_text()) if judge_path.exists() else None
    if judge and judge["runs"] != {s: preds[s]["run"] for s in SYSTEMS}:
        sys.exit("judge.json was computed on different runs; rerun judge")
    rows = []
    for s in manifest["samples"]:
        sid, ref = s["sample_id"], s["evaluation"]["reference_answers"][0]
        r = {"sample_id": sid, "gold_paper_in_top5": s["annotation"]["gold_paper_in_top5"],
             "qa_level": s["annotation"]["qa_level"],
             "any_truncated": any(p["truncated"] for p in s["provenance"]["documents"])}
        for system in SYSTEMS:
            p = preds[system]["rows"].get(sid)
            text = p["prediction"] if p and p["status"] == "ok" else ""
            r[system] = {"prediction": text, "char_f1": char_f1(text, ref), "ref_recall": ref_recall(text, ref),
                         "match": float(substring_match(text, ref)), "hangul": hangul_ratio(text),
                         "korean_answer": float(hangul_ratio(text) >= 0.5),
                         "eos": float(bool(p and p.get("stopped_at_eos")))}
            if judge:
                r[system]["judge"] = float(judge["labels"][sid][system]["judge"])
        rows.append(r)

    metrics = ["judge", "char_f1", "ref_recall", "match", "korean_answer", "eos"] if judge else \
              ["char_f1", "ref_recall", "match", "korean_answer", "eos"]

    def summarize(subset):
        out = {"n": len(subset)}
        for m in metrics:
            a = [r["raw"][m] for r in subset]
            b = [r["pisco"][m] for r in subset]
            out[m] = {"raw": round(sum(a) / len(a), 3), "pisco": round(sum(b) / len(b), 3),
                      "pisco_minus_raw_pp_ci": paired_ci(a, b)}
        return out

    result = {"manifest": MANIFEST.name, "runs": {s: preds[s]["run"] for s in SYSTEMS},
              "all": summarize(rows),
              "gold_paper_in_top5": summarize([r for r in rows if r["gold_paper_in_top5"]]),
              "gold_paper_not_in_top5": summarize([r for r in rows if not r["gold_paper_in_top5"]]),
              "no_doc_truncated": summarize([r for r in rows if not r["any_truncated"]])}
    for lvl in sorted({r["qa_level"] for r in rows}):
        result[f"qa_level_{lvl}"] = summarize([r for r in rows if r["qa_level"] == lvl])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "scores.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    (OUT_DIR / "rows.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    {"judge": cmd_judge, "score": cmd_score}[sys.argv[1]]()
