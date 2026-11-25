"""성능·응답시간 요약 (보고서 2장 2·3절 표 순서).

    python run/summary.py

영문 세트는 artifacts/step08/runs의 예측을 직접 채점합니다(manifest 해시가 같은 가장 최근 run, Match /
MultiHop-RAG는 yes/no-aware Match). KISTI와 응답시간은 run/kisti.sh, run/latency.sh 산출물을 읽습니다.
없는 결과는 '-'로 표시합니다. 차이의 [ ]는 같은 문항 paired bootstrap 95% CI(2000회, seed 0)입니다.
"""

import csv
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from pisco_research.scoring import match, yes_no_match  # noqa: E402

RUNS = ROOT / "artifacts" / "step08" / "runs"
MAN = ROOT / "data" / "manifests"
ART = ROOT / "artifacts"

# 이름: (manifest, 채점)
SETS = {"NQ": ("nq_gold1_dist4_n1000", match), "HotpotQA": ("hotpotqa_2hop_gold2_dist3_n1000", match),
        "2Wiki": ("2wiki_2hop_gold2_dist3_n1000", match), "MuSiQue": ("musique_2hop_gold2_dist3_n1000", match),
        "MultiHop-RAG": ("multihoprag_2ev_yesno_forced_n882", yes_no_match)}
# 방법: (run 접두사, manifest 접미사, 체크포인트 파일명; 같은 폴더의 중간 체크포인트 run은 제외)
METHODS = {"원문": ("raw", "", None), "Baseline": ("pisco", "", None),
           "Decoder LoRA": ("fusion_decoder_lora", "", "lora.pt"), "Baseline-6": ("pisco", "_chunk6", None),
           "OURS": ("fusion_rep_full", "_full", "gen.pt")}


def predictions(prefix, manifest_id, ckpt=None):
    path = MAN / f"{manifest_id}.json"
    if not path.exists():
        return None
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    for d in sorted(RUNS.glob(f"{prefix}-*"), reverse=True):
        meta = d / "run.json"
        if not meta.exists():
            continue
        meta = json.loads(meta.read_text())
        if meta["manifest"]["sha256"] != sha:
            continue
        if ckpt and not str(meta.get("system_config", {}).get("checkpoint", "")).endswith("/" + ckpt):
            continue
        rows = [json.loads(line) for line in (d / "predictions.jsonl").read_text().splitlines()]
        if all(r["status"] == "ok" for r in rows):
            return {r["sample_id"]: r["prediction"] for r in rows}
    return None


def correct(name, method):
    """같은 문항 순서의 0/1 목록, 또는 run이 없으면 None."""
    base, scorer = SETS[name]
    prefix, suffix, ckpt = METHODS[method]
    preds = predictions(prefix, base + suffix, ckpt)
    if preds is None:
        return None
    samples = json.loads((MAN / f"{base}.json").read_text())["samples"]
    return [int(scorer(preds[s["sample_id"]], s["evaluation"]["reference_answers"])) for s in samples]


def paired_ci(a, b, n_boot=2000, seed=0):
    rng, n = random.Random(seed), len(a)
    diffs = sorted(sum(b[i] - a[i] for i in (rng.randrange(n) for _ in range(n))) / n for _ in range(n_boot))
    return diffs[int(0.025 * n_boot)] * 100, diffs[int(0.975 * n_boot) - 1] * 100


def diff(a, b):
    """b − a (%p) [CI]."""
    if a is None or b is None:
        return "-"
    lo, hi = paired_ci(a, b)
    return f"{(sum(b) - sum(a)) / len(a) * 100:+.1f} [{lo:+.1f}, {hi:+.1f}]"


def acc(v):
    return "-" if v is None else f"{sum(v) / len(v):.3f}"


def table(header, rows):
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for r in rows:
        print("| " + " | ".join(r) + " |")
    print()


def read(path):
    p = ART / path
    return json.loads(p.read_text()) if p.exists() else {}


def main():
    print("## 원문 대비 압축 성능 격차 (Match)\n")
    rows = []
    for name in SETS:
        raw, pisco = correct(name, "원문"), correct(name, "Baseline")
        rows.append([name, acc(raw), acc(pisco), diff(raw, pisco)])
    table(["데이터셋", "원문", "PISCO", "PISCO − 원문"], rows)

    final = correct("MuSiQue", "Baseline")
    hop1 = predictions("pisco", "musique_2hop_hop1_n1000")
    hop2 = predictions("pisco", "musique_2hop_hop2_n1000")
    if final and hop1 and hop2:
        print("## MuSiQue 2-evidence 오류 유형 (PISCO)\n")
        counts = {2: 0, 1: 0, 0: 0}
        refs = {}
        for hop, preds in (("hop1", hop1), ("hop2", hop2)):
            samples = json.loads((MAN / f"musique_2hop_{hop}_n1000.json").read_text())["samples"]
            refs[hop] = [int(match(preds[s["sample_id"]], s["evaluation"]["reference_answers"])) for s in samples]
        for f, a, b in zip(final, refs["hop1"], refs["hop2"]):
            if not f:
                counts[a + b] += 1
        wrong = len(final) - sum(final)
        print(f"최종 정답 {sum(final)}, 오답 {wrong}. 오답 중 하위 질문 둘 다 정답 {counts[2]} ({counts[2] / wrong:.1%}), "
              f"하나만 정답 {counts[1]} ({counts[1] / wrong:.1%}), 둘 다 오답 {counts[0]} ({counts[0] / wrong:.1%})\n")

    print("## 단일·다중 증거 성능 (5 memory + 추가 블록 1)\n")
    rows = []
    for name in ("MuSiQue", "MultiHop-RAG", "NQ"):
        res = {m: correct(name, m) for m in METHODS}
        rows.append([name] + [acc(res[m]) for m in METHODS] + [diff(res["Baseline"], res["OURS"])])
    k28 = read("step28/scores.json").get("compAB_gold", {}).get("all", {})
    k31 = read("step31/scores.json").get("compAB_gold", {})
    k31 = k31.get("correct", k31)
    k32 = read("step32/scores_dm.json").get("compAB_gold", {})
    fmt = lambda v: "-" if v is None else f"{v:.3f}"
    ci = k28.get("block-pisco")
    rows.append(["KISTI multi-evidence QA", fmt(k28.get("raw")), fmt(k28.get("pisco")), fmt(k32.get("dm")),
                 fmt(k31.get("c6")), fmt(k28.get("block")),
                 f"{ci[0]:+.1f} [{ci[1]:+.1f}, {ci[2]:+.1f}]" if ci else "-"])
    table(["데이터셋"] + list(METHODS) + ["OURS − Baseline"], rows)

    print("## 문서 수 증가 (KISTI multi-evidence QA, 판정 정확도)\n")
    k29 = read("step29/scores.json")
    rows = []
    for k in (5, 10, 20, 40):
        a = k29.get(f"comp_k{k}", {}).get("all", {})
        ci = a.get("block-pisco")
        rows.append([str(k), fmt(a.get("raw")), fmt(a.get("pisco")), fmt(a.get("block")),
                     f"{ci[0]:+.1f} [{ci[1]:+.1f}, {ci[2]:+.1f}]" if ci else "-"])
    table(["k", "원문", "Baseline", "OURS", "OURS − Baseline"], rows)

    curves = ART / "step27" / "curves.csv"
    if curves.exists():
        print("## 응답시간 (원문 대비 가속 = 원문 / 방법, 처리량은 방법 / 원문)\n")
        by_n = {}
        for r in csv.DictReader(curves.open()):
            by_n.setdefault(r["n_docs"], {})[r["method"]] = r
        rows = []
        for n, m in by_n.items():
            raw, ours, online = m["raw"], m["ours"], m["online"]
            rows.append([n, f"{float(raw['raw_input_tokens']):,.0f}", f"{float(raw['ttft_s']):.3f}",
                         f"{float(ours['ttft_s']):.3f} ({float(ours['ttft_accel']):.2f}×)",
                         f"{float(online['ttft_s']):.3f} ({float(online['ttft_accel']):.2f}×)",
                         f"{float(ours['ttit_accel']):.2f}×", f"{float(ours['throughput_accel']):.2f}×"])
        table(["N", "원문 토큰", "원문 TTFT(s)", "OURS TTFT", "online TTFT", "OURS TTIT", "OURS 처리량"], rows)


if __name__ == "__main__":
    main()
