"""Step 31: KISTI multi-evidence benchmark — baseline (PISCO, 5 memories) vs baseline-6 (the same five chunk texts
re-split into six PISCO units, scripts/24_prepare_chunk6_manifests.py) vs ours (5 memories + Step 19 repetition
block): 6 units / 48 vectors in both baseline-6 and ours.

  CUDA_VISIBLE_DEVICES=0 ~/anaconda3/envs/refrag/bin/python scripts/31_c6_compare.py judge --set parA_gold
  python scripts/31_c6_compare.py score

Sets: Step 28 gold2_dist3 sets (parallel A/B, compositional A/B), pooled A+B per family as in Step 28.
Raw / baseline / ours labels are Step 28's (artifacts/step28/judge); baseline-6 is judged here with the same judge.
"""

import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ART = PROJECT_ROOT / "artifacts" / "step31"
SETS = {f"{fam}{t}_gold": f"kisti_{'me' if fam == 'par' else 'comp'}{t}_gold2_dist3_ko" for fam in ("par", "comp") for t in "AB"}


def load_score():
    argv, sys.argv = sys.argv, [sys.argv[0]]
    spec = importlib.util.spec_from_file_location("sc28", PROJECT_ROOT / "scripts" / "28_score.py")
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    sys.argv = argv
    return sc


def cmd_judge(set_name):
    sc = load_score()
    sc.SYSTEMS = {"c6": ("pisco", "_chunk6")}
    sc.SETS = {set_name: SETS[set_name]}
    sc.OUT = ART
    sc.cmd_judge(set_name)


def cmd_score():
    sc = load_score()
    names = ("raw", "pisco", "c6", "block")
    rows_by = {}
    for set_name, base in SETS.items():
        j = json.loads((PROJECT_ROOT / "artifacts" / "step28" / "judge" / f"{set_name}.json").read_text())
        jc = json.loads((ART / "judge" / f"{set_name}.json").read_text())
        preds = sc.predictions(base)
        if j["runs"] != {k: v["run"] for k, v in preds.items()}:
            sys.exit(f"{set_name}: Step 28 labels were computed on different runs")
        rows = []
        for sid, lab in j["labels"].items():
            r = {n: lab[n] for n in ("raw", "pisco", "block")}
            r["c6"] = jc["labels"][sid]["c6"]
            rows.append({n: {"correct": float(all(v)), "any": float(any(v)), "sub_mean": sum(v) / len(v)} for n, v in r.items()})
        rows_by[set_name] = rows
    groups = {**{k: [k] for k in SETS}, "parAB_gold": ["parA_gold", "parB_gold"], "compAB_gold": ["compA_gold", "compB_gold"],
              "allME_gold": list(SETS)}
    result, lines = {}, []
    for g, members in groups.items():
        rows = [r for m in members for r in rows_by[m]]
        metrics = ("correct", "any", "sub_mean") if g.startswith("par") else ("correct",)
        result[g] = {}
        for metric in metrics:
            res = {"n": len(rows), **{n: round(sum(r[n][metric] for r in rows) / len(rows), 3) for n in names}}
            for a, b in (("pisco", "c6"), ("pisco", "block"), ("c6", "block"), ("raw", "c6"), ("raw", "block")):
                res[f"{b}-{a}"] = sc.paired_ci([r[a][metric] for r in rows], [r[b][metric] for r in rows])
            result[g][metric] = res
            lines.append(f"{g:11s} {metric:8s} n={res['n']:4d} raw {res['raw']:.3f} base {res['pisco']:.3f} base6 {res['c6']:.3f} "
                         f"ours {res['block']:.3f} | base6-base {res['c6-pisco']} ours-base {res['block-pisco']} ours-base6 {res['block-c6']}")
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "scores.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    if sys.argv[1] == "judge":
        cmd_judge(sys.argv[sys.argv.index("--set") + 1])
    else:
        cmd_score()
