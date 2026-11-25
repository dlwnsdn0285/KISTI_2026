"""Step 32: KISTI multi-evidence benchmark — OURS (5 memories + Step 19 repetition block, PISCO decoder) vs two
decoder-LoRA models, Step 28 gold2_dist3 sets (parallel A/B, compositional A/B; pooled as in Step 28/31).

  dq      the same repetition block + decoder_adapter LoRA trained over [memories, frozen block]
          (Step 23 D_Q; 08_run_eval --checkpoint artifacts/step19/ckpt/rep_full/gen.pt --decoder-lora artifacts/step23/ckpt/D_Q/lora.pt)
  stage2  Step 15 stage2 = Step 13 append block (fused from the five memories + question, no text) frozen + decoder
          LoRA 2 epochs (artifacts/add/{fusion,lora}.pt, identical to artifacts/step15/ckpt/stage2)

  CUDA_VISIBLE_DEVICES=0 ~/anaconda3/envs/refrag/bin/python scripts/32_lora_compare.py judge --set parA_gold
  python scripts/32_lora_compare.py score
Raw / baseline / OURS labels are Step 28's; dq and stage2 are judged here with the same judge.
"""

import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ART = PROJECT_ROOT / "artifacts" / "step32"
SETS = {f"{fam}{t}_gold": f"kisti_{'me' if fam == 'par' else 'comp'}{t}_gold2_dist3_ko" for fam in ("par", "comp") for t in "AB"}
NEW = {"dq": ("fusion_rep_full+D_Q", "_full"), "stage2": ("fusion_add", "")}


def load_score():
    argv, sys.argv = sys.argv, [sys.argv[0]]
    spec = importlib.util.spec_from_file_location("sc28", PROJECT_ROOT / "scripts" / "28_score.py")
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    sys.argv = argv
    return sc


def cmd_judge(set_name):
    sc = load_score()
    sc.SYSTEMS = NEW
    sc.SETS = {set_name: SETS[set_name]}
    sc.OUT = ART
    sc.cmd_judge(set_name)


def cmd_score():
    sc = load_score()
    names = ("raw", "pisco", "block", "dq", "stage2")
    rows_by = {}
    for set_name, base in SETS.items():
        j = json.loads((PROJECT_ROOT / "artifacts" / "step28" / "judge" / f"{set_name}.json").read_text())
        jn = json.loads((ART / "judge" / f"{set_name}.json").read_text())
        preds = sc.predictions(base)
        if j["runs"] != {k: v["run"] for k, v in preds.items()}:
            sys.exit(f"{set_name}: Step 28 labels were computed on different runs")
        rows = []
        for sid, lab in j["labels"].items():
            r = {n: lab[n] for n in ("raw", "pisco", "block")}
            r.update({n: jn["labels"][sid][n] for n in NEW})
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
            for a, b in (("block", "dq"), ("block", "stage2"), ("dq", "stage2"), ("pisco", "dq"), ("pisco", "stage2"),
                         ("raw", "block"), ("raw", "dq"), ("raw", "stage2")):
                res[f"{b}-{a}"] = sc.paired_ci([r[a][metric] for r in rows], [r[b][metric] for r in rows])
            result[g][metric] = res
            lines.append(f"{g:11s} {metric:8s} n={res['n']:4d} raw {res['raw']:.3f} base {res['pisco']:.3f} ours {res['block']:.3f} "
                         f"ours+DQ {res['dq']:.3f} stage2 {res['stage2']:.3f} | DQ-ours {res['dq-block']} stage2-ours {res['stage2-block']} "
                         f"stage2-DQ {res['stage2-dq']}")
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "scores.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print("\n".join(lines))


# ---- 5 memories + decoder LoRA, no extra block (Step 12 decoder LoRA = Step 23 D_M), compositional sets only
DM = {"dm": ("fusion_decoder_lora", "")}      # 08_run_eval --checkpoint artifacts/step12/ckpt/decoder_lora/lora.pt
COMP = ("compA_gold", "compB_gold")


def cmd_judge_dm(set_name):
    sc = load_score()
    sc.SYSTEMS = DM
    sc.SETS = {set_name: SETS[set_name]}
    sc.OUT = ART / "dm"
    sc.cmd_judge(set_name)


def cmd_score_dm():
    sc = load_score()
    names = ("raw", "pisco", "dm", "block", "dq")
    rows = []
    for set_name in COMP:
        j = json.loads((PROJECT_ROOT / "artifacts" / "step28" / "judge" / f"{set_name}.json").read_text())
        jn = json.loads((ART / "judge" / f"{set_name}.json").read_text())
        jd = json.loads((ART / "dm" / "judge" / f"{set_name}.json").read_text())
        for sid, lab in j["labels"].items():
            r = {"raw": lab["raw"], "pisco": lab["pisco"], "block": lab["block"], "dq": jn["labels"][sid]["dq"],
                 "dm": jd["labels"][sid]["dm"]}
            rows.append({n: float(all(v)) for n, v in r.items()})
    res = {"n": len(rows), **{n: round(sum(r[n] for r in rows) / len(rows), 3) for n in names}}
    for a, b in (("pisco", "dm"), ("dm", "block"), ("dm", "dq"), ("pisco", "dq"), ("raw", "dm"), ("raw", "dq")):
        res[f"{b}-{a}"] = sc.paired_ci([r[a] for r in rows], [r[b] for r in rows])
    (ART / "scores_dm.json").write_text(json.dumps({"compAB_gold": res}, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    if sys.argv[1] == "judge":
        cmd_judge(sys.argv[sys.argv.index("--set") + 1])
    elif sys.argv[1] == "judge_dm":
        cmd_judge_dm(sys.argv[sys.argv.index("--set") + 1])
    elif sys.argv[1] == "score_dm":
        cmd_score_dm()
    else:
        cmd_score()
