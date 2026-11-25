"""Step 11e: Raw vs PISCO vs PISCO+fusion (cross-doc) vs PISCO+fusion (per-doc) on the evaluation manifests.

    ~/anaconda3/envs/pisco-ref/bin/python scripts/11_compare_fusion.py

For each manifest the latest completed run of each system is used (fusion runs are
matched by run-id prefix fusion_<tag>). Metrics: Match everywhere; yes/no-aware
Match for MultiHop-RAG. Differences are paired bootstrap CIs against the PISCO
baseline (same questions). Also reports the mean relative change of the memory
vectors made by each fusion variant.

Output: artifacts/step11/compare.json
"""

import hashlib
import json
import random
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pisco_research.scoring import match, yes_no_match  # noqa: E402

RUNS_DIR = PROJECT_ROOT / "artifacts" / "step08" / "runs"
MANIFESTS = {
    "musique_2hop": ("musique_2hop_gold2_dist3_n1000.json", "match"),
    "multihoprag_2ev": ("multihoprag_2ev_gold2_dist3_n1061.json", "yn_match"),
    "multihoprag_forced": ("multihoprag_2ev_yesno_forced_n882.json", "yn_match"),
    "hotpotqa_2hop": ("hotpotqa_2hop_gold2_dist3_n1000.json", "match"),
    "2wiki_2hop": ("2wiki_2hop_gold2_dist3_n1000.json", "match"),
    "nq_gold": ("nq_gold1_dist4_n1000.json", "match"),
    # Step 14: k-1 contexts (last distractor dropped) so the appended block is the 5th slot
    "musique_2hop_k4": ("musique_2hop_gold2_dist3_n1000_k4.json", "match"),
    "multihoprag_2ev_k4": ("multihoprag_2ev_gold2_dist3_n1061_k4.json", "yn_match"),
    "multihoprag_forced_k4": ("multihoprag_2ev_yesno_forced_n882_k4.json", "yn_match"),
    "nq_gold_k4": ("nq_gold1_dist4_n1000_k4.json", "match"),
}
SYSTEMS = {"raw": "raw", "pisco": "pisco", "fusion_cross": "fusion_cross", "fusion_perdoc": "fusion_perdoc",
           "decoder_lora": ("fusion_decoder_lora", "lora.pt"), "encoder_lora": "fusion_encoder_lora",
           "append": "fusion_append", "const": "fusion_const", "append_untrained": "fusion_append_untrained",
           "joint": "fusion_joint", "append_k4": "fusion_append_k4",
           "stage2": "fusion_stage2", "stage2_half": "fusion_stage2_half",
           "dlora500": ("fusion_decoder_lora", "lora_step500.pt")}   # run prefix comes from the checkpoint directory
N_BOOT, SEED = 2000, 0


def latest_run(prefix, manifest_path):
    """prefix: run-id prefix, or (prefix, checkpoint basename) when several checkpoints share a directory."""
    ckpt_name = None
    if isinstance(prefix, tuple):
        prefix, ckpt_name = prefix
    sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    runs = []
    for d in sorted(RUNS_DIR.glob(f"{prefix}-*")):
        if not (d / "run.json").exists():
            continue
        meta = json.loads((d / "run.json").read_text())
        if meta["manifest"]["sha256"] != sha:
            continue
        if ckpt_name and not str(meta.get("system_config", {}).get("checkpoint", "")).endswith("/" + ckpt_name):
            continue
        runs.append(d)
    return runs[-1] if runs else None


def paired_ci(a, b):
    """bootstrap CI of mean(b) - mean(a) over questions."""
    n = len(a)
    rng = random.Random(SEED)
    diffs = []
    for _ in range(N_BOOT):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(b[i] for i in idx) / n - sum(a[i] for i in idx) / n)
    diffs.sort()
    return [round(diffs[int(0.025 * N_BOOT)], 4), round(diffs[int(0.975 * N_BOOT) - 1], 4)]


def main():
    report = {}
    for name, (fname, metric) in MANIFESTS.items():
        path = PROJECT_ROOT / "data" / "manifests" / fname
        if not path.exists():
            continue
        manifest = json.loads(path.read_text())
        scorer = yes_no_match if metric == "yn_match" else match
        rows, runs, extra = {}, {}, {}
        for system, prefix in SYSTEMS.items():
            run_dir = latest_run(prefix, path)
            if run_dir is None:
                continue
            preds = {(r := json.loads(line))["sample_id"]: r for line in (run_dir / "predictions.jsonl").read_text().splitlines()}
            if any(preds.get(s["sample_id"], {}).get("status") != "ok" for s in manifest["samples"]):
                print(f"{name}/{system}: incomplete run {run_dir.name}, skipped")
                continue
            runs[system] = run_dir.name
            rows[system] = [float(scorer(preds[s["sample_id"]]["prediction"], s["evaluation"]["reference_answers"]))
                            for s in manifest["samples"]]
            if "memory_rel_change" in next(iter(preds.values())):
                extra[system] = {"memory_rel_change_mean": round(statistics.mean(p["memory_rel_change"] for p in preds.values()), 4)}
        if "pisco" not in rows:
            continue
        entry = {"manifest": fname, "metric": metric, "n": len(manifest["samples"]), "runs": runs, "accuracy": {},
                 "diff_vs_pisco": {}, "by_question_type": {}, "extra": extra}
        for system, v in rows.items():
            entry["accuracy"][system] = round(statistics.fmean(v), 4)
            if system != "pisco":
                entry["diff_vs_pisco"][system] = {"diff": round(statistics.fmean(v) - statistics.fmean(rows["pisco"]), 4),
                                                  "ci95": paired_ci(rows["pisco"], v)}
        if "fusion_cross" in rows and "fusion_perdoc" in rows:
            entry["diff_cross_vs_perdoc"] = {"diff": round(statistics.fmean(rows["fusion_cross"]) - statistics.fmean(rows["fusion_perdoc"]), 4),
                                             "ci95": paired_ci(rows["fusion_perdoc"], rows["fusion_cross"])}
        types = {}
        for i, s in enumerate(manifest["samples"]):
            types.setdefault(str(s["annotation"].get("question_type")), []).append(i)
        for t, idx in sorted(types.items()):
            entry["by_question_type"][t] = {"n": len(idx), **{sys_: round(sum(v[i] for i in idx) / len(idx), 4) for sys_, v in rows.items()}}
        report[name] = entry
        acc = " ".join(f"{s} {a:.3f}" for s, a in entry["accuracy"].items())
        print(f"{name:20s} [{metric}] n={entry['n']} {acc} | " + " ".join(
            f"{s}-pisco {d['diff']*100:+.1f} {d['ci95']}" for s, d in entry["diff_vs_pisco"].items() if s != "raw"))
    out = PROJECT_ROOT / "artifacts" / "step11" / "compare.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("written:", out.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
