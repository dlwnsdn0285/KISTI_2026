"""Step 09c/09d analysis: Raw (Mistral-7B, text) vs compressed (PISCO) on the fixed-context manifests.

    ~/anaconda3/envs/pisco-ref/bin/python scripts/09c_compare_raw_vs_compressed.py

Reads the latest completed pisco/raw run for each manifest (same manifest
sha256), scores them with the shared scorer, and reports per dataset:
Match / EM / F1 with bootstrap 95% CIs (resampling questions, paired between
systems), the paired difference, a question-level contingency table, a
breakdown by question type and by evidence preservation, and answer length.

Output: artifacts/step09c/raw_vs_compressed.json
"""

import hashlib
import json
import random
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pisco_research.scoring import score_run  # noqa: E402

DATASETS = {  # name -> manifest file; NQ (Step 09d) is 1 gold + 4 distractors, not a fair counterpart
    "hotpotqa": "hotpotqa_2hop_gold2_dist3_n1000.json",
    "2wiki": "2wiki_2hop_gold2_dist3_n1000.json",
    "musique": "musique_2hop_gold2_dist3_n1000.json",
    "nq": "nq_gold1_dist4_n1000.json",
}
RUNS_DIR = PROJECT_ROOT / "artifacts" / "step08" / "runs"
N_BOOT, SEED = 2000, 0


def latest_run(system, manifest_path):
    sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    runs = [d for d in sorted(RUNS_DIR.glob(f"{system}-*")) if (d / "run.json").exists()
            and json.loads((d / "run.json").read_text())["manifest"]["sha256"] == sha]
    if not runs:
        raise FileNotFoundError(f"no completed {system} run for {manifest_path.name}")
    return runs[-1]


def bootstrap(values_by_system, n_boot=N_BOOT, seed=SEED):
    """Paired bootstrap over questions -> 95% CI per system and for compressed - raw."""
    rng = random.Random(seed)
    n = len(values_by_system["raw"])
    stats = {"raw": [], "pisco": [], "diff": []}
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        r = sum(values_by_system["raw"][i] for i in idx) / n
        p = sum(values_by_system["pisco"][i] for i in idx) / n
        stats["raw"].append(r)
        stats["pisco"].append(p)
        stats["diff"].append(p - r)
    lo, hi = int(0.025 * n_boot), int(0.975 * n_boot) - 1
    return {k: [round(sorted(v)[lo], 4), round(sorted(v)[hi], 4)] for k, v in stats.items()}


def group_rates(manifest, rows_by_system, key):
    groups = {}
    for i, s in enumerate(manifest["samples"]):
        g = str(s["annotation"].get(key))
        entry = groups.setdefault(g, {"n": 0, "raw": 0, "pisco": 0})
        entry["n"] += 1
        for system in ("raw", "pisco"):
            entry[system] += rows_by_system[system][i]["match"]
    for entry in groups.values():
        entry["raw"], entry["pisco"] = entry["raw"] / entry["n"], entry["pisco"] / entry["n"]
        entry["diff"] = entry["pisco"] - entry["raw"]
    return groups


def main():
    report = {}
    for name, manifest_file in DATASETS.items():
        manifest_path = PROJECT_ROOT / "data" / "manifests" / manifest_file
        if not manifest_path.exists():
            print(f"{name}: manifest missing, skipped")
            continue
        manifest = json.loads(manifest_path.read_text())
        rows, preds, run_ids = {}, {}, {}
        for system in ("raw", "pisco"):
            run_dir = latest_run(system, manifest_path)
            run_ids[system] = run_dir.name
            preds[system] = [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text().splitlines()]
            rows[system] = score_run(manifest, preds[system])["rows"]

        metrics = {}
        for metric in ("match", "em", "f1"):
            values = {s: [float(r[metric]) for r in rows[s]] for s in rows}
            metrics[metric] = {"raw": statistics.fmean(values["raw"]), "pisco": statistics.fmean(values["pisco"]),
                               "ci95": bootstrap(values)}
            metrics[metric]["diff"] = metrics[metric]["pisco"] - metrics[metric]["raw"]

        r_m, p_m = [r["match"] for r in rows["raw"]], [r["match"] for r in rows["pisco"]]
        ann_key = "all_evidence_preserved" if "all_evidence_preserved" in manifest["samples"][0]["annotation"] \
            else "gold_not_truncated"
        report[name] = {
            "manifest": str(manifest_path.relative_to(PROJECT_ROOT)), "runs": run_ids,
            "n": len(manifest["samples"]),
            "failed": {s: sum(r["status"] != "ok" for r in rows[s]) for s in rows},
            "metrics": metrics,
            "match_contingency": {"both": sum(a and b for a, b in zip(r_m, p_m)),
                                  "raw_only": sum(a and not b for a, b in zip(r_m, p_m)),
                                  "pisco_only": sum(b and not a for a, b in zip(r_m, p_m)),
                                  "neither": sum(not a and not b for a, b in zip(r_m, p_m))},
            "match_retention": sum(a and b for a, b in zip(r_m, p_m)) / max(1, sum(r_m)),
            "by_question_type": group_rates(manifest, rows, "question_type"),
            f"by_{ann_key}": group_rates(manifest, rows, ann_key),
            "by_answer_is_yes_no": group_rates(manifest, rows, "answer_is_yes_no"),
            "median_output_tokens": {s: statistics.median(len(p["output_ids"]) for p in preds[s]) for s in preds},
            "median_decoder_input_tokens": {s: statistics.median(p["decoder_input_length"] for p in preds[s]) for s in preds},
        }
        m = metrics["match"]
        print(f"{name:9s} Match raw {m['raw']:.3f} pisco {m['pisco']:.3f} diff {m['diff']:+.3f} CI {m['ci95']['diff']} | "
              f"F1 raw {metrics['f1']['raw']:.3f} pisco {metrics['f1']['pisco']:.3f} | EM raw {metrics['em']['raw']:.3f} "
              f"pisco {metrics['em']['pisco']:.3f} | retention {report[name]['match_retention']:.3f}")

    out = PROJECT_ROOT / "artifacts" / "step09c" / "raw_vs_compressed.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("written:", out.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
