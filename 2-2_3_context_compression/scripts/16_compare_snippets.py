"""Step 16 comparison: PISCO (5 memories) vs + query-selected excerpt (compressed / raw) vs + oracle excerpt
(compressed / raw) vs Raw, same questions, paired bootstrap CIs vs PISCO.

    ~/anaconda3/envs/pisco-ref/bin/python scripts/16_compare_snippets.py

Also: MultiHop-RAG free-form commitment decomposition and the MuSiQue "PISCO got both hops" subset.
Output: artifacts/step16/compare.json
"""

import hashlib
import json
import random
import re
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pisco_research.scoring import match, normalize_answer, yes_no_match  # noqa: E402

RUNS_DIR = PROJECT_ROOT / "artifacts" / "step08" / "runs"
BASES = {"musique_2hop": ("musique_2hop_gold2_dist3_n1000", "match"),
         "multihoprag_forced": ("multihoprag_2ev_yesno_forced_n882", "yn_match"),
         "multihoprag_2ev": ("multihoprag_2ev_gold2_dist3_n1061", "yn_match"),
         "nq_gold": ("nq_gold1_dist4_n1000", "match")}
CONDITIONS = [("raw", "raw", ""), ("pisco", "pisco", ""),
              ("snip_compressed", "pisco", "_snip"), ("oracle_compressed", "pisco", "_oracle"),
              ("snip_raw", "hybrid", "_snip"), ("oracle_raw", "hybrid", "_oracle"),
              # Step 17: dedicated excerpt-block generators (separate LoRA), text-only vs text + memories
              ("gen_text_snip", "fusion_text", "_snip"), ("gen_textmem_snip", "fusion_textmem", "_snip"),
              ("gen_text_oracle", "fusion_text", "_oracle"), ("gen_textmem_oracle", "fusion_textmem", "_oracle"),
              # Step 18: residual generators (memory-weighted reconstruction loss)
              ("res_text_snip", "fusion_res_text", "_snip"), ("res_textmem_snip", "fusion_res_textmem", "_snip"),
              ("res_text_oracle", "fusion_res_text", "_oracle"), ("res_textmem_oracle", "fusion_res_textmem", "_oracle")]
N_BOOT, SEED = 2000, 0


def latest_run(prefix, manifest_path):
    sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    runs = [d for d in sorted(RUNS_DIR.glob(f"{prefix}-*")) if (d / "run.json").exists()
            and json.loads((d / "run.json").read_text())["manifest"]["sha256"] == sha]
    return runs[-1] if runs else None


def load_rows(prefix, manifest_path, scorer):
    run_dir = latest_run(prefix, manifest_path)
    if run_dir is None:
        return None, None
    manifest = json.loads(manifest_path.read_text())
    preds = {(r := json.loads(line))["sample_id"]: r for line in (run_dir / "predictions.jsonl").read_text().splitlines()}
    if any(preds.get(s["sample_id"], {}).get("status") != "ok" for s in manifest["samples"]):
        return None, None
    rows = {s["sample_id"]: {"ok": float(scorer(preds[s["sample_id"]]["prediction"], s["evaluation"]["reference_answers"])),
                             "pred": preds[s["sample_id"]]["prediction"], "n_tokens": len(preds[s["sample_id"]]["output_ids"])}
            for s in manifest["samples"]}
    return rows, run_dir.name


def paired_ci(a, b):
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
    hop_subset = None
    for name, (base, metric) in BASES.items():
        scorer = yes_no_match if metric == "yn_match" else match
        base_path = PROJECT_ROOT / "data" / "manifests" / f"{base}.json"
        ids = [s["sample_id"] for s in json.loads(base_path.read_text())["samples"]]
        yes_no_ids = {s["sample_id"] for s in json.loads(base_path.read_text())["samples"] if s["annotation"].get("answer_is_yes_no")}
        entry = {"metric": metric, "n": len(ids), "runs": {}, "accuracy": {}, "diff_vs_pisco": {}, "commitment": {}}
        rows = {}
        for cond, system, suffix in CONDITIONS:
            path = PROJECT_ROOT / "data" / "manifests" / f"{base}{suffix}.json"
            if not path.exists():
                continue
            r, run_id = load_rows(system, path, scorer)
            if r is None:
                continue
            rows[cond] = r
            entry["runs"][cond] = run_id
            entry["accuracy"][cond] = round(statistics.fmean(r[i]["ok"] for i in ids), 4)
            if metric == "yn_match" and yes_no_ids:
                yn = [i for i in ids if i in yes_no_ids]
                verd = [re.search(r"\b(yes|no)\b", normalize_answer(r[i]["pred"])) for i in yn]
                commit = [v is not None for v in verd]
                entry["commitment"][cond] = {"commit_rate": round(sum(commit) / len(yn), 4),
                                             "acc_when_committed": round(sum(r[i]["ok"] for i, c in zip(yn, commit) if c) / max(1, sum(commit)), 4),
                                             "median_tokens": statistics.median(r[i]["n_tokens"] for i in yn)}
        if "pisco" in rows:
            p = [rows["pisco"][i]["ok"] for i in ids]
            for cond, r in rows.items():
                if cond == "pisco":
                    continue
                v = [r[i]["ok"] for i in ids]
                entry["diff_vs_pisco"][cond] = {"diff": round(statistics.fmean(v) - statistics.fmean(p), 4), "ci95": paired_ci(p, v)}
        entry["residual_vs_repetition"] = {}
        for sfx in ("snip", "oracle"):
            for res_c, rep_c in ((f"res_textmem_{sfx}", f"gen_text_{sfx}"), (f"res_text_{sfx}", f"gen_text_{sfx}"),
                                 (f"res_textmem_{sfx}", f"res_text_{sfx}")):
                a, b = rows.get(rep_c), rows.get(res_c)
                if a and b:
                    va, vb = [a[i]["ok"] for i in ids], [b[i]["ok"] for i in ids]
                    entry["residual_vs_repetition"][f"{res_c} - {rep_c}"] = {"diff": round(statistics.fmean(vb) - statistics.fmean(va), 4),
                                                                              "ci95": paired_ci(va, vb)}
        entry["textmem_vs_text"] = {}
        for sfx in ("snip", "oracle"):
            a, b = rows.get(f"gen_text_{sfx}"), rows.get(f"gen_textmem_{sfx}")
            if a and b:
                va, vb = [a[i]["ok"] for i in ids], [b[i]["ok"] for i in ids]
                entry["textmem_vs_text"][sfx] = {"diff": round(statistics.fmean(vb) - statistics.fmean(va), 4), "ci95": paired_ci(va, vb)}
        if name == "musique_2hop":
            h1 = {s["sample_id"]: s for s in json.loads((PROJECT_ROOT / "data" / "manifests" / "musique_2hop_hop1_n1000.json").read_text())["samples"]}
            h2 = {s["sample_id"]: s for s in json.loads((PROJECT_ROOT / "data" / "manifests" / "musique_2hop_hop2_n1000.json").read_text())["samples"]}
            r1, _ = load_rows("pisco", PROJECT_ROOT / "data" / "manifests" / "musique_2hop_hop1_n1000.json", match)
            r2, _ = load_rows("pisco", PROJECT_ROOT / "data" / "manifests" / "musique_2hop_hop2_n1000.json", match)
            hop_subset = [i for i in ids if r1[i]["ok"] and r2[i]["ok"]]
            entry["pisco_both_hops_correct_subset"] = {"n": len(hop_subset),
                                                       **{c: round(statistics.fmean(r[i]["ok"] for i in hop_subset), 4) for c, r in rows.items()}}
        report[name] = entry
        print(f"{name:20s} [{metric}] " + " ".join(f"{c} {a:.3f}" for c, a in entry["accuracy"].items()))
        print("   vs pisco: " + " ".join(f"{c} {d['diff']*100:+.1f} {d['ci95']}" for c, d in entry["diff_vs_pisco"].items() if c != "raw"))
        for k_, v in entry["residual_vs_repetition"].items():
            print(f"   {k_}: {v['diff']*100:+.1f} {v['ci95']}")
        if entry["textmem_vs_text"]:
            print("   textmem - text: " + " ".join(f"{k} {v['diff']*100:+.1f} {v['ci95']}" for k, v in entry["textmem_vs_text"].items()))
        if entry["commitment"]:
            print("   commitment: " + " ".join(f"{c} {v['commit_rate']:.3f}/{v['acc_when_committed']:.3f}" for c, v in entry["commitment"].items()))
        if "pisco_both_hops_correct_subset" in entry:
            print("   both-hops subset:", entry["pisco_both_hops_correct_subset"])
    out = PROJECT_ROOT / "artifacts" / "step16" / "compare.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("written:", out.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
