"""Step 19 comparison: full-document block generators (repetition control vs residual, with/without memory input)
against Raw, PISCO and the Step 17 excerpt generator; paired CIs.

    ~/anaconda3/envs/pisco-ref/bin/python scripts/19_compare_full.py

Output: artifacts/step19/compare.json
"""

import importlib.util
import json
import re
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pisco_research.scoring import match, normalize_answer, yes_no_match  # noqa: E402

spec = importlib.util.spec_from_file_location("cmp16", PROJECT_ROOT / "scripts" / "16_compare_snippets.py")
cmp16 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cmp16)

BASES = cmp16.BASES
CONDITIONS = [("raw", "raw", ""), ("pisco", "pisco", ""),
              ("gen_text_snip", "fusion_text", "_snip"), ("gen_text_oracle", "fusion_text", "_oracle"),
              ("rep_full", "fusion_rep_full", "_full"), ("res_full_text", "fusion_res_full_text", "_full"),
              ("res_full_textmem", "fusion_res_full_textmem", "_full")]
PAIRS = [("res_full_textmem", "rep_full"), ("res_full_text", "rep_full"), ("res_full_textmem", "res_full_text"),
         ("rep_full", "gen_text_snip"), ("res_full_textmem", "gen_text_snip")]


def main():
    report = {}
    for name, (base, metric) in BASES.items():
        scorer = yes_no_match if metric == "yn_match" else match
        base_path = PROJECT_ROOT / "data" / "manifests" / f"{base}.json"
        base_samples = json.loads(base_path.read_text())["samples"]
        ids = [s["sample_id"] for s in base_samples]
        yes_no_ids = {s["sample_id"] for s in base_samples if s["annotation"].get("answer_is_yes_no")}
        rows, entry = {}, {"metric": metric, "n": len(ids), "runs": {}, "accuracy": {}, "diff_vs_pisco": {}, "pairs": {}, "commitment": {}}
        for cond, prefix, suffix in CONDITIONS:
            path = PROJECT_ROOT / "data" / "manifests" / f"{base}{suffix}.json"
            if not path.exists():
                continue
            r, run_id = cmp16.load_rows(prefix, path, scorer)
            if r is None:
                continue
            rows[cond], entry["runs"][cond] = r, run_id
            entry["accuracy"][cond] = round(statistics.fmean(r[i]["ok"] for i in ids), 4)
            if metric == "yn_match":
                yn = [i for i in ids if i in yes_no_ids]
                verd = [re.search(r"\b(yes|no)\b", normalize_answer(r[i]["pred"])) for i in yn]
                commit = [v is not None for v in verd]
                entry["commitment"][cond] = {"commit_rate": round(sum(commit) / len(yn), 4),
                                             "acc_when_committed": round(sum(r[i]["ok"] for i, c in zip(yn, commit) if c) / max(1, sum(commit)), 4)}
        if "pisco" in rows:
            p = [rows["pisco"][i]["ok"] for i in ids]
            for cond, r in rows.items():
                if cond != "pisco":
                    v = [r[i]["ok"] for i in ids]
                    entry["diff_vs_pisco"][cond] = {"diff": round(statistics.fmean(v) - statistics.fmean(p), 4), "ci95": cmp16.paired_ci(p, v)}
        for a, b in PAIRS:
            if a in rows and b in rows:
                va, vb = [rows[b][i]["ok"] for i in ids], [rows[a][i]["ok"] for i in ids]
                entry["pairs"][f"{a} - {b}"] = {"diff": round(statistics.fmean(vb) - statistics.fmean(va), 4), "ci95": cmp16.paired_ci(va, vb)}
        if name == "musique_2hop":
            r1, _ = cmp16.load_rows("pisco", PROJECT_ROOT / "data" / "manifests" / "musique_2hop_hop1_n1000.json", match)
            r2, _ = cmp16.load_rows("pisco", PROJECT_ROOT / "data" / "manifests" / "musique_2hop_hop2_n1000.json", match)
            sub = [i for i in ids if r1[i]["ok"] and r2[i]["ok"]]
            entry["pisco_both_hops_correct_subset"] = {"n": len(sub), **{c: round(statistics.fmean(r[i]["ok"] for i in sub), 4) for c, r in rows.items()}}
        report[name] = entry
        print(f"{name:20s} [{metric}] " + " ".join(f"{c} {a:.3f}" for c, a in entry["accuracy"].items()))
        print("   vs pisco: " + " ".join(f"{c} {d['diff']*100:+.1f} {d['ci95']}" for c, d in entry["diff_vs_pisco"].items() if c != "raw"))
        for k, v in entry["pairs"].items():
            print(f"   {k}: {v['diff']*100:+.1f} {v['ci95']}")
        if entry["commitment"]:
            print("   commitment: " + " ".join(f"{c} {v['commit_rate']:.3f}/{v['acc_when_committed']:.3f}" for c, v in entry["commitment"].items()))
        if "pisco_both_hops_correct_subset" in entry:
            print("   both-hops subset:", entry["pisco_both_hops_correct_subset"])
    out = PROJECT_ROOT / "artifacts" / "step19" / "compare.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("written:", out.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
