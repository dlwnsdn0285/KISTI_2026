"""Step 10a analysis: hop1 / hop2 / composed, Raw vs PISCO, on identical MuSiQue contexts.

    ~/anaconda3/envs/pisco-ref/bin/python scripts/10a_analyze_musique_hops.py

composed = Step 09c runs on musique_2hop_gold2_dist3_n1000 (same documents,
original question). hop1 / hop2 = Step 10a runs. All three share sample_ids.

Pre-registered reading:
  * hop1 retention as low as composed  -> bridge entity lost in compression
  * hop1, hop2 retention high, composed low -> composition (linking) failure
  * direct test: among questions where a system answers BOTH hops alone,
    how often does it answer the composed question? (Raw vs PISCO)

Output: artifacts/step10a/musique_hops.json
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

MANIFESTS = {"hop1": "musique_2hop_hop1_n1000.json", "hop2": "musique_2hop_hop2_n1000.json",
             "composed": "musique_2hop_gold2_dist3_n1000.json"}
SYSTEMS = ("raw", "pisco")
RUNS_DIR = PROJECT_ROOT / "artifacts" / "step08" / "runs"
N_BOOT, SEED = 2000, 0


def latest_run(system, manifest_path):
    sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    runs = [d for d in sorted(RUNS_DIR.glob(f"{system}-*")) if (d / "run.json").exists()
            and json.loads((d / "run.json").read_text())["manifest"]["sha256"] == sha]
    if not runs:
        raise FileNotFoundError(f"no completed {system} run for {manifest_path.name}")
    return runs[-1]


def ci(values_fn, n, n_boot=N_BOOT, seed=SEED):
    """Bootstrap over questions (same resample for every quantity in values_fn)."""
    rng = random.Random(seed)
    out = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        out.append(values_fn(idx))
    lo, hi = int(0.025 * n_boot), int(0.975 * n_boot) - 1
    return [round(sorted(out)[lo], 4), round(sorted(out)[hi], 4)]


def rate(xs, idx=None):
    idx = range(len(xs)) if idx is None else idx
    idx = list(idx)
    return sum(xs[i] for i in idx) / max(1, len(idx))


def main():
    manifests, M = {}, {}           # M[probe][system] -> list[bool] in sample order
    runs = {}
    for probe, f in MANIFESTS.items():
        path = PROJECT_ROOT / "data" / "manifests" / f
        manifests[probe] = json.loads(path.read_text())
        M[probe], runs[probe] = {}, {}
        for system in SYSTEMS:
            run_dir = latest_run(system, path)
            runs[probe][system] = run_dir.name
            preds = [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text().splitlines()]
            rows = score_run(manifests[probe], preds)["rows"]
            if any(r["status"] != "ok" for r in rows):
                raise RuntimeError(f"{run_dir.name}: failed/missing predictions")
            M[probe][system] = [bool(r["match"]) for r in rows]
    ids = [s["sample_id"] for s in manifests["composed"]["samples"]]
    for probe in ("hop1", "hop2"):
        if [s["sample_id"] for s in manifests[probe]["samples"]] != ids:
            raise RuntimeError("sample order differs between manifests")
    ann = [s["annotation"] for s in manifests["hop1"]["samples"]]
    n = len(ids)

    # 1) accuracy per probe, paired diff and retention
    per_probe = {}
    for probe in MANIFESTS:
        r, p = M[probe]["raw"], M[probe]["pisco"]
        per_probe[probe] = {
            "raw": rate(r), "pisco": rate(p), "diff": rate(p) - rate(r),
            "diff_ci95": ci(lambda idx: rate(p, idx) - rate(r, idx), n),
            "retention": sum(a and b for a, b in zip(r, p)) / max(1, sum(r)),
            "contingency": {"both": sum(a and b for a, b in zip(r, p)), "raw_only": sum(a and not b for a, b in zip(r, p)),
                            "pisco_only": sum(b and not a for a, b in zip(r, p)),
                            "neither": sum(not a and not b for a, b in zip(r, p))},
        }

    # 2) composition test: P(composed | both hops correct) per system
    composition = {}
    for system in SYSTEMS:
        both = [M["hop1"][system][i] and M["hop2"][system][i] for i in range(n)]
        sel = [i for i in range(n) if both[i]]
        composition[system] = {"n_both_hops_correct": len(sel),
                               "composed_given_both": rate(M["composed"][system], sel),
                               "composed_given_not_both": rate(M["composed"][system], [i for i in range(n) if not both[i]])}

    def comp_gap(idx):
        vals = {}
        for system in SYSTEMS:
            sel = [i for i in idx if M["hop1"][system][i] and M["hop2"][system][i]]
            vals[system] = rate(M["composed"][system], sel)
        return vals["pisco"] - vals["raw"]
    composition["pisco_minus_raw_ci95"] = ci(comp_gap, n)

    # gap decomposition: composed = P(both) * P(c|both) + P(not both) * P(c|not both).
    # Counterfactual = PISCO's hop accuracy with Raw's conditional composition rates.
    p_both = composition["pisco"]["n_both_hops_correct"] / n
    counterfactual = p_both * composition["raw"]["composed_given_both"] + \
        (1 - p_both) * composition["raw"]["composed_given_not_both"]
    raw_c, pisco_c = rate(M["composed"]["raw"]), rate(M["composed"]["pisco"])
    composition["gap_decomposition"] = {
        "total": pisco_c - raw_c, "counterfactual_composed": counterfactual,
        "from_hop_accuracy": counterfactual - raw_c, "from_composition": pisco_c - counterfactual}
    # same question set for both systems: both hops correct for BOTH systems
    common = [i for i in range(n) if all(M[h][s][i] for h in ("hop1", "hop2") for s in SYSTEMS)]
    common_set = set(common)

    def common_gap(idx):
        sel = [i for i in idx if i in common_set]
        return rate(M["composed"]["pisco"], sel) - rate(M["composed"]["raw"], sel)
    composition["common_both_hops_correct"] = {
        "n": len(common), "raw": rate(M["composed"]["raw"], common), "pisco": rate(M["composed"]["pisco"], common),
        "diff_ci95": ci(common_gap, n)}

    # 3) 8-way pattern table (hop1, hop2, composed) per system
    patterns = {}
    for system in SYSTEMS:
        tab = {}
        for i in range(n):
            key = "".join("1" if M[h][system][i] else "0" for h in ("hop1", "hop2", "composed"))
            tab[key] = tab.get(key, 0) + 1
        patterns[system] = dict(sorted(tab.items()))

    # 4) where does PISCO lose? decompose the composed raw-only losses by PISCO's hop results
    losses = [i for i in range(n) if M["composed"]["raw"][i] and not M["composed"]["pisco"][i]]
    loss_by_hops = {}
    for i in losses:
        key = f"pisco_hop1={int(M['hop1']['pisco'][i])},hop2={int(M['hop2']['pisco'][i])}"
        loss_by_hops[key] = loss_by_hops.get(key, 0) + 1

    # 5) strata: hop kinds, answer survives truncation
    def strata(key_fn):
        groups = {}
        for i in range(n):
            groups.setdefault(str(key_fn(i)), []).append(i)
        out = {}
        for g, sel in sorted(groups.items()):
            out[g] = {"n": len(sel)}
            for probe in MANIFESTS:
                out[g][probe] = {s: round(rate(M[probe][s], sel), 4) for s in SYSTEMS}
        return out
    by_stratum = {
        "hop_kinds": strata(lambda i: ann[i]["hop_kinds"]),
        "hop1_answer_in_used_text": strata(lambda i: ann[i]["hop1_answer_in_used_text"]),
        "hop2_answer_in_used_text": strata(lambda i: ann[i]["hop2_answer_in_used_text"]),
        "both_answers_in_used_text": strata(lambda i: ann[i]["hop1_answer_in_used_text"] and ann[i]["hop2_answer_in_used_text"]),
    }

    # 6) does PISCO's composed answer mention the bridge entity? (raw vs pisco, composed failures)
    comp_preds = {}
    for system in SYSTEMS:
        run_dir = RUNS_DIR / runs["composed"][system]
        comp_preds[system] = {json.loads(line)["sample_id"]: json.loads(line)["prediction"]
                              for line in (run_dir / "predictions.jsonl").read_text().splitlines()}
    from pisco_research.scoring import normalize_answer
    bridge_mention = {}
    for system in SYSTEMS:
        wrong = [i for i in range(n) if not M["composed"][system][i]]
        hit = sum(normalize_answer(ann[i]["sub_answers"][0]) in normalize_answer(comp_preds[system][ids[i]]) for i in wrong)
        bridge_mention[system] = {"n_composed_wrong": len(wrong), "bridge_entity_mentioned": hit}

    # 7) robustness: Match is a substring test, so answers that enumerate every document
    #    ("X (Document 1) ... Y (Document 5)") can match by accident. Count them and
    #    re-score treating enumeration-style answers (>= 3 "Document" mentions) as wrong.
    enumeration = {}
    for probe in MANIFESTS:
        enumeration[probe] = {}
        for system in SYSTEMS:
            run_dir = RUNS_DIR / runs[probe][system]
            preds = {(r := json.loads(line))["sample_id"]: r["prediction"]
                     for line in (run_dir / "predictions.jsonl").read_text().splitlines()}
            enum = [preds[sid].count("Document") >= 3 for sid in ids]
            strict = [m and not e for m, e in zip(M[probe][system], enum)]
            enumeration[probe][system] = {"n_enumeration": sum(enum),
                                          "match_and_enumeration": sum(m and e for m, e in zip(M[probe][system], enum)),
                                          "strict_match": rate(strict)}

    examples = [{"sample_id": ids[i], "question": ann[i]["original_question"], "sub_answers": ann[i]["sub_answers"],
                 "pisco_composed": comp_preds["pisco"][ids[i]], "raw_composed": comp_preds["raw"][ids[i]]}
                for i in losses if M["hop1"]["pisco"][i] and M["hop2"]["pisco"][i]][:15]

    report = {"runs": runs, "n": n, "metric": "Match", "per_probe": per_probe, "composition": composition,
              "patterns_hop1_hop2_composed": patterns, "composed_raw_only_losses": {"n": len(losses), **loss_by_hops},
              "by_stratum": by_stratum, "composed_wrong_mentions_bridge_entity": bridge_mention,
              "enumeration_robustness": enumeration,
              "examples_pisco_both_hops_ok_but_composed_lost": examples}
    out = PROJECT_ROOT / "artifacts" / "step10a" / "musique_hops.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    for probe, v in per_probe.items():
        print(f"{probe:9s} raw {v['raw']:.3f} pisco {v['pisco']:.3f} diff {v['diff']:+.3f} CI {v['diff_ci95']} "
              f"retention {v['retention']:.3f}")
    print("composition:", json.dumps(composition, indent=1))
    print("patterns:", patterns)
    print("composed raw-only losses by PISCO hop results:", len(losses), loss_by_hops)
    print("bridge mention in wrong composed answers:", bridge_mention)
    print("enumeration robustness:", json.dumps(enumeration))
    print("written:", out.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()