"""Step 10b analysis: Raw vs PISCO on MultiHop-RAG 2-evidence questions, with the
HotpotQA / 2Wiki comparison questions of Step 09c re-scored as auxiliary benchmark.

    ~/anaconda3/envs/pisco-ref/bin/python scripts/10b_analyze_multihoprag.py

Metrics
  match      paper metric (normalized reference is a substring of the prediction)
  yn_match   yes/no-aware Match (scoring.yes_no_match): for yes/no references the
             first whole-word yes/no of the prediction must equal the reference;
             other references use plain Match. This is the primary metric here
             because 83% of the references are yes/no.

Output: artifacts/step10b/multihoprag_analysis.json
"""

import hashlib
import json
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pisco_research.scoring import match, normalize_answer, yes_no_match  # noqa: E402

RUNS_DIR = PROJECT_ROOT / "artifacts" / "step08" / "runs"
PRIMARY = PROJECT_ROOT / "data" / "manifests" / "multihoprag_2ev_gold2_dist3_n1061.json"
AUXILIARY = {"hotpotqa": PROJECT_ROOT / "data" / "manifests" / "hotpotqa_2hop_gold2_dist3_n1000.json",
             "2wiki": PROJECT_ROOT / "data" / "manifests" / "2wiki_2hop_gold2_dist3_n1000.json"}
SYSTEMS = ("raw", "pisco")
N_BOOT, SEED = 2000, 0
ABSTAIN = re.compile(r"(insufficient|not provide|does not (mention|specify|state|contain|say)|no information|"
                     r"cannot be determined|not possible to determine|not mentioned|unable to determine|"
                     r"do not provide|don't provide|no mention)")


def latest_run(system, manifest_path):
    sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    runs = [d for d in sorted(RUNS_DIR.glob(f"{system}-*")) if (d / "run.json").exists()
            and json.loads((d / "run.json").read_text())["manifest"]["sha256"] == sha]
    if not runs:
        raise FileNotFoundError(f"no completed {system} run for {manifest_path.name}")
    return runs[-1]


def load(manifest_path):
    manifest = json.loads(manifest_path.read_text())
    preds, runs = {}, {}
    for system in SYSTEMS:
        run_dir = latest_run(system, manifest_path)
        runs[system] = run_dir.name
        rows = [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text().splitlines()]
        preds[system] = {r["sample_id"]: r for r in rows}
    samples = []
    for s in manifest["samples"]:
        refs = s["evaluation"]["reference_answers"]
        row = {"sample_id": s["sample_id"], "refs": refs, "ann": s["annotation"], "question": s["model_input"]["question"]}
        for system in SYSTEMS:
            p = preds[system][s["sample_id"]]
            if p["status"] != "ok":
                raise RuntimeError(f"{runs[system]} {s['sample_id']}: {p['status']}")
            text = p["prediction"]
            found = re.search(r"\b(yes|no)\b", normalize_answer(text))
            row[system] = {"pred": text, "match": match(text, refs), "yn_match": yes_no_match(text, refs),
                           "first_yes_no": found.group(1) if found else None,
                           "abstain": bool(ABSTAIN.search(text.lower())), "n_tokens": len(p["output_ids"]),
                           "hit_cap": not p.get("stopped_at_eos", True)}
        samples.append(row)
    return manifest, runs, samples


def rate(samples, system, metric, idx=None):
    idx = range(len(samples)) if idx is None else list(idx)
    return sum(samples[i][system][metric] for i in idx) / max(1, len(idx))


def paired(samples, metric, idx=None):
    idx = list(range(len(samples)) if idx is None else idx)
    n = len(idx)
    r = [samples[i]["raw"][metric] for i in idx]
    p = [samples[i]["pisco"][metric] for i in idx]
    rng = random.Random(SEED)
    diffs = []
    for _ in range(N_BOOT):
        b = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(p[i] for i in b) / n - sum(r[i] for i in b) / n)
    diffs.sort()
    lo, hi = int(0.025 * N_BOOT), int(0.975 * N_BOOT) - 1
    return {"n": n, "raw": sum(r) / n, "pisco": sum(p) / n, "diff": sum(p) / n - sum(r) / n,
            "diff_ci95": [round(diffs[lo], 4), round(diffs[hi], 4)],
            "retention": sum(a and b for a, b in zip(r, p)) / max(1, sum(r)),
            "both": sum(a and b for a, b in zip(r, p)), "raw_only": sum(a and not b for a, b in zip(r, p)),
            "pisco_only": sum(b and not a for a, b in zip(r, p)), "neither": sum(not a and not b for a, b in zip(r, p))}


def by_group(samples, key_fn, metrics=("yn_match", "match")):
    groups = {}
    for i, s in enumerate(samples):
        groups.setdefault(str(key_fn(s)), []).append(i)
    return {g: {m: paired(samples, m, idx) for m in metrics} for g, idx in sorted(groups.items())}


def yes_no_behaviour(samples, idx):
    out = {}
    for system in SYSTEMS:
        c = Counter(samples[i][system]["first_yes_no"] for i in idx)
        out[system] = {"predicted_yes": c["yes"], "predicted_no": c["no"], "no_yes_no_word": c[None],
                       "abstain_phrase": sum(samples[i][system]["abstain"] for i in idx),
                       "median_output_tokens": statistics.median(samples[i][system]["n_tokens"] for i in idx)}
    ref = Counter(normalize_answer(samples[i]["refs"][0]) for i in idx)
    out["reference_yes"], out["reference_no"] = ref["yes"], ref["no"]
    # accuracy conditional on the reference (reveals yes-bias)
    for lab in ("yes", "no"):
        sel = [i for i in idx if normalize_answer(samples[i]["refs"][0]) == lab]
        out[f"given_ref_{lab}"] = {s: round(rate(samples, s, "yn_match", sel), 4) for s in SYSTEMS} | {"n": len(sel)}
    return out


def main():
    manifest, runs, samples = load(PRIMARY)
    fixture = json.loads((PROJECT_ROOT / "data" / "multihoprag" / "multihoprag_2ev_gold2_dist3_fixture.json").read_text())
    n = len(samples)
    report = {"primary": {"manifest": PRIMARY.name, "runs": runs, "n": n, "metrics": {}}}
    P = report["primary"]
    P["metrics"] = {m: paired(samples, m) for m in ("yn_match", "match")}
    P["by_question_type"] = by_group(samples, lambda s: s["ann"]["question_type"])
    P["by_answer_kind"] = by_group(samples, lambda s: s["ann"]["answer_kind"])
    P["by_type_and_kind"] = by_group(samples, lambda s: f"{s['ann']['question_type']}/{s['ann']['answer_kind']}")
    P["by_same_source"] = by_group(samples, lambda s: f"same_source={s['ann']['same_source']}")
    P["by_gold_adjacent"] = by_group(samples, lambda s: f"gold_adjacent={abs(s['ann']['gold_positions'][0] - s['ann']['gold_positions'][1]) == 1}")

    def gap_bucket(d):
        return "0-7d" if d <= 7 else "8-30d" if d <= 30 else ">30d"
    temporal = [i for i, s in enumerate(samples) if s["ann"]["question_type"] == "temporal"]
    groups = {}
    for i in temporal:
        groups.setdefault(gap_bucket(samples[i]["ann"]["date_gap_days"]), []).append(i)
    P["temporal_by_date_gap"] = {g: paired(samples, "yn_match", idx) for g, idx in sorted(groups.items())}

    yn_idx = [i for i, s in enumerate(samples) if s["ann"]["answer_kind"] == "yes_no"]
    P["yes_no_behaviour"] = {"all": yes_no_behaviour(samples, yn_idx)}
    for t in ("comparison", "temporal"):
        P["yes_no_behaviour"][t] = yes_no_behaviour(samples, [i for i in yn_idx if samples[i]["ann"]["question_type"] == t])
    # --- decomposition of the yes/no gap: not committing vs wrong verdict ---------------------
    committed = {s: [i for i in yn_idx if samples[i][s]["first_yes_no"] is not None] for s in SYSTEMS}
    both_committed = [i for i in yn_idx if all(samples[i][s]["first_yes_no"] for s in SYSTEMS)]
    P["yes_no_commitment"] = {
        "n_yes_no_questions": len(yn_idx),
        "commit_rate": {s: round(len(committed[s]) / len(yn_idx), 4) for s in SYSTEMS},
        "accuracy_when_committed": {s: round(rate(samples, s, "yn_match", committed[s]), 4) for s in SYSTEMS},
        "both_committed": paired(samples, "yn_match", both_committed),
        "verdict_agreement_when_both_committed": round(sum(samples[i]["raw"]["first_yes_no"] == samples[i]["pisco"]["first_yes_no"]
                                                           for i in both_committed) / max(1, len(both_committed)), 4),
        # counterfactual: PISCO's commit rate with Raw's accuracy-when-committed
        "gap_from_not_committing": round((len(committed["pisco"]) - len(committed["raw"])) / len(yn_idx)
                                         * rate(samples, "raw", "yn_match", committed["raw"]), 4),
    }
    yn_gap = rate(samples, "pisco", "yn_match", yn_idx) - rate(samples, "raw", "yn_match", yn_idx)
    P["yes_no_commitment"]["gap_total_yes_no"] = round(yn_gap, 4)
    P["yes_no_commitment"]["gap_from_wrong_verdict"] = round(yn_gap - P["yes_no_commitment"]["gap_from_not_committing"], 4)

    # --- what does a non-committing PISCO answer look like? ---------------------------------
    def evidence_coverage(s, system):
        """How many of the two gold facts leave a lexical trace in the prediction.

        Content words (>= 4 letters, not in the question) of each fact; a fact
        counts as used if >= 2 of them (or all, if it has fewer) appear."""
        pred = set(normalize_answer(s[system]["pred"]).split())
        qwords = set(normalize_answer(s["question"]).split())
        used = 0
        for fact in [f for sents in manifest_evidence[s["sample_id"]].values() for f in sents]:
            words = {w for w in normalize_answer(fact).split() if len(w) >= 4 and w not in qwords}
            need = min(2, len(words)) or 1
            used += len(words & pred) >= need
        return used
    manifest_evidence = {s["sample_id"]: s_fx.get("evidence_sentences", {})
                         for s, s_fx in zip(manifest["samples"], fixture["samples"])}
    P["evidence_coverage_yes_no"] = {}
    for system in SYSTEMS:
        cov = Counter(evidence_coverage(samples[i], system) for i in yn_idx)
        P["evidence_coverage_yes_no"][system] = {"both_facts": cov[2], "one_fact": cov[1], "no_fact": cov[0]}
    non_commit = [i for i in yn_idx if samples[i]["pisco"]["first_yes_no"] is None]
    cats = Counter()
    for i in non_commit:
        s = samples[i]
        if s["pisco"]["abstain"]:
            cats["abstain_phrase"] += 1
        else:
            cats[f"covers_{evidence_coverage(s, 'pisco')}_facts_no_verdict"] += 1
    P["pisco_non_commit_categories"] = dict(cats)
    # generation cap (128 new tokens): a verdict placed after a long explanation is cut off
    P["generation_cap_yes_no"] = {}
    for system in SYSTEMS:
        cap = [i for i in yn_idx if samples[i][system]["hit_cap"]]
        nc = [i for i in yn_idx if samples[i][system]["first_yes_no"] is None]
        P["generation_cap_yes_no"][system] = {
            "hit_cap": len(cap), "hit_cap_rate": round(len(cap) / len(yn_idx), 4),
            "non_commit_and_hit_cap": sum(samples[i][system]["hit_cap"] for i in nc),
            "non_commit_total": len(nc),
            "commit_rate_when_not_capped": round(sum(samples[i][system]["first_yes_no"] is not None for i in yn_idx
                                                     if not samples[i][system]["hit_cap"]) / max(1, len(yn_idx) - len(cap)), 4)}
    P["examples_pisco_non_commit"] = [{"sample_id": samples[i]["sample_id"], "question": samples[i]["question"],
                                       "reference": samples[i]["refs"][0], "pisco": samples[i]["pisco"]["pred"][:300],
                                       "raw": samples[i]["raw"]["pred"][:200]} for i in non_commit[:8]]

    P["abstention"] = {s: {"n": sum(x[s]["abstain"] for x in samples),
                           "yn_match_when_abstaining": round(rate(samples, s, "yn_match",
                                                                  [i for i, x in enumerate(samples) if x[s]["abstain"]]), 4)}
                       for s in SYSTEMS}
    # the "other" references ("consistent", "agree", ...) are scored by substring Match only
    P["note_other_answers"] = dict(Counter(normalize_answer(s["refs"][0]) for s in samples if s["ann"]["answer_kind"] == "other"))

    # examples: raw correct, pisco wrong, by type
    P["examples_raw_only"] = {}
    for t in ("inference", "comparison", "temporal"):
        ex = [s for s in samples if s["ann"]["question_type"] == t and s["raw"]["yn_match"] and not s["pisco"]["yn_match"]]
        P["examples_raw_only"][t] = [{"sample_id": s["sample_id"], "question": s["question"], "reference": s["refs"][0],
                                      "raw": s["raw"]["pred"][:300], "pisco": s["pisco"]["pred"][:300]} for s in ex[:6]]

    # auxiliary: HotpotQA / 2Wiki comparison questions of Step 09c, both metrics
    report["auxiliary_comparison_09c"] = {}
    for name, path in AUXILIARY.items():
        _, aruns, asamples = load(path)
        comp = [s for s in asamples if s["ann"]["question_type"] == "comparison"]
        yn = [s for s in comp if s["ann"]["answer_is_yes_no"]]
        ent = [s for s in comp if not s["ann"]["answer_is_yes_no"]]
        report["auxiliary_comparison_09c"][name] = {
            "runs": aruns,
            "comparison_yes_no": {m: paired(yn, m) for m in ("yn_match", "match")},
            "comparison_entity": {m: paired(ent, m) for m in ("yn_match", "match")},
            "yes_no_behaviour": yes_no_behaviour(yn, list(range(len(yn)))),
        }

    # forced-choice control (same documents, question + " Answer yes or no first."), if its runs exist
    FORCED = PROJECT_ROOT / "data" / "manifests" / "multihoprag_2ev_yesno_forced_n882.json"
    try:
        _, fruns, fsamples = load(FORCED)
    except FileNotFoundError as exc:
        report["forced_choice_control"] = {"status": f"not run: {exc}"}
    else:
        free = {s["sample_id"]: s for s in samples}
        fidx = list(range(len(fsamples)))
        F = {"manifest": FORCED.name, "runs": fruns, "n": len(fsamples),
             "commit_rate": {s: round(sum(x[s]["first_yes_no"] is not None for x in fsamples) / len(fsamples), 4) for s in SYSTEMS},
             "yn_match": paired(fsamples, "yn_match"),
             "by_question_type": {g: d["yn_match"] for g, d in by_group(fsamples, lambda s: s["ann"]["question_type"]).items()},
             "yes_no_behaviour": yes_no_behaviour(fsamples, fidx),
             "verdict_agreement": round(sum(x["raw"]["first_yes_no"] == x["pisco"]["first_yes_no"] for x in fsamples) / len(fsamples), 4),
             "free_form_same_questions": {s: round(sum(free[x["sample_id"]][s]["yn_match"] for x in fsamples) / len(fsamples), 4)
                                          for s in SYSTEMS},
             "median_output_tokens": {s: statistics.median(x[s]["n_tokens"] for x in fsamples) for s in SYSTEMS}}
        report["forced_choice_control"] = F

    out = PROJECT_ROOT / "artifacts" / "step10b" / "multihoprag_analysis.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    def line(label, d):
        print(f"{label:34s} n={d['n']:4d} raw {d['raw']:.3f} pisco {d['pisco']:.3f} diff {d['diff']*100:+.1f}%p "
              f"CI [{d['diff_ci95'][0]*100:+.1f}, {d['diff_ci95'][1]*100:+.1f}] ret {d['retention']:.2f}")
    for m in ("yn_match", "match"):
        line(f"ALL [{m}]", P["metrics"][m])
    for g, d in P["by_type_and_kind"].items():
        line(f"{g} [yn]", d["yn_match"])
    for g, d in P["by_same_source"].items():
        line(f"{g} [yn]", d["yn_match"])
    for g, d in P["by_gold_adjacent"].items():
        line(f"{g} [yn]", d["yn_match"])
    for g, d in P["temporal_by_date_gap"].items():
        line(f"temporal gap {g} [yn]", d)
    print("yes/no behaviour:", json.dumps(P["yes_no_behaviour"]["all"]))
    print("commitment:", json.dumps(P["yes_no_commitment"]))
    print("evidence coverage (yes/no):", P["evidence_coverage_yes_no"])
    print("pisco non-commit categories:", P["pisco_non_commit_categories"])
    print("generation cap (yes/no):", json.dumps(P["generation_cap_yes_no"]))
    print("abstention:", P["abstention"])
    F = report["forced_choice_control"]
    if "yn_match" in F:
        line("FORCED all [yn]", F["yn_match"])
        for g, d in F["by_question_type"].items():
            line(f"FORCED {g} [yn]", d)
        print("FORCED commit:", F["commit_rate"], "agreement:", F["verdict_agreement"],
              "free-form same Qs:", F["free_form_same_questions"], "tokens:", F["median_output_tokens"])
        print("FORCED behaviour:", json.dumps(F["yes_no_behaviour"]))
    else:
        print("forced-choice control:", F["status"])
    for name, a in report["auxiliary_comparison_09c"].items():
        line(f"aux {name} comparison yes/no [yn]", a["comparison_yes_no"]["yn_match"])
        line(f"aux {name} comparison yes/no [match]", a["comparison_yes_no"]["match"])
        line(f"aux {name} comparison entity [match]", a["comparison_entity"]["match"])
    print("written:", out.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
