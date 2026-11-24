"""Feedback-then-Refine loop over unsupported claim-reference pairs.

Loop 0 : re-verify the original claim with the (open-source) ch.4 verifier.
Loop t : for every still-unsupported pair
           (1) feedback  - why is it unsupported? (condition dependent)
           (2) refine    - rewrite the claim / re-cite / drop the citation
           (3) re-verify - re-retrieve top-5 evidence with the new claim, LLM judge AND NLI
Conditions
  no_fb            : refine with evidence only ("this sentence is not supported, fix it")
  reason_fb        : + the verifier's free-text reasoning as feedback
  struct_fb        : + structured feedback (error type, unsupported span, evidence, suggestion)
  struct_fb_recite : structured feedback that may also choose re-citation (same survey's
                     bibliography) or dropping the citation
"""
import argparse
import asyncio
import copy
import json
import time
from pathlib import Path

from common import (FTR_ROOT, JUDGE_NAME, JUDGE_URL, LLM, NLI, REFINER_NAME, REFINER_URL, Retriever, combine, find_ocr_json, judge_one,
                    load_reference_titles, survey_dir_of)

CONDITIONS = ["no_fb", "reason_fb", "struct_fb", "struct_fb_recite"]
N_RECITE_CANDIDATES = 3

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
FEEDBACK_HEAD = """You are a meticulous reviewer of citations in scientific survey papers.
A citing sentence (the claim) was judged as NOT supported by its cited reference.
Using the evidence paragraphs retrieved from the cited reference and the verifier's verdict,
diagnose precisely why the claim is not supported and how it should be fixed.
The sentence may cite several references: only the content attributed to the TARGET citation marker
is in scope. The suggestion must keep the claim's original topic; never suggest turning the claim
into a statement about a different subject just to match the reference.

Error types:
- unrelated: the cited reference does not discuss the claim's topic at all
- overclaim: the reference discusses the topic but the claim is stronger, more general or more certain than the evidence
- contradiction: the evidence states the opposite of (part of) the claim
- detail_mismatch: specific details (numbers, methods, populations, findings) differ from the evidence
- scope_mismatch: the claim attributes to the reference something outside its scope (e.g., a different domain, task or setting)
"""

ACTION_BLOCK = """
Also choose a corrective action:
- "revise": the cited reference covers the claim's topic at least partially; rewrite the claim to match it
- "recite": the cited reference does not support the claim, but one of the CANDIDATE references
  from the same survey's bibliography clearly discusses the claim's topic; replace the citation with it
{drop_line}If the cited reference does not discuss the claim's topic at all, do not choose "revise":
{fallback}
"""
DROP_LINE = """- "drop": neither the cited reference nor any candidate can support a meaningful version of the claim;
  remove this citation (last resort)
"""

# An explicit JSON template is required: with only a key list, Qwen3-30B-A3B often degenerates
# into repeated keys (observed in a first run; 0/12 valid vs 12/12 with the template).
JSON_TEMPLATE = """
Return strictly one JSON object in exactly this format:
{{
  "error_type": "<one of: unrelated, overclaim, contradiction, detail_mismatch, scope_mismatch>",
  "unsupported_part": "<exact span of the claim lacking support>",
  "supporting_evidence": "<short quote(s) from the evidence related to the claim, or empty>",
  "reason": "<1-3 sentences>",
  "suggestion": "<concrete instruction for revising the claim so that it becomes supported while keeping its role in the survey>"{action_fields}
}}"""
ACTION_FIELDS = (',\n  "action": "<one of: {actions}>",\n'
                 '  "recite_candidate": "<C1, C2 or C3 when action is recite, otherwise null>"')


def feedback_system(recite: bool, allow_drop: bool) -> str:
    s = FEEDBACK_HEAD
    if recite:
        s += ACTION_BLOCK.format(
            drop_line=DROP_LINE if allow_drop else "",
            fallback=('choose "recite" when a candidate clearly discusses it, otherwise "drop".'
                      if allow_drop else 'choose "recite" when a candidate clearly discusses it.'))
    actions = "revise, recite, drop" if allow_drop else "revise, recite"
    return s + JSON_TEMPLATE.format(action_fields=ACTION_FIELDS.format(actions=actions) if recite else "")


VALID_ERROR_TYPES = {"unrelated", "overclaim", "contradiction", "detail_mismatch", "scope_mismatch"}


def valid_feedback(o):
    return o.get("error_type") in VALID_ERROR_TYPES and bool(str(o.get("suggestion", "")).strip())


def valid_refine(o):
    return bool(str(o.get("revised_claim", "")).strip())


DROP_FROM_LOOP = 3  # dropping a citation is allowed only after two revise/re-cite attempts


REFINE_SYSTEM = """You revise one sentence of a scientific survey so that it is faithful to its cited reference.
Rules:
1. Every factual statement in the revised sentence must be supported by the evidence paragraphs.
2. Keep the original sentence's topic and its role in the survey as much as possible; make the minimal
   necessary change (narrow, soften, correct, or remove the unsupported parts).
3. Do not copy the evidence verbatim; write one fluent survey-style sentence in your own words.
4. The sentence may cite several references. Only revise the content attributed to the TARGET citation;
   keep the other parts and every citation marker exactly as they appear (e.g., "ref [12]", "[3]").
5. Keep the original topic; do not turn the sentence into a statement about a different subject.
6. Do not add meta-commentary about the citation or the evidence (e.g., "the cited study does not address ...").
Return strictly a JSON object: {"revised_claim": "..."}"""


def fmt_evidence(paragraphs, max_chars=1500):
    return "\n\n".join(f"[{p['rank']}] {p['text'][:max_chars]}" for p in paragraphs) or "(no paragraphs)"


def feedback_user(s, candidates=None):
    u = (f"Survey title: {s['survey_title']}\n"
         f"Claim: {s['claim']}\n"
         f"Target citation marker: {s['marker']}\n"
         f"Cited reference title: {s['title']}\n\n"
         f"Verifier verdict: {s['last']['llm_label']} / NLI: {', '.join(s['last']['nli_labels'])}\n"
         f"Verifier reasoning: {s['last']['reasoning']}\n\n"
         f"Evidence paragraphs from the cited reference:\n{fmt_evidence(s['last']['evidence'])}")
    if candidates is not None:
        c = "\n\n".join(f"[C{i+1}] (ref {c['ref_no']}) {c['title']}\n    best-matching paragraph: "
                        f"{c['best_paragraph'][:700]}" for i, c in enumerate(candidates))
        u += f"\n\nCANDIDATE references from the same survey's bibliography:\n{c or '(none)'}"
    return u


def refine_user(s, cond, fb, evidence):
    u = (f"Survey title: {s['survey_title']}\n"
         f"Sentence: {s['claim']}\n"
         f"Target citation marker: {s['marker']}\n"
         f"Cited reference title: {s['title']}\n\n"
         f"Evidence paragraphs from the cited reference:\n{fmt_evidence(evidence)}\n\n")
    if cond == "no_fb":
        u += "This sentence was judged as not supported by the cited reference. Revise it."
    elif cond == "reason_fb":
        u += ("This sentence was judged as not supported by the cited reference.\n"
              f"Verifier feedback: {s['last']['reasoning']}\nRevise it.")
    else:
        u += ("This sentence was judged as not supported by the cited reference.\nFeedback:\n"
              f"- error type: {fb.get('error_type')}\n"
              f"- unsupported part: {fb.get('unsupported_part')}\n"
              f"- related evidence: {fb.get('supporting_evidence')}\n"
              f"- reason: {fb.get('reason')}\n"
              f"- suggestion: {fb.get('suggestion')}\n")
        if fb.get("action") == "recite":
            u += "- the citation has been replaced; the evidence above comes from the NEW reference.\n"
        u += "Revise the sentence following the feedback."
    return u


# ---------------------------------------------------------------------------
# Verification (batched)
# ---------------------------------------------------------------------------
async def verify_batch(states, retriever, nli, judge):
    """Retrieve top-5 for each state's current claim from its current reference; NLI + judge."""
    if not states:
        return
    q = retriever.encode_queries([s["claim"] for s in states])
    for s, e in zip(states, q):
        s["_evidence"] = retriever.topk(e, Path(s["ocr"])) if s["ocr"] else []
    pairs, idx = [], []
    for i, s in enumerate(states):
        for p in s["_evidence"]:
            pairs.append((p["text"], s["claim"]))
            idx.append(i)
    labels = nli(pairs)
    per = [[] for _ in states]
    for i, l in zip(idx, labels):
        per[i].append(l)
    verdicts = await judge.gather([judge_one(judge, s["claim"], s["_evidence"]) for s in states])
    for s, v, nl in zip(states, verdicts, per):
        s["last"] = {"evidence": s.pop("_evidence"), "nli_labels": nl,
                     "llm_label": v["label"], "reasoning": v["reasoning"],
                     "final": combine(v["label"], nl)}


def recite_candidates(s, retriever, q_emb, titles_cache):
    sd = Path(s["survey_dir"])
    if s["survey_dir"] not in titles_cache:
        titles_cache[s["survey_dir"]] = load_reference_titles(sd)
    titles = titles_cache[s["survey_dir"]]
    cands = []
    for ocr in sorted((sd / "reference_ocr").glob("*.json")):
        try:
            ref_no = int(ocr.name.split("_")[0])
        except ValueError:
            continue
        if ref_no == s["ref_no"]:
            continue
        top = retriever.topk(q_emb, ocr, k=1)
        if top:
            cands.append({"ref_no": ref_no, "title": titles.get(ref_no, ocr.stem),
                          "ocr": str(ocr), "score": top[0]["score"],
                          "best_paragraph": top[0]["text"]})
    best = {}
    for c in sorted(cands, key=lambda c: -c["score"]):  # one entry per refNo
        best.setdefault(c["ref_no"], c)
    return list(best.values())[:N_RECITE_CANDIDATES]


def replace_marker(claim, old_marker, new_ref_no):
    old = (old_marker or "").strip()
    if old.startswith("[") and old.endswith("]") and old in claim:
        return claim.replace(old, f"[{new_ref_no}]", 1), True
    if old and old in claim:
        return claim.replace(old, f"[{new_ref_no}]", 1), True
    return claim, False


def snapshot(s, loop, action=None, feedback=None):
    s["history"].append({"loop": loop, "claim": s["claim"], "ref_no": s["ref_no"],
                         "title": s["title"], "action": action, "feedback": feedback,
                         "llm_label": s["last"]["llm_label"], "reasoning": s["last"]["reasoning"],
                         "nli_labels": s["last"]["nli_labels"], "final": s["last"]["final"],
                         "evidence": s["last"]["evidence"], "status": s["status"]})


# ---------------------------------------------------------------------------
async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default=str(FTR_ROOT / "data" / "sample.jsonl"))
    ap.add_argument("--out", default=str(FTR_ROOT / "results" / "run_main"))
    ap.add_argument("--loops", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--resume", default="",
                    help="comma-separated conditions to resume from their saved <out>/<cond>.jsonl")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    conds = args.conditions.split(",")

    rows = [json.loads(l) for l in open(args.sample)]
    if args.limit:
        rows = rows[: args.limit]

    retriever, nli = Retriever(), NLI()
    judge = LLM(JUDGE_URL, JUDGE_NAME, concurrency=48, max_tokens=512)
    refiner = LLM(REFINER_URL, REFINER_NAME, concurrency=24, max_tokens=768)  # per replica; FP8 MoE degenerates at large batch on L40S

    # ---- loop 0: shared re-verification of the original claims
    base = []
    for r in rows:
        sd = survey_dir_of(r)
        ref_no = int(r["provenance"][0]["reference_number"])
        ocr = find_ocr_json(sd, ref_no)
        base.append({"uid": r["uid"], "group": f"{r['model'] or '-'}/{r['condition']}",
                     "orig_llm_label": r["llm_label"]["label"], "survey_title": r["paper_title"],
                     "survey_dir": str(sd), "orig_claim": r["sentence"], "claim": r["sentence"],
                     "orig_ref_no": ref_no, "ref_no": ref_no, "title": r["title"],
                     "marker": r["marker_text"], "ocr": str(ocr) if ocr else None,
                     "status": "active", "history": []})
    t0 = time.time()
    loop0 = out / "_loop0.jsonl"
    if loop0.exists():  # reuse the shared loop-0 verification
        base = [json.loads(l) for l in open(loop0)]
    else:
        await verify_batch(base, retriever, nli, judge)
        for s in base:
            if s["last"]["final"] == "supported":
                s["status"] = "supported"
            snapshot(s, 0)
        with open(loop0, "w") as f:
            for s in base:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[loop 0] unsupported={sum(s['status']=='active' for s in base)}/{len(base)} "
          f"({time.time()-t0:.0f}s)", flush=True)

    states = {c: copy.deepcopy(base) for c in conds}
    done = {c: 0 for c in conds}
    for c in [x for x in args.resume.split(",") if x]:
        states[c] = [json.loads(l) for l in open(out / f"{c}.jsonl")]
        done[c] = max(h["loop"] for s in states[c] for h in s["history"])
        print(f"[resume] {c} from loop {done[c]}", flush=True)
    titles_cache = {}
    for t in range(1, args.loops + 1):
        t0 = time.time()
        live = [c for c in conds if done[c] < t]
        # ---- (1) feedback
        fb_jobs = []
        for c in live:
            if c not in ("struct_fb", "struct_fb_recite"):
                continue
            act = [s for s in states[c] if s["status"] == "active"]
            if c == "struct_fb_recite" and act:
                q = retriever.encode_queries([s["claim"] for s in act])
                for s, e in zip(act, q):
                    s["_cands"] = recite_candidates(s, retriever, e, titles_cache)
            for s in act:
                sysmsg = feedback_system(c == "struct_fb_recite", t >= DROP_FROM_LOOP)
                fb_jobs.append((s, refiner.json_call(
                    sysmsg, feedback_user(s, s.get("_cands") if c == "struct_fb_recite" else None),
                    validate=valid_feedback)))
        fbs = await refiner.gather([j for _, j in fb_jobs])
        for (s, _), fb in zip(fb_jobs, fbs):
            s["_fb"] = fb

        # ---- (2) actions: drop / recite (switch reference, re-retrieve evidence) / revise
        to_refine, recite_states = [], []
        for c in live:
            for s in states[c]:
                if s["status"] != "active":
                    continue
                fb = s.get("_fb", {})
                s["_action"] = "revise"
                if c == "struct_fb_recite":
                    a = str(fb.get("action", "revise")).lower()
                    cands = s.get("_cands", [])
                    if a == "drop" and t >= DROP_FROM_LOOP:
                        s["_action"] = "drop"
                    elif a == "recite" and cands:
                        k = str(fb.get("recite_candidate") or "C1").upper().lstrip("C")
                        k = int(k) - 1 if k.isdigit() and 0 < int(k) <= len(cands) else 0
                        cand = cands[k]
                        s["_action"] = "recite"
                        s["claim"], _ = replace_marker(s["claim"], s["marker"], cand["ref_no"])
                        s["marker"] = f"[{cand['ref_no']}]"
                        s["ref_no"], s["title"], s["ocr"] = cand["ref_no"], cand["title"], cand["ocr"]
                        recite_states.append(s)
                if s["_action"] == "drop":
                    s["status"] = "dropped"
                    snapshot(s, t, "drop", fb)
                else:
                    to_refine.append((c, s))
        if recite_states:  # evidence of the newly cited reference for the refiner
            q = retriever.encode_queries([s["claim"] for s in recite_states])
            for s, e in zip(recite_states, q):
                s["last"] = dict(s["last"], evidence=retriever.topk(e, Path(s["ocr"])))

        # ---- (3) refine
        outs = await refiner.gather([refiner.json_call(
            REFINE_SYSTEM, refine_user(s, c, s.get("_fb", {}), s["last"]["evidence"]), validate=valid_refine)
            for c, s in to_refine])
        for (c, s), o in zip(to_refine, outs):
            new = (o.get("revised_claim") or "").strip()
            if new:
                s["claim"] = new

        # ---- (4) re-verify
        refined = [s for _, s in to_refine]
        await verify_batch(refined, retriever, nli, judge)
        for s in refined:
            if s["last"]["final"] == "supported":
                s["status"] = "supported"
            snapshot(s, t, s.pop("_action"), s.pop("_fb", None))
            s.pop("_cands", None)
        msg = " ".join(f"{c}={sum(x['status']=='active' for x in states[c])}" for c in conds)
        print(f"[loop {t}] unsupported: {msg} ({time.time()-t0:.0f}s)", flush=True)

        for c in conds:
            with open(out / f"{c}.jsonl", "w") as f:
                for s in states[c]:
                    f.write(json.dumps({k: v for k, v in s.items() if not k.startswith("_")},
                                       ensure_ascii=False) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
