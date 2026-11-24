"""Per-loop metrics for the FtR runs + meaning-preservation judging.

For every condition c and loop t (state of each pair after loop t):
  unsupported / supported / dropped rates,
  for pairs that became supported through refinement:
    preservation score (judge LLM, 1-5), embedding similarity to the original claim,
    ROUGE-L precision against the best evidence paragraph (copying), length ratio,
  faithful&preserved rate = supported with preservation >= 4, over all N.
"""
import argparse
import asyncio
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from common import FTR_ROOT, JUDGE_NAME, JUDGE_URL, LLM

PRESERVE_SYSTEM = """You compare an ORIGINAL sentence from a scientific survey with a REVISED version of it.
Rate how well the revised sentence preserves the original's core claim and its role in the survey
(topic and main point). Removing, narrowing or softening unsupported details is acceptable and should
not be penalised much; changing the topic or the main point, or producing a vacuous sentence, should be.
5: same core claim; only wording or precision changed
4: core claim kept, but scope narrowed/softened or a secondary detail removed
3: same topic, but the main point substantially changed
2: a different claim on a related topic
1: unrelated or vacuous sentence
Return strictly a JSON object: {"score": <1-5>, "reason": "<one sentence>"}"""


def toks(s):
    s = re.sub(r"ref\s*\[[^\]]*\]|\[[\d,\s\-–]+\]", " ", s.lower())
    return re.findall(r"[a-z0-9]+", s)


def lcs(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def copy_rate(claim, evidence):
    """ROUGE-L precision of the claim w.r.t. its most similar evidence paragraph."""
    c = toks(claim)
    if not c or not evidence:
        return 0.0
    return max(lcs(c, toks(p["text"])) / len(c) for p in evidence)


def state_at(hist, t):
    h = [x for x in hist if x["loop"] <= t]
    return h[-1]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(FTR_ROOT / "results" / "run_main"))
    ap.add_argument("--loops", type=int, default=5)
    args = ap.parse_args()
    run = Path(args.run)
    conds = [p.stem for p in sorted(run.glob("*.jsonl")) if not p.stem.startswith("_")]
    data = {c: [json.loads(l) for l in open(run / f"{c}.jsonl")] for c in conds}

    # ---- preservation judging for every distinct (original, revised) pair
    pres_path = run / "_preservation.json"
    pres = json.load(open(pres_path)) if pres_path.exists() else {}
    todo = set()
    for c in conds:
        for s in data[c]:
            for h in s["history"]:
                if h["loop"] > 0 and h["claim"] != s["orig_claim"]:
                    k = s["orig_claim"] + "␞" + h["claim"]
                    if k not in pres:
                        todo.add(k)
    if todo:
        judge = LLM(JUDGE_URL, JUDGE_NAME, concurrency=48, max_tokens=200)
        todo = sorted(todo)
        outs = await judge.gather([judge.json_call(
            PRESERVE_SYSTEM, f"ORIGINAL: {k.split(chr(0x241e))[0]}\nREVISED: {k.split(chr(0x241e))[1]}")
            for k in todo])
        for k, o in zip(todo, outs):
            try:
                pres[k] = int(o.get("score"))
            except (TypeError, ValueError):
                pres[k] = None
        json.dump(pres, open(pres_path, "w"), ensure_ascii=False)

    # ---- embedding similarity (bge) for all distinct revised claims
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")
    texts = sorted({s["orig_claim"] for c in conds for s in data[c]} |
                   {h["claim"] for c in conds for s in data[c] for h in s["history"]})
    E = dict(zip(texts, emb.encode(texts, batch_size=128, normalize_embeddings=True)))

    summary = {}
    for c in conds:
        N = len(data[c])
        rows = []
        for t in range(0, args.loops + 1):
            st = [(s, state_at(s["history"], t)) for s in data[c]]
            cnt = Counter(h["status"] for _, h in st)
            ref = [(s, h) for s, h in st if h["status"] == "supported" and h["loop"] > 0]
            ps = [pres.get(s["orig_claim"] + "␞" + h["claim"], 5 if h["claim"] == s["orig_claim"] else None)
                  for s, h in ref]
            ps_valid = [p for p in ps if p is not None]
            sims = [float(E[s["orig_claim"]] @ E[h["claim"]]) for s, h in ref]
            cps = [copy_rate(h["claim"], h["evidence"]) for s, h in ref]
            cps0 = [copy_rate(s["orig_claim"], s["history"][0]["evidence"]) for s, h in ref]
            lens = [len(toks(h["claim"])) / max(1, len(toks(s["orig_claim"]))) for s, h in ref]
            n_pres = sum(1 for p in ps if p is not None and p >= 4)
            sup0 = sum(1 for s, h in st if h["status"] == "supported" and h["loop"] == 0)
            rows.append({
                "loop": t, "N": N,
                "unsupported_rate": cnt["active"] / N,
                "supported_rate": cnt["supported"] / N,
                "dropped_rate": cnt["dropped"] / N,
                "refined_supported": len(ref),
                "faithful_preserved_rate": (sup0 + n_pres) / N,
                "preservation_mean": float(np.mean(ps_valid)) if ps_valid else None,
                "preservation_ge4": n_pres / len(ref) if ref else None,
                "sim_mean": float(np.mean(sims)) if sims else None,
                "copy_rouge_l_p": float(np.mean(cps)) if cps else None,
                "copy_rouge_l_p_orig": float(np.mean(cps0)) if cps0 else None,
                "len_ratio": float(np.mean(lens)) if lens else None,
            })
        # breakdowns at the final loop
        fin = [(s, state_at(s["history"], args.loops)) for s in data[c]]
        act0 = [(s, h) for s, h in fin if s["history"][0]["status"] == "active"]
        by_orig = defaultdict(Counter)
        by_group = defaultdict(Counter)
        for s, h in act0:
            by_orig[s["orig_llm_label"]][h["status"]] += 1
            by_group[s["group"]][h["status"]] += 1
        actions = Counter(x["action"] for s in data[c] for x in s["history"] if x["loop"] > 0)
        etype = defaultdict(Counter)
        first_resolved = Counter()
        for s, h in act0:
            fb1 = next((x for x in s["history"] if x["loop"] == 1), None)
            if fb1 and fb1.get("feedback"):
                etype[str(fb1["feedback"].get("error_type"))][h["status"]] += 1
            if h["status"] == "supported":
                first_resolved[next(x["loop"] for x in s["history"] if x["status"] == "supported")] += 1
        summary[c] = {"per_loop": rows,
                      "final_by_orig_label": {k: dict(v) for k, v in by_orig.items()},
                      "final_by_group": {k: dict(v) for k, v in by_group.items()},
                      "actions": dict(actions),
                      "final_by_error_type": {k: dict(v) for k, v in etype.items()},
                      "resolved_at_loop": dict(sorted(first_resolved.items()))}
    json.dump(summary, open(run / "_summary.json", "w"), indent=2, ensure_ascii=False)

    for c in conds:
        print(f"\n== {c}")
        print("loop  unsup  sup   drop  F&P   pres  pres>=4  sim   copy(orig->new)  len")
        for r in summary[c]["per_loop"]:
            f = lambda x, d=3: "-" if x is None else f"{x:.{d}f}"
            print(f"{r['loop']:>4}  {r['unsupported_rate']:.3f}  {r['supported_rate']:.3f} "
                  f"{r['dropped_rate']:.3f} {r['faithful_preserved_rate']:.3f} {f(r['preservation_mean'],2)} "
                  f"{f(r['preservation_ge4'])}  {f(r['sim_mean'])}  {f(r['copy_rouge_l_p_orig'])}->{f(r['copy_rouge_l_p'])}  {f(r['len_ratio'],2)}")
        print("actions", summary[c]["actions"])
        print("by error type", summary[c]["final_by_error_type"])


if __name__ == "__main__":
    asyncio.run(main())
