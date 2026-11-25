"""Step 21a: question-conditioned sentence importance by teacher leave-one-out (no labels).

    CUDA_VISIBLE_DEVICES=4 python scripts/21_teacher_importance.py --manifest data/manifests/hotpotqa_train_gold2_dist3_n8000_full.json \
        --answers teacher --shard 0 --n-shards 4
    CUDA_VISIBLE_DEVICES=4 python scripts/21_teacher_importance.py --manifest data/manifests/musique_2hop_gold2_dist3_n1000_full.json \
        --answers raw --limit 300

For every question: the raw backbone (Mistral-7B-Instruct, Raw prompt) reads the five used_texts and we
take the teacher answer a (training set: the Step 11 teacher labels; evaluation sets: the Raw run's own
predictions on the base manifest — both are the original model's answers, no gold labels). Each
document is split into sentences that tile it exactly. For each sentence s:
    importance(s) = log p(a | all five documents) - log p(a | five documents with s removed)
High importance = the answer needs s for THIS question. Sentences are reported with character spans
in the sixth document's text (the five used_texts joined by "\\n"), so training can map tokens to them.
With --attention-n N, the first N questions also get an attention-based importance (mean attention
from answer tokens to the sentence's tokens over layers and heads) for a cross-check.

Output: artifacts/step21/importance/<manifest_id>[.shard<i>of<n>].jsonl
"""

import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

CPU_THREADS = int(os.environ.get("PISCO_CPU_THREADS", "4"))
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, str(CPU_THREADS))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["HF_HUB_OFFLINE"] = "1"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
OUT_DIR = PROJECT_ROOT / "artifacts" / "step21" / "importance"
TEACHER = str(PROJECT_ROOT / "artifacts" / "step11" / "teacher" / "hotpotqa_train_gold2_dist3_n8000.shard*.jsonl")
RUNS_DIR = PROJECT_ROOT / "artifacts" / "step08" / "runs"
PIECE_RE = re.compile(r"[^.!?]*(?:[.!?]+[\"')\]]*|$)\s*")
MIN_SENT_CHARS = 20
CHUNK = 8


def split_sentences(text):
    """Character spans that tile `text` exactly; very short pieces are merged into the previous one."""
    spans, pos = [], 0
    for m in PIECE_RE.finditer(text):
        if m.end() == m.start():
            continue
        s, e = m.start(), m.end()
        if spans and (e - s < MIN_SENT_CHARS or spans[-1][1] - spans[-1][0] < MIN_SENT_CHARS):
            spans[-1] = (spans[-1][0], e)
        else:
            spans.append((s, e))
        pos = e
    if not spans:
        spans = [(0, len(text))]
    if spans[0][0] != 0 or spans[-1][1] != len(text) or any(a[1] != b[0] for a, b in zip(spans, spans[1:])):
        raise RuntimeError("sentence spans do not tile the document")
    return spans


def raw_answers(manifest_path):
    """Latest Raw run on the base (non-_full) manifest: sample_id -> generated ids."""
    base = manifest_path.name.replace("_full.json", ".json")
    sha = hashlib.sha256((manifest_path.parent / base).read_bytes()).hexdigest()
    runs = [d for d in sorted(RUNS_DIR.glob("raw-*")) if (d / "run.json").exists()
            and json.loads((d / "run.json").read_text())["manifest"]["sha256"] == sha]
    if not runs:
        raise FileNotFoundError(f"no Raw run for {base}")
    return {r["sample_id"]: r["output_ids"] for r in map(json.loads, (runs[-1] / "predictions.jsonl").read_text().splitlines())}, runs[-1].name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--answers", choices=["teacher", "raw"], required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--attention-n", type=int, default=0)
    args = p.parse_args()
    import torch
    torch.set_num_threads(CPU_THREADS)
    from pisco_research.raw import RawBackbone
    manifest = json.loads(args.manifest.read_text())
    samples = manifest["samples"][:args.limit] if args.limit else manifest["samples"]
    samples = samples[args.shard::args.n_shards]
    if args.answers == "teacher":
        answers = {}
        for f in sorted(glob.glob(TEACHER)):
            for line in open(f):
                r = json.loads(line)
                answers[r["sample_id"]] = r["output_ids"]
        answer_source = "step11 teacher labels"
    else:
        answers, answer_source = raw_answers(args.manifest)
    ref = json.loads((PROJECT_ROOT / "reference" / "manifest.json").read_text())
    raw = RawBackbone(ref["snapshots"]["backbone"]["local_dir"])
    tok, model = raw.tokenizer, raw.model
    eos = tok.eos_token_id

    def answer_logps(prompts, ans_ids):
        """Sum log p(answer) for each prompt (answer ids fixed), in chunks, right padding."""
        out = []
        for i in range(0, len(prompts), CHUNK):
            rows, starts = [], []
            for pr in prompts[i:i + CHUNK]:
                ids = tok(pr, add_special_tokens=False)["input_ids"]
                starts.append(len(ids))
                rows.append(ids + ans_ids)
            L = max(len(r) for r in rows)
            ids = torch.full((len(rows), L), tok.bos_token_id, dtype=torch.long)
            mask = torch.zeros((len(rows), L), dtype=torch.long)
            for j, r in enumerate(rows):
                ids[j, :len(r)] = torch.tensor(r)
                mask[j, :len(r)] = 1
            with torch.no_grad():
                logits = model(input_ids=ids.cuda(), attention_mask=mask.cuda()).logits.float()
            lp = torch.log_softmax(logits, dim=-1)
            for j, st in enumerate(starts):
                pos = torch.arange(st - 1, st - 1 + len(ans_ids), device=lp.device)
                tgt = torch.tensor(ans_ids, device=lp.device)
                out.append(lp[j, pos, tgt].sum().item())
            del logits, lp
        return out

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f".shard{args.shard}of{args.n_shards}" if args.n_shards > 1 else ""
    out_path = OUT_DIR / f"{manifest['manifest_id']}{suffix}.jsonl"
    t0 = time.perf_counter()
    with open(out_path, "w") as f:
        for n, s in enumerate(samples):
            sid = s["sample_id"]
            docs = s["model_input"]["documents"][:-1]
            used = [d["used_text"] for d in docs]
            full = s["model_input"]["documents"][-1]["text"]
            if full != "\n".join(used):
                raise RuntimeError(f"{sid}: sixth document is not the joined used_texts")
            ans = [t for t in answers[sid]]
            if eos in ans:
                ans = ans[:ans.index(eos) + 1]
            if not ans:
                continue
            q = s["model_input"]["question"]
            sents, prompts = [], [raw.build_prompt(q, used)]
            offset = 0
            for di, text in enumerate(used):
                for a, b in split_sentences(text):
                    ablated = list(used)
                    ablated[di] = (text[:a] + text[b:]).strip()
                    prompts.append(raw.build_prompt(q, ablated))
                    sents.append({"doc": di, "start": offset + a, "end": offset + b})
                offset += len(text) + 1
            lps = answer_logps(prompts, ans)
            base = lps[0]
            for st, lp in zip(sents, lps[1:]):
                st["importance"] = round(base - lp, 4)
            rec = {"sample_id": sid, "base_logp": round(base, 4), "answer_len": len(ans), "n_sentences": len(sents),
                   "sentences": sents, "answer_source": answer_source}
            if n < args.attention_n:
                rec["attention"] = attention_importance(raw, q, used, full, sents, ans)
            f.write(json.dumps(rec) + "\n")
            if n % 50 == 0:
                print(f"[imp {manifest['manifest_id']}{suffix}] {n}/{len(samples)} {time.perf_counter()-t0:.0f}s "
                      f"sents {len(sents)} max_imp {max(x['importance'] for x in sents):.2f}", flush=True)
    print("written:", out_path.relative_to(PROJECT_ROOT), len(samples))


def attention_importance(raw, question, used, full, sents, ans):
    """Mean attention (all layers, all heads) from answer tokens to each sentence's tokens."""
    import torch
    tok, model = raw.tokenizer, raw.model
    prompt = raw.build_prompt(question, used)
    base = prompt.find(full)
    if base < 0:
        return None
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"] + ans
    prev = model.config._attn_implementation
    model.config._attn_implementation = "eager"
    try:
        with torch.no_grad():
            out = model(input_ids=torch.tensor([ids]).cuda(), output_attentions=True)
    finally:
        model.config._attn_implementation = prev
    P = len(enc["input_ids"])
    att = torch.stack([a[0].float().mean(0) for a in out.attentions]).mean(0)   # [T, T]
    ans_att = att[P - 1:P - 1 + len(ans), :P].mean(0)                           # attention onto prompt tokens
    scores = []
    for st in sents:
        a, b = base + st["start"], base + st["end"]
        sel = [i for i, (x, y) in enumerate(enc["offset_mapping"]) if x < b and y > a]
        scores.append(round(ans_att[sel].sum().item(), 6) if sel else 0.0)
    return scores


if __name__ == "__main__":
    main()
