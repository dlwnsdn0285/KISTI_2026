"""Step 19: full-document block generator under capacity pressure (PISCO compressor + decoder frozen).

Generalises Step 18: the generator reads the FIVE documents exactly as the memories saw them
(<= 640 tokens, the sixth document of *_full manifests) instead of a 120-token excerpt, so one
8-vector block cannot hold everything. --no-residual trains the repetition control (QA KD only).

    CUDA_VISIBLE_DEVICES=4 python scripts/19_train_full_gen.py --variant text --no-residual --out artifacts/step19/ckpt/rep_full
    CUDA_VISIBLE_DEVICES=5 python scripts/19_train_full_gen.py --variant text    --out artifacts/step19/ckpt/res_full_text
    CUDA_VISIBLE_DEVICES=6 python scripts/19_train_full_gen.py --variant textmem --out artifacts/step19/ckpt/res_full_textmem

Original Step 18 notes follow.
Step 18: train a RESIDUAL excerpt-block generator (PISCO compressor + decoder frozen).

    CUDA_VISIBLE_DEVICES=4 python scripts/18_train_residual_gen.py --variant textmem --out artifacts/step18/ckpt/res_textmem
    CUDA_VISIBLE_DEVICES=5 python scripts/18_train_residual_gen.py --variant text    --out artifacts/step18/ckpt/res_text

Loss = QA KD (as Step 17)  +  lambda * residual reconstruction:
    prompt "Repeat the relevant excerpts as completely as possible." with [5 memories, block];
    target = the excerpt text. Per-token CE is weighted by w_t = CE_t(memory only), stop-grad,
    normalised to mean 1 per sample and clipped at W_MAX. Tokens the memories already let the decoder
    reproduce get low weight; tokens the memories miss (values, polarity, the hop-2 answer) get high
    weight, so under the fixed 8-vector budget the block is pushed to carry what the memories lack.
textmem: the generator sees the five memories (through a learned per-dimension affine MemProj,
initialised to token-embedding scale) + question + excerpt. text: question + excerpt only (control).
Same data, split, budget and schedule as Step 17 (1 epoch, lr 5e-5).
Checkpoint: <out>/gen.pt {"kind": "excerptgen", "variant", "state_dict", "memproj"}
"""

import argparse
import glob
import hashlib
import importlib.util
import json
import math
import os
import random
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
MANIFEST = PROJECT_ROOT / "data" / "manifests" / "hotpotqa_train_gold2_dist3_n8000_full.json"
MEMORY = PROJECT_ROOT / "artifacts" / "step11" / "memory" / "hotpotqa_train_gold2_dist3_n8000.pt"
LABELS = str(PROJECT_ROOT / "artifacts" / "step11" / "teacher" / "hotpotqa_train_gold2_dist3_n8000.shard*.jsonl")
RESTATE = "Repeat the documents as completely as possible."
GEN_MAX_LEN = 736          # question (<= ~70 tokens) + five documents (<= 650 tokens)
EXCERPT_PREFIX = "Relevant excerpts: "
W_MAX = 5.0


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["text", "textmem"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--lam", type=float, default=1.0, help="weight of the residual reconstruction loss")
    p.add_argument("--no-residual", action="store_true", help="repetition control: QA KD only")
    p.add_argument("--max-recon-tokens", type=int, default=660)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--proj-lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--val-n", type=int, default=200)
    p.add_argument("--val-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-answer-tokens", type=int, default=128)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main():
    args = parse()
    global TAG
    TAG = ("rep" if args.no_residual else "res") + "_full_" + args.variant
    import torch
    torch.set_num_threads(CPU_THREADS)
    torch.manual_seed(args.seed)
    from pisco_research.excerpt_gen import (ADAPTER, adapter_params, add_excerpt_adapter, generate_block_res,
                                            init_memproj, token_ce)
    from pisco_research.fusion import build_training_batch, embed_with_memory
    from pisco_research.reference import PiscoReference
    spec = importlib.util.spec_from_file_location("step03", PROJECT_ROOT / "scripts" / "03_load_reference.py")
    step03 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(step03)

    manifest = json.loads(MANIFEST.read_text())
    mem = torch.load(MEMORY, map_location="cpu", weights_only=False)
    mem_index = {sid: i for i, sid in enumerate(mem["sample_ids"])}
    labels = {}
    for f in sorted(glob.glob(LABELS)):
        for line in open(f):
            r = json.loads(line)
            labels[r["sample_id"]] = r
    examples = []
    for s in manifest["samples"]:
        sid = s["sample_id"]
        if sid not in labels or sid not in mem_index:
            continue
        docs = s["model_input"]["documents"]
        examples.append({"sample_id": sid, "question": s["model_input"]["question"], "k": len(docs) - 1,
                         "excerpt": docs[-1]["text"], "teacher": labels[sid]["teacher"], "mem_i": mem_index[sid]})
    rng = random.Random(args.seed)            # same split as Steps 11-17
    rng.shuffle(examples)
    if args.smoke:
        examples = examples[:24]
        args.val_n, args.val_every, args.warmup = 8, 2, 1
    val, train = examples[:args.val_n], examples[args.val_n:]
    steps_total = 4 if args.smoke else int(math.ceil(len(train) * args.epochs / (args.batch_size * args.accum)))

    model, _ = step03.load_reference_model({})
    wrapper = PiscoReference(model)
    tok = wrapper.tokenizer
    copied = add_excerpt_adapter(model)
    trainable = adapter_params(model)
    memproj = init_memproj(model, mem["memory"][:256]) if args.variant == "textmem" else None

    def apply_trainable():
        for n, p in model.decoder.named_parameters():
            p.requires_grad_(n in trainable)
    model.decoder.train()
    for m in model.decoder.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    model.decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    apply_trainable()
    groups = [{"params": list(trainable.values()), "lr": args.lr}]
    if memproj is not None:
        groups.append({"params": list(memproj.parameters()), "lr": args.proj_lr})
    base_lrs = [g["lr"] for g in groups]
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.98))
    n_params = sum(p.numel() for g in groups for p in g["params"])

    def lr_scale(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        progress = (step - args.warmup) / max(1, steps_total - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    def excerpt_target(e):
        return tok.decode(tok(e["excerpt"], add_special_tokens=False)["input_ids"][:args.max_recon_tokens])

    def batch_loss(batch, train_mode):
        questions = [e["question"] for e in batch]
        answers = [tok.decode(tok(e["teacher"], add_special_tokens=False)["input_ids"][:args.max_answer_tokens]) for e in batch]
        targets = [excerpt_target(e) for e in batch]
        k = batch[0]["k"]
        qa_ids, qa_mask, qa_lab, qa_blocks = [t.cuda() if torch.is_tensor(t) else t for t in build_training_batch(wrapper, questions, answers, k + 1)]
        memory = mem["memory"][[e["mem_i"] for e in batch]].cuda()
        do_res = not (train_mode and args.no_residual)       # the control still reports residual metrics on validation
        if do_res:
            rs_ids, rs_mask, rs_lab, rs_blocks = [t.cuda() if torch.is_tensor(t) else t for t in build_training_batch(wrapper, [RESTATE] * len(batch), targets, k + 1)]
            r0_ids, r0_mask, r0_lab, r0_blocks = [t.cuda() if torch.is_tensor(t) else t for t in build_training_batch(wrapper, [RESTATE] * len(batch), targets, k)]
        # memory-only reconstruction: what the decoder already gets from the five memories (no grad)
        model.decoder.set_adapter("decoder_adapter")
        weights, ce0 = [], []
        if do_res:
            with torch.no_grad():
                out0 = model.decoder(inputs_embeds=embed_with_memory(model, r0_ids, r0_blocks, memory), attention_mask=r0_mask)
                ce0 = token_ce(out0.logits, r0_lab)
                del out0
            for c in ce0:
                w = c / c.mean().clamp_min(1e-6)
                weights.append(w.clamp(max=W_MAX))
        ctx = torch.enable_grad() if train_mode else torch.no_grad()
        with ctx:
            apply_trainable()
            block_src = generate_block_res(model, tok, questions, [e["excerpt"] for e in batch],
                                           memory if args.variant == "textmem" else None, memproj, GEN_MAX_LEN)
            block = block_src.detach().requires_grad_(train_mode)
            model.decoder.set_adapter("decoder_adapter")
            apply_trainable()
            fused = torch.cat([memory, block[:, None].to(memory.dtype)], dim=1)
            out_qa = model.decoder(inputs_embeds=embed_with_memory(model, qa_ids, qa_blocks, fused), attention_mask=qa_mask, labels=qa_lab)
            if do_res:
                out_rs = model.decoder(inputs_embeds=embed_with_memory(model, rs_ids, rs_blocks, fused), attention_mask=rs_mask)
                ce1 = token_ce(out_rs.logits, rs_lab)
                del out_rs
                res = torch.stack([(w * c).sum() / w.sum() for w, c in zip(weights, ce1)]).mean()
                plain = torch.stack([c.mean() for c in ce1]).mean()
                base = torch.stack([(w * c).sum() / w.sum() for w, c in zip(weights, ce0)]).mean()
            else:
                res = plain = base = torch.tensor(float("nan"))
            loss = out_qa.loss + (0.0 if args.no_residual else args.lam * res)
        return loss, block_src, block, {"qa": out_qa.loss.item(), "res": res.item(), "res_memonly": base.item(), "restate_ce": plain.item()}

    def backward(loss, block_src, block, scale):
        (loss / scale).backward()
        model.decoder.set_adapter(ADAPTER)
        apply_trainable()
        block_src.backward(block.grad.to(block_src.dtype))

    def evaluate(xs):
        acc = {"qa": 0.0, "res": 0.0, "res_memonly": 0.0, "restate_ce": 0.0}
        nb = 0
        for i in range(0, len(xs), args.batch_size):
            _, _, _, parts = batch_loss(xs[i:i + args.batch_size], False)
            for k_ in acc:
                acc[k_] += parts[k_]
            nb += 1
        return {k_: v / nb for k_, v in acc.items()}

    def save(path, extra):
        torch.save({"kind": "excerptgen", "variant": args.variant, "adapter": ADAPTER, "residual": not args.no_residual,
                    "input": "full", "input_max_len": GEN_MAX_LEN, "restate_prompt": RESTATE,
                    "state_dict": {n: p.detach().cpu() for n, p in trainable.items()},
                    "memproj": {k_: v.detach().cpu() for k_, v in memproj.state_dict().items()} if memproj is not None else None,
                    "extra": extra}, path)

    args.out.mkdir(parents=True, exist_ok=True)
    log = {"args": {k_: str(v) for k_, v in vars(args).items()}, "n_train": len(train), "n_val": len(val), "steps_total": steps_total,
           "trainable_params": n_params, "copied_tensors": copied, "w_max": W_MAX, "events": []}
    v0 = evaluate(val)
    log["events"].append({"step": 0, **{f"val_{k_}": v for k_, v in v0.items()}})
    print(f"[{TAG}] trainable {n_params/1e6:.1f}M steps {steps_total} val@0 " + " ".join(f"{k_} {v:.4f}" for k_, v in v0.items()), flush=True)
    step, t0, pos, running = 0, time.perf_counter(), 0, []
    while step < steps_total:
        opt.zero_grad(set_to_none=True)
        acc = 0.0
        for _ in range(args.accum):
            if pos + args.batch_size > len(train):
                rng.shuffle(train)
                pos = 0
            batch = train[pos:pos + args.batch_size]
            pos += args.batch_size
            loss, bs, b, _ = batch_loss(batch, True)
            backward(loss, bs, b, args.accum)
            acc += loss.item() / args.accum
        params = [p for g in groups for p in g["params"]]
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        for g, base_lr in zip(opt.param_groups, base_lrs):
            g["lr"] = base_lr * lr_scale(step)
        opt.step()
        step += 1
        running.append(acc)
        if step % 20 == 0 or step == steps_total:
            print(f"[{TAG}] step {step}/{steps_total} loss {sum(running[-20:])/len(running[-20:]):.4f} "
                  f"{time.perf_counter()-t0:.0f}s mem {torch.cuda.max_memory_allocated()/2**30:.1f}G", flush=True)
        if step % args.val_every == 0 or step == steps_total:
            v = evaluate(val)
            log["events"].append({"step": step, "train_loss": sum(running[-args.val_every:]) / len(running[-args.val_every:]),
                                  **{f"val_{k_}": x for k_, x in v.items()}})
            print(f"[{TAG}] step {step} val " + " ".join(f"{k_} {x:.4f}" for k_, x in v.items()), flush=True)
    extra = {"step": step, "variant": args.variant, "lam": args.lam, **{f"val_{k_}_final": x for k_, x in log["events"][-1].items() if k_.startswith("val_")},
             **{f"val_{k_}_init": x for k_, x in v0.items()}, "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest()}
    save(args.out / "gen.pt", extra)
    log["final"] = extra
    log["seconds"] = round(time.perf_counter() - t0)
    (args.out / "train_log.json").write_text(json.dumps(log, indent=2))
    print("saved:", args.out / "gen.pt", extra)


if __name__ == "__main__":
    main()
