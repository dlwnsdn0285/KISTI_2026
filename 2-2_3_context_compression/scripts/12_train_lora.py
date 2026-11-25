"""Step 12a: control trainings for the fusion result — fine-tune ONE of PISCO's own LoRA adapters
with the same data and objective as Step 11 (teacher KD on HotpotQA train, 8000).

    CUDA_VISIBLE_DEVICES=6 python scripts/12_train_lora.py --target decoder --out artifacts/step12/ckpt/decoder_lora
    CUDA_VISIBLE_DEVICES=7 python scripts/12_train_lora.py --target encoder --out artifacts/step12/ckpt/encoder_lora

  decoder  only decoder_adapter LoRA weights train (memory = cached, fixed)   -> "the decoder cannot combine"
  encoder  only encoder_adapter LoRA weights train (memory recomputed with grad) -> "the memory lacks information"

Everything else is frozen; dropout off; gradient checkpointing on. Same split, batch, schedule as Step 11.
Checkpoint: <out>/lora.pt {"kind": "lora", "adapter": ..., "state_dict": {decoder param name: tensor}}
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
MANIFEST = PROJECT_ROOT / "data" / "manifests" / "hotpotqa_train_gold2_dist3_n8000.json"
MEMORY = PROJECT_ROOT / "artifacts" / "step11" / "memory" / "hotpotqa_train_gold2_dist3_n8000.pt"
LABELS = str(PROJECT_ROOT / "artifacts" / "step11" / "teacher" / "hotpotqa_train_gold2_dist3_n8000.shard*.jsonl")
ADAPTER = {"decoder": "decoder_adapter", "encoder": "encoder_adapter"}


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--target", choices=["decoder", "encoder"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--memory", type=Path, default=MEMORY)
    p.add_argument("--labels", default=LABELS)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--val-n", type=int, default=200)
    p.add_argument("--val-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-answer-tokens", type=int, default=128)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def load_examples(args):
    manifest = json.loads(args.manifest.read_text())
    import torch
    mem = torch.load(args.memory, map_location="cpu", weights_only=False)
    if mem["manifest_sha256"] != hashlib.sha256(args.manifest.read_bytes()).hexdigest():
        raise RuntimeError("cached memory was computed for a different manifest")
    mem_index = {sid: i for i, sid in enumerate(mem["sample_ids"])}
    labels = {}
    for f in sorted(glob.glob(args.labels)):
        for line in open(f):
            r = json.loads(line)
            labels[r["sample_id"]] = r
    examples = []
    for s in manifest["samples"]:
        sid = s["sample_id"]
        if sid not in labels or sid not in mem_index:
            continue
        examples.append({"sample_id": sid, "question": s["model_input"]["question"],
                         "docs": [d["text"] for d in s["model_input"]["documents"]],
                         "k": len(s["model_input"]["documents"]), "teacher": labels[sid]["teacher"], "mem_i": mem_index[sid]})
    return examples, mem["memory"]


def main():
    args = parse()
    import torch
    torch.set_num_threads(CPU_THREADS)
    torch.manual_seed(args.seed)
    from pisco_research.fusion import build_training_batch, embed_with_memory
    from pisco_research.reference import PiscoReference
    spec = importlib.util.spec_from_file_location("step03", PROJECT_ROOT / "scripts" / "03_load_reference.py")
    step03 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(step03)

    examples, memory_all = load_examples(args)
    rng = random.Random(args.seed)            # same shuffle as Step 11 -> same split
    rng.shuffle(examples)
    if args.smoke:
        examples = examples[:24]
        args.val_n, args.val_every, args.save_every, args.warmup = 8, 2, 1000, 1
    val, train = examples[:args.val_n], examples[args.val_n:]
    steps_total = int(math.ceil(len(train) * args.epochs / (args.batch_size * args.accum)))
    if args.smoke:
        steps_total = 4

    model, _ = step03.load_reference_model({})
    wrapper = PiscoReference(model)
    adapter = ADAPTER[args.target]
    trainable = {n: p for n, p in model.decoder.named_parameters() if "lora_" in n and f".{adapter}." in n}
    if not trainable:
        raise RuntimeError(f"no LoRA parameters found for {adapter}")

    def apply_trainable():
        """set_adapter() in PEFT toggles requires_grad; re-apply our choice before every forward."""
        for n, p in model.decoder.named_parameters():
            p.requires_grad_(n in trainable)
    model.decoder.train()
    for m in model.decoder.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    model.decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    apply_trainable()
    n_params = sum(p.numel() for p in trainable.values())
    opt = torch.optim.AdamW(list(trainable.values()), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        progress = (step - args.warmup) / max(1, steps_total - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    tok = wrapper.tokenizer

    def compress_with_grad(batch):
        enc = wrapper.prepare_encoder_inputs([d for e in batch for d in e["docs"]])
        device = model.decoder.device
        apply_trainable()
        memory = model.compress(enc["input_ids"].to(device), enc["attention_mask"].to(device))   # sets encoder_adapter
        k = batch[0]["k"]
        return memory.view(len(batch), k, wrapper.n_mem, wrapper.hidden_size)

    def batch_loss(batch, train_mode):
        questions = [e["question"] for e in batch]
        answers = [tok.decode(tok(e["teacher"], add_special_tokens=False)["input_ids"][:args.max_answer_tokens]) for e in batch]
        k = batch[0]["k"]
        input_ids, attention_mask, labels, blocks = build_training_batch(wrapper, questions, answers, k)
        input_ids, attention_mask, labels = input_ids.cuda(), attention_mask.cuda(), labels.cuda()
        ctx = torch.enable_grad() if train_mode else torch.no_grad()
        with ctx:
            if args.target == "encoder":
                memory_src = compress_with_grad(batch)
                # Two-stage backward: checkpointed segments are recomputed during backward with the
                # adapter active at that time, so the encoder pass must be re-run under encoder_adapter.
                memory = memory_src.detach().requires_grad_(train_mode)
            else:
                memory_src, memory = None, memory_all[[e["mem_i"] for e in batch]].cuda()
            if "decoder_adapter" in model.adapter_keys:
                model.decoder.set_adapter("decoder_adapter")
            apply_trainable()
            embeds = embed_with_memory(model, input_ids, blocks, memory)
            out = model.decoder(inputs_embeds=embeds, attention_mask=attention_mask, labels=labels)
        return out.loss, memory_src, memory

    def backward(loss, memory_src, memory, scale):
        (loss / scale).backward()
        if memory_src is not None:
            model.decoder.set_adapter("encoder_adapter")
            apply_trainable()
            memory_src.backward(memory.grad)

    def evaluate(examples_):
        losses = []
        for i in range(0, len(examples_), args.batch_size):
            losses.append(batch_loss(examples_[i:i + args.batch_size], train_mode=False)[0].item())
        return sum(losses) / len(losses)

    def save(path, extra):
        torch.save({"kind": "lora", "adapter": adapter, "target": args.target,
                    "state_dict": {n: p.detach().cpu() for n, p in trainable.items()}, "extra": extra}, path)

    args.out.mkdir(parents=True, exist_ok=True)
    log = {"args": {k: str(v) for k, v in vars(args).items()}, "n_train": len(train), "n_val": len(val),
           "steps_total": steps_total, "trainable_params": n_params, "adapter": adapter, "events": []}
    val0 = evaluate(val)
    log["events"].append({"step": 0, "val_loss": val0, "note": "pretrained PISCO"})
    print(f"[{args.target}] trainable {n_params/1e6:.1f}M ({len(trainable)} tensors) train {len(train)} val {len(val)} "
          f"steps {steps_total} val_loss@0 {val0:.4f}", flush=True)

    step, t0, pos = 0, time.perf_counter(), 0
    running = []
    while step < steps_total:
        opt.zero_grad(set_to_none=True)
        acc = 0.0
        for _ in range(args.accum):
            if pos + args.batch_size > len(train):
                rng.shuffle(train)
                pos = 0
            batch = train[pos:pos + args.batch_size]
            pos += args.batch_size
            loss, memory_src, memory = batch_loss(batch, train_mode=True)
            backward(loss, memory_src, memory, args.accum)
            acc += loss.item() / args.accum
        torch.nn.utils.clip_grad_norm_(list(trainable.values()), 1.0)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.step()
        step += 1
        running.append(acc)
        if step % 20 == 0 or step == steps_total:
            print(f"[{args.target}] step {step}/{steps_total} loss {sum(running[-20:])/len(running[-20:]):.4f} "
                  f"lr {lr_at(step-1):.2e} {time.perf_counter()-t0:.0f}s mem {torch.cuda.max_memory_allocated()/2**30:.1f}G", flush=True)
        if step % args.val_every == 0 or step == steps_total:
            v = evaluate(val)
            log["events"].append({"step": step, "train_loss": sum(running[-args.val_every:]) / len(running[-args.val_every:]), "val_loss": v})
            print(f"[{args.target}] step {step} val_loss {v:.4f}", flush=True)
        if step % args.save_every == 0 and step != steps_total:
            save(args.out / f"lora_step{step}.pt", {"step": step})
    extra = {"step": step, "target": args.target, "val_loss_final": log["events"][-1]["val_loss"], "val_loss_pretrained": val0,
             "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(), "labels": args.labels}
    save(args.out / "lora.pt", extra)
    log["final"] = extra
    log["seconds"] = round(time.perf_counter() - t0)
    (args.out / "train_log.json").write_text(json.dumps(log, indent=2))
    print("saved:", args.out / "lora.pt", extra)


if __name__ == "__main__":
    main()
