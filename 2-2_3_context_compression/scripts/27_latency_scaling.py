"""Step 27: response-time scaling with the number of retrieved documents (KISTI v2 index).

  # 1) documents: top-NMAX unique chunks per query from the Step 26 v2 index (conda env `refrag`)
  CUDA_VISIBLE_DEVICES=0 ~/anaconda3/envs/refrag/bin/python scripts/27_latency_scaling.py prepare
  # 2) timing, one GPU per group of N values (conda env `pisco-ref`)
  CUDA_VISIBLE_DEVICES=0 ~/anaconda3/envs/pisco-ref/bin/python scripts/27_latency_scaling.py run --ns 320
  # 3) merge -> artifacts/step27/{results.json,curves.csv}
  python scripts/27_latency_scaling.py summarize

Conditions (same N documents, same question, greedy, exactly MAX_NEW new tokens: min_new_tokens = max_new_tokens)
  raw     Mistral-7B-Instruct-v0.2 reads the N chunk texts (Step 08 raw prompt)
  ours    PISCO memories of the N chunks precomputed (CPU cache; CPU->GPU transfer timed) + extra block from the
          top-5 chunks (Step 19 rep_full generator, question + 5 texts) + PISCO decoder over N*8 + 8 memory slots
  online  as ours, but the N memories are computed by the PISCO compressor at query time (timed)
Metrics (REFRAG definitions)
  ttft        query arrival -> first new token: transfer/compression + block + decoder prefill + first token
  ttit        mean time per new token after the first (timestamps of every decoding step, one generate call)
  throughput  batch B queries end to end: B * MAX_NEW / (pre + generate) tokens/s; B starts at BATCH_MAX and is
              halved on CUDA OOM (the batch actually used is recorded per condition and N)
Every condition of a given N runs on the same GPU, so the per-N accelerations (raw / method for ttft and ttit,
method / raw for throughput) never mix GPUs. Tokenization and decoder-input construction are outside the timers
(as in Step 25). The PISCO decoder was trained with 5 documents: answers for N > 5 are not used, only times.
"""

import argparse
import importlib.util
import json
import os
import random
import statistics
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
# LAT_TAG / LAT_NMAX / LAT_QUERIES / LAT_WARMUP / LAT_THR_REPS override the defaults (e.g. the precise k=5 run:
# LAT_TAG=k5 LAT_NMAX=5 LAT_QUERIES=100 LAT_WARMUP=5 LAT_THR_REPS=10); outputs then carry the tag
TAG = os.environ.get("LAT_TAG", "")
DOCS_FILE = PROJECT_ROOT / "data" / "kisti" / (f"latency_{TAG}.json" if TAG else "latency_top320.json")
OUT_DIR = PROJECT_ROOT / "artifacts" / "step27"
GEN = PROJECT_ROOT / "artifacts" / "step19" / "ckpt" / "rep_full" / "gen.pt"
KO_SUFFIX = " 한국어로 간단히 답하세요."
NMAX = int(os.environ.get("LAT_NMAX", 320))
N_QUERIES = int(os.environ.get("LAT_QUERIES", 20))
WARMUP = int(os.environ.get("LAT_WARMUP", 2))
MAX_NEW, TOP_BLOCK, BATCH_MAX = 128, 5, 8
THR_REPS = int(os.environ.get("LAT_THR_REPS", 3))
COMPRESS_CHUNK = 64       # documents per compressor call (online and cache building); the official compress keeps
                          # all hidden states and full logits, ~6 GB per 64 x 131-token chunk
N_POOL = max(N_QUERIES + WARMUP, BATCH_MAX * THR_REPS)   # queries prepared (latency: first WARMUP + N_QUERIES; throughput: first BATCH_MAX * THR_REPS)


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stats(xs):
    xs = sorted(xs)
    return {"median": round(statistics.median(xs), 5), "p90": round(xs[int(0.9 * (len(xs) - 1))], 5),
            "mean": round(statistics.fmean(xs), 5), "std": round(statistics.stdev(xs), 5) if len(xs) > 1 else 0.0,
            "n": len(xs)}


def cmd_prepare():
    """Top-NMAX unique chunks for N_POOL KISTI questions (fixed seed) from the v2 index."""
    sys.argv += ["--index", "v2"]
    m120 = load("m120", "26_kisti_prepare_m120.py")
    import torch
    from sentence_transformers import SentenceTransformer
    queries = json.loads(m120.child125.QUERY_FILE.read_text())
    picked = random.Random(0).sample(queries, N_POOL)
    vecs, rows = m120.load_index()
    mat = torch.from_numpy(vecs).to("cuda")
    del vecs
    model = SentenceTransformer(m120.child125.E5_DIR, device="cuda").half()
    qvec = model.encode([q["query"] for q in picked], normalize_embeddings=True, convert_to_tensor=True)
    _, top_idx = (qvec.to(mat.dtype) @ mat.T).topk(2 * NMAX, dim=1)
    out = []
    for q, idx in zip(picked, top_idx.cpu().tolist()):
        docs, seen = [], set()
        for j in idx:
            if rows[j]["text"] not in seen:
                seen.add(rows[j]["text"])
                docs.append({"doc_id": rows[j]["chunk_id"], "text": rows[j]["text"], "n_tokens": rows[j]["n_tokens"]})
            if len(docs) == NMAX:
                break
        if len(docs) < NMAX:
            sys.exit(f"{q['query_id']}: fewer than {NMAX} unique chunks")
        out.append({"sample_id": q["query_id"], "question": q["query"] + KO_SUFFIX, "documents": docs})
    DOCS_FILE.write_text(json.dumps({"index": str(m120.INDEX_DIR), "warmup": WARMUP, "queries": out}, ensure_ascii=False))
    print(f"wrote {DOCS_FILE}: {len(out)} queries x {NMAX} docs")


class StepClock:
    """StoppingCriteria that records a synchronized timestamp after every decoding step."""
    def __init__(self, torch):
        self.torch, self.times = torch, []

    def __call__(self, input_ids, scores, **kwargs):
        self.torch.cuda.synchronize()
        self.times.append(time.perf_counter())
        return self.torch.zeros(input_ids.size(0), dtype=self.torch.bool, device=input_ids.device)


def timed_generate(torch, generate_fn):
    """generate_fn(stopping_criteria) -> (t_first_token, ttit, t_total) measured from the call."""
    from transformers import StoppingCriteriaList
    clock = StepClock(torch)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    generate_fn(StoppingCriteriaList([clock]))
    torch.cuda.synchronize()
    t_end = time.perf_counter()
    if len(clock.times) != MAX_NEW:
        raise RuntimeError(f"expected {MAX_NEW} decoding steps, got {len(clock.times)}")
    return clock.times[0] - t0, (clock.times[-1] - clock.times[0]) / (MAX_NEW - 1), t_end - t0


def oom_batches(torch, fn, items):
    """Run fn(batch_items) with B = BATCH_MAX, halving on OOM; -> (B, result)."""
    b = BATCH_MAX
    while b >= 1:
        try:
            return b, fn(items[:b])
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            b //= 2
    raise RuntimeError("OOM at batch 1")


def cmd_run(ns, out):
    import torch
    torch.set_num_threads(CPU_THREADS)
    from pisco_research import raw as raw_mod
    from pisco_research import reference as ref_mod
    from pisco_research.excerpt_gen import add_excerpt_adapter, generate_block
    data = json.loads(DOCS_FILE.read_text())
    queries = data["queries"]
    sync = torch.cuda.synchronize
    result = {"gpu": torch.cuda.get_device_name(0), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "dtype": "bfloat16", "max_new_tokens": MAX_NEW, "n_queries": N_QUERIES, "warmup": WARMUP,
              "top_block": TOP_BLOCK, "generator": str(GEN.relative_to(PROJECT_ROOT)), "ns": {}}

    # ---------------------------------------------------------------- raw
    ref = json.loads((PROJECT_ROOT / "reference" / "manifest.json").read_text())
    raw = raw_mod.RawBackbone(ref["snapshots"]["backbone"]["local_dir"])
    rtok = raw.tokenizer
    if rtok.pad_token is None:
        rtok.pad_token = rtok.bos_token   # batch padding only (masked); same id as the generate pad_token_id below

    def raw_inputs(qs, n):
        prompts = [raw.build_prompt(q["question"], [d["text"] for d in q["documents"][:n]]) for q in qs]
        enc = rtok(prompts, return_tensors="pt", padding="longest", add_special_tokens=False)
        if enc["input_ids"].size(1) > raw_mod.MAX_CONTEXT_TOKENS:
            raise ValueError(f"raw prompt {enc['input_ids'].size(1)} tokens > {raw_mod.MAX_CONTEXT_TOKENS}")
        return enc["input_ids"].to("cuda"), enc["attention_mask"].to("cuda")

    def raw_generate(ids, mask):
        return lambda sc: raw.model.generate(input_ids=ids, attention_mask=mask, do_sample=False, max_new_tokens=MAX_NEW,
                                             min_new_tokens=MAX_NEW, pad_token_id=rtok.bos_token_id, stopping_criteria=sc)

    for n in ns:
        rows = []
        for i, q in enumerate(queries[:WARMUP + N_QUERIES]):
            ids, mask = raw_inputs([q], n)
            with torch.no_grad():
                t_first, ttit, _ = timed_generate(torch, raw_generate(ids, mask))
            if i >= WARMUP:
                rows.append({"ttft": t_first, "ttit": ttit, "input_tokens": ids.size(1)})

        def raw_thr(batch):
            ids, mask = raw_inputs(batch, n)
            with torch.no_grad():
                _, _, total = timed_generate(torch, raw_generate(ids, mask))
            return len(batch) * MAX_NEW / total
        thr = run_throughput(torch, raw_thr, queries)
        result["ns"].setdefault(str(n), {})["raw"] = summarize_rows(rows, thr)
        print(n, "raw", brief(result["ns"][str(n)]["raw"]), flush=True)
    del raw
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------- PISCO (ours / online)
    ref_mod.DECODER_MAX_LENGTH = 8192   # timing only: N*8 memory slots exceed the official 2048 truncation limit
    step03 = load("step03", "03_load_reference.py")
    import fcntl
    with open(PROJECT_ROOT / "artifacts" / "step03" / "load.lock", "w") as lock:   # the loader rebuilds runtime_model/
        fcntl.flock(lock, fcntl.LOCK_EX)
        model, _ = step03.load_reference_model({})
        fcntl.flock(lock, fcntl.LOCK_UN)
    wrapper = ref_mod.PiscoReference(model)
    tok = wrapper.tokenizer
    add_excerpt_adapter(model)
    c = torch.load(GEN, map_location="cpu", weights_only=False)
    model.decoder.load_state_dict(c["state_dict"], strict=False)
    gen_max_len = c["input_max_len"]
    model.decoder.eval()
    dev = model.decoder.device
    nmax = max(ns)

    def compress_ids(enc_list):
        """Pre-tokenized encoder inputs (chunks) -> memory [sum, 8, H]; compressor forward only."""
        mems = []
        with torch.no_grad():
            for ids, mask in enc_list:
                mems.append(model.compress(ids, mask))
        return torch.cat(mems)

    def encoder_chunks(texts):
        out = []
        for i in range(0, len(texts), COMPRESS_CHUNK):
            enc = wrapper.prepare_encoder_inputs(texts[i:i + COMPRESS_CHUNK])
            out.append((enc["input_ids"].to(dev), enc["attention_mask"].to(dev)))
        return out

    # memory cache for `ours`: computed once per query for the largest N, kept on CPU (not timed)
    cache = {}
    for q in queries[:WARMUP + N_QUERIES] + queries[:BATCH_MAX * THR_REPS]:
        if q["sample_id"] not in cache:
            cache[q["sample_id"]] = compress_ids(encoder_chunks([d["text"] for d in q["documents"][:nmax]])).cpu()

    def pisco_pass(batch, n, online, enc_pre=None):
        """-> (pre_seconds, decoder generate fn). enc_pre: pre-tokenized encoder chunks per query (online)."""
        sync(); t0 = time.perf_counter()
        if online:
            mems = [compress_ids(e) for e in enc_pre]
        else:
            mems = [cache[q["sample_id"]][:n].to(dev) for q in batch]
        sync(); t_m = time.perf_counter() - t0
        texts = ["\n".join(d["text"] for d in q["documents"][:TOP_BLOCK]) for q in batch]
        with torch.no_grad():
            blocks = generate_block(model, tok, [q["question"] for q in batch], texts, None, gen_max_len)
        mem_list = [torch.cat([m, b[None].to(m.dtype)], dim=0) for m, b in zip(mems, blocks)]
        sync(); pre = time.perf_counter() - t0
        pisco_pass.parts = {"memory": t_m, "block": pre - t_m}   # transfer (ours) or compression (online); block
        dec = wrapper.build_decoder_inputs([q["question"] for q in batch], mem_list)   # tokenization: not timed
        if "decoder_adapter" in model.adapter_keys:
            model.decoder.set_adapter("decoder_adapter")
        fn = lambda sc: model.decoder.generate(inputs_embeds=dec.inputs_embeds, attention_mask=dec.attention_mask,
                                               generation_config=model.generation_config, max_new_tokens=MAX_NEW,
                                               min_new_tokens=MAX_NEW, stopping_criteria=sc)
        return pre, fn, dec.input_ids.size(1)

    for n in ns:
        for cond in ("ours", "online"):
            online = cond == "online"
            rows = []
            for i, q in enumerate(queries[:WARMUP + N_QUERIES]):
                enc_pre = [encoder_chunks([d["text"] for d in q["documents"][:n]])] if online else None
                pre, fn, dec_len = pisco_pass([q], n, online, enc_pre)
                with torch.no_grad():
                    t_first, ttit, _ = timed_generate(torch, fn)
                if i >= WARMUP:
                    rows.append({"ttft": pre + t_first, "ttit": ttit, "pre": pre, "memory": pisco_pass.parts["memory"],
                                 "block": pisco_pass.parts["block"], "prefill_first_token": t_first,
                                 "decoder_input_tokens": dec_len})

            def thr_fn(batch):
                enc_pre = [encoder_chunks([d["text"] for d in q["documents"][:n]]) for q in batch] if online else None
                pre, fn, _ = pisco_pass(batch, n, online, enc_pre)
                with torch.no_grad():
                    _, _, total = timed_generate(torch, fn)
                return len(batch) * MAX_NEW / (pre + total)
            thr = run_throughput(torch, thr_fn, queries)
            result["ns"][str(n)][cond] = summarize_rows(rows, thr)
            print(n, cond, brief(result["ns"][str(n)][cond]), flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))   # partial results survive a later failure
    print("written:", out)


def run_throughput(torch, fn, queries):
    """Warm-up once, then THR_REPS batches of disjoint queries at the largest batch that fits."""
    b, _ = oom_batches(torch, fn, queries[:BATCH_MAX])
    vals = [fn(queries[r * b:(r + 1) * b]) for r in range(THR_REPS)]
    return {"batch": b, "tokens_per_s": stats(vals)}


def summarize_rows(rows, thr):
    return {**{k: stats([r[k] for r in rows]) for k in rows[0]}, "throughput": thr}


def brief(d):
    return {"ttft": d["ttft"]["median"], "ttit": d["ttit"]["median"], "thr": d["throughput"]["tokens_per_s"]["median"],
            "batch": d["throughput"]["batch"]}


def cmd_summarize():
    merged, gpus = {}, {}
    for f in sorted(OUT_DIR.glob("run_*.json")):
        r = json.loads(f.read_text())
        for n, conds in r["ns"].items():
            if n in merged:
                sys.exit(f"N={n} appears in more than one run file")
            merged[n] = conds
            gpus[n] = f"{r['gpu']} (CUDA {r['cuda_visible_devices']})"
    lines = ["n_docs,raw_input_tokens,method,ttft_s,ttit_s,throughput_tok_s,batch,ttft_accel,ttit_accel,throughput_accel"]
    out = {}
    for n in sorted(merged, key=int):
        c = merged[n]
        raw = c["raw"]
        tokens = raw["input_tokens"]["median"]
        out[n] = {"raw_input_tokens_median": tokens, "gpu": gpus[n], "conditions": c, "acceleration": {}}
        for m in ("raw", "ours", "online"):
            if m not in c:
                continue
            ttft, ttit = c[m]["ttft"]["median"], c[m]["ttit"]["median"]
            thr = c[m]["throughput"]["tokens_per_s"]["median"]
            acc = {"ttft": raw["ttft"]["median"] / ttft, "ttit": raw["ttit"]["median"] / ttit,
                   "throughput": thr / raw["throughput"]["tokens_per_s"]["median"]}
            out[n]["acceleration"][m] = {k: round(v, 3) for k, v in acc.items()}
            lines.append(f"{n},{tokens},{m},{ttft:.4f},{ttit:.5f},{thr:.2f},{c[m]['throughput']['batch']},"
                         f"{acc['ttft']:.3f},{acc['ttit']:.3f},{acc['throughput']:.3f}")
    (OUT_DIR / "results.json").write_text(json.dumps(out, indent=2))
    (OUT_DIR / "curves.csv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "run", "summarize"])
    ap.add_argument("--ns", type=int, nargs="+")
    ap.add_argument("--index", default=None)   # consumed by the Step 26 module during `prepare`
    a = ap.parse_args()
    if a.cmd == "prepare":
        cmd_prepare()
    elif a.cmd == "run":
        if os.environ.get("CUDA_VISIBLE_DEVICES") in (None, "") or "," in os.environ["CUDA_VISIBLE_DEVICES"]:
            sys.exit("Set CUDA_VISIBLE_DEVICES to exactly one assigned GPU.")
        cmd_run(a.ns, OUT_DIR / (f"{TAG}/run_{'_'.join(map(str, a.ns))}.json" if TAG else f"run_{'_'.join(map(str, a.ns))}.json"))
    else:
        cmd_summarize()
