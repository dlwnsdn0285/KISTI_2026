"""Step 08: common input manifest -> predictions (PISCO-B / Raw) -> one shared scorer.

    PY=~/anaconda3/envs/pisco-ref/bin/python
    $PY scripts/08_run_eval.py build-manifest                          # CPU, tokenizer only
    CUDA_VISIBLE_DEVICES=0 $PY scripts/08_run_eval.py predict --system pisco
    CUDA_VISIBLE_DEVICES=0 $PY scripts/08_run_eval.py predict --system raw
    CUDA_VISIBLE_DEVICES=0 $PY scripts/08_run_eval.py predict --system official   # path A (Step 09)
    $PY scripts/08_run_eval.py score                                   # latest run of each system

predict/score take --manifest PATH (default: the synthetic manifest).

Systems read only `model_input` of each manifest sample. Each predict call
writes a new run directory (never overwrites): run.json + predictions.jsonl.
"""

import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

# Shared server: the model runs on the GPU, so CPU threads only serve loading and
# tokenization. Cap them (torch defaults to 32 per process here). Must be set
# before torch is imported; override with PISCO_CPU_THREADS.
CPU_THREADS = int(os.environ.get("PISCO_CPU_THREADS", "4"))
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, str(CPU_THREADS))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pisco_research import eval_inputs, scoring  # noqa: E402

FIXTURE_PATH = PROJECT_ROOT / "fixtures" / "synthetic_qa.json"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "synthetic_v1.json"
RUNS_DIR = PROJECT_ROOT / "artifacts" / "step08" / "runs"
REFERENCE_MANIFEST = PROJECT_ROOT / "reference" / "manifest.json"
MAX_NEW_TOKENS = 128
# Not part of the evaluation set: only exercises the truncation path (> 128 tokens).
LONG_PROBE_TEXT = " ".join(f"Stone {i} of the northern wall was laid in the year {1700 + i}." for i in range(40))


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def backbone_dir():
    return json.loads(REFERENCE_MANIFEST.read_text())["snapshots"]["backbone"]["local_dir"]


def import_script(filename, name):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------ build-manifest

def cmd_build_manifest():
    os.environ["HF_HUB_OFFLINE"] = "1"
    ref = json.loads(REFERENCE_MANIFEST.read_text())
    tok = eval_inputs.load_pisco_tokenizer(backbone_dir())
    source = {"repo_id": ref["backbone"]["repo_id"], "revision": ref["backbone"]["revision_sha"],
              "added_special_tokens": eval_inputs.PISCO_ADDED_TOKENS}
    manifest = eval_inputs.build_manifest(FIXTURE_PATH, tok, source, "synthetic_v1")

    _, probe_ids, probe_truncated = eval_inputs.pisco_encoder_span(tok, LONG_PROBE_TEXT)
    probe_used = tok.decode(probe_ids)
    manifest["truncation_probe"] = {
        "n_tokens_standalone": len(tok(LONG_PROBE_TEXT, add_special_tokens=False)["input_ids"]),
        "n_tokens_used_by_pisco": len(probe_ids), "truncated": probe_truncated,
        "used_text_is_prefix_of_original": LONG_PROBE_TEXT.startswith(probe_used),
        "used_text_tail": probe_used[-60:]}

    if MANIFEST_PATH.exists():
        old = json.loads(MANIFEST_PATH.read_text())
        if eval_inputs.sha256_json(old["samples"]) != eval_inputs.sha256_json(manifest["samples"]):
            sys.exit(f"{MANIFEST_PATH} exists with different samples; refusing to overwrite (new manifest_id needed)")
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    for s in manifest["samples"]:
        print(s["sample_id"], s["model_input_sha256"][:16], [p["n_tokens_used_by_pisco"] for p in s["provenance"]["documents"]],
              "truncated:", [p["truncated"] for p in s["provenance"]["documents"]])
    print("truncation probe:", manifest["truncation_probe"])


# ------------------------------------------------------------------- predict

def new_run_dir(system, manifest):
    run_id = f"{system}-{manifest['manifest_id']}-{datetime.datetime.now().strftime('%Y%m%dT%H%M%S')}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)  # a run id is never reused
    return run_id, run_dir


def verify_pisco_encoder_ids(wrapper, manifest):
    """The manifest's expected encoder ids must equal what the real PISCO model builds."""
    mismatches = []
    for s in manifest["samples"]:
        docs = s["model_input"]["documents"]
        enc = wrapper.prepare_encoder_inputs([d["text"] for d in docs])
        for d, row, prov in zip(docs, enc["input_ids"].tolist(), s["provenance"]["documents"]):
            if eval_inputs.sha256_json(row) != prov["pisco_encoder_ids_sha256"]:
                mismatches.append(d["doc_id"])
    expected, _, _ = eval_inputs.pisco_encoder_span(wrapper.tokenizer, LONG_PROBE_TEXT)
    probe_equal = wrapper.prepare_encoder_inputs([LONG_PROBE_TEXT])["input_ids"][0].tolist() == expected
    return {"doc_mismatches": mismatches, "truncation_probe_equal": probe_equal}


def predict_pisco(manifest, run_meta):
    from pisco_research.reference import COMPRESSION_POLICY, PiscoReference
    step03 = import_script("03_load_reference.py", "step03")
    run_meta["load"] = {}
    model, _ = step03.load_reference_model(run_meta["load"])
    wrapper = PiscoReference(model)
    run_meta["system_config"] = {"path": "B (Research Base wrapper)", "compression_policy": COMPRESSION_POLICY,
                                 "generation": "official generation_config + decoder eos/pad fallback (greedy)"}
    run_meta["encoder_id_verification"] = verify_pisco_encoder_ids(wrapper, manifest)
    if run_meta["encoder_id_verification"]["doc_mismatches"] or not run_meta["encoder_id_verification"]["truncation_probe_equal"]:
        raise RuntimeError(f"manifest/PISCO encoder mismatch: {run_meta['encoder_id_verification']}")

    def run_one(mi):
        res = wrapper.answer([mi["question"]], [[d["text"] for d in mi["documents"]]], max_new_tokens=MAX_NEW_TOKENS)
        ids = res["output"].output_ids[0].cpu().tolist()
        dec = res["decoder_inputs"]
        return {"prediction": res["output"].texts[0], "output_ids": ids,
                "decoder_input_length": dec.input_ids.size(1),
                "total_memory_vectors": sum(m.size(0) * m.size(1) for m in res["memory"])}
    return run_one


def predict_official(manifest, run_meta):
    """Path A: the official HF API called directly, one question (its k docs) per call."""
    import torch
    step03 = import_script("03_load_reference.py", "step03")
    step06 = import_script("06_check_consistency.py", "step06")  # observe(): records real output IDs
    run_meta["load"] = {}
    model, _ = step03.load_reference_model(run_meta["load"])
    run_meta["system_config"] = {"path": "A (official HF API)",
                                 "calls": "compress_documents(k docs) + generate_from_compressed_documents_and_questions",
                                 "compression_policy": "per_question (one question per call)"}

    def run_one(mi):
        with torch.no_grad(), step06.observe(model) as store:
            memory = model.compress_documents([d["text"] for d in mi["documents"]])
            texts = model.generate_from_compressed_documents_and_questions(
                questions=[mi["question"]], compressed_documents=memory, max_new_tokens=MAX_NEW_TOKENS)
        gen = store["generate"][0]
        return {"prediction": texts[0], "output_ids": gen["output_ids"][0].tolist(),
                "decoder_input_length": gen["inputs_embeds"].size(1)}
    return run_one


def predict_raw(manifest, run_meta):
    from pisco_research.raw import RAW_DOC_SEPARATOR, SYSTEM_PROMPT, RawBackbone
    ref = json.loads(REFERENCE_MANIFEST.read_text())
    pinned_source = (PROJECT_ROOT / ref["files"]["modelling_pisco.py"]["local_path"]).read_text()
    if SYSTEM_PROMPT not in pinned_source:
        raise RuntimeError("Raw system prompt differs from the pinned PISCO source")
    t0 = time.perf_counter()
    raw = RawBackbone(backbone_dir())
    import torch
    run_meta["load"] = {"backbone_dir": backbone_dir(), "load_seconds": round(time.perf_counter() - t0, 1),
                        "dtype": str(raw.model.dtype), "attn_implementation": raw.model.config._attn_implementation,
                        "vocab_size": raw.model.get_input_embeddings().weight.shape[0],
                        "gpu": torch.cuda.get_device_name(0)}
    run_meta["system_config"] = {"path": "Raw (original backbone)", "doc_rendering": f"used_text joined by {RAW_DOC_SEPARATOR!r}",
                                 "generation": "greedy, do_sample=False, pad=bos(1), eos from backbone generation_config"}

    def run_one(mi):
        out = raw.generate(mi["question"], [d["used_text"] for d in mi["documents"]], max_new_tokens=MAX_NEW_TOKENS)
        return {"prediction": out["text"], "output_ids": out["output_ids"].tolist(),
                "decoder_input_length": out["input_length"], "prompt": out["prompt"]}
    return run_one


def predict_fusion(manifest, run_meta, checkpoint):
    """Path B + fusion module (Step 11): memory -> fusion(memory, question) -> decoder."""
    import torch
    from pisco_research.fusion import FusionModule, fuse
    from pisco_research.reference import COMPRESSION_POLICY, PiscoReference
    step03 = import_script("03_load_reference.py", "step03")
    run_meta["load"] = {}
    model, _ = step03.load_reference_model(run_meta["load"])
    wrapper = PiscoReference(model)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if run_meta.get("decoder_lora") and ckpt.get("kind") != "excerptgen":
        raise RuntimeError("--decoder-lora is supported only with an excerpt-generator checkpoint")
    appender, excerpt_gen = None, None
    if ckpt.get("kind") == "excerptgen":
        # Step 17: last document of the manifest is a raw excerpt; a dedicated generator turns it into one block
        from pisco_research.excerpt_gen import MemProj, add_excerpt_adapter, generate_block_res
        add_excerpt_adapter(model)
        missing, unexpected = model.decoder.load_state_dict(ckpt["state_dict"], strict=False)
        if unexpected:
            raise RuntimeError(f"excerpt generator did not map onto the decoder: {unexpected[:3]}")
        model.decoder.eval()
        excerpt_gen, fusion, extra = ckpt["variant"], None, ckpt.get("extra", {})
        # Step 19: full-document generators read the sixth document's full text (all five used_texts)
        gen_field = "text" if ckpt.get("input") == "full" else "used_text"
        gen_max_len = ckpt.get("input_max_len", 128)
        memproj = None
        if ckpt.get("memproj"):          # Step 18 residual generator: affine map on the memory prefix
            memproj = MemProj(wrapper.hidden_size).to(model.decoder.device)
            memproj.load_state_dict(ckpt["memproj"])
            memproj.eval()
        run_meta["system_config"] = {"path": f"B + excerpt generator ({ckpt['variant']})", "compression_policy": COMPRESSION_POLICY,
                                     "checkpoint": str(Path(checkpoint).resolve()), "checkpoint_sha256": sha256_file(checkpoint),
                                     "gen_extra": extra, "n_tensors_loaded": len(ckpt["state_dict"]),
                                     "generation": "official generation_config + decoder eos/pad fallback (greedy)"}
        if run_meta.get("decoder_lora"):
            # Step 32: a decoder_adapter LoRA trained over [memories, frozen block] (Step 23 D_Q / D_R)
            lp = Path(run_meta["decoder_lora"])
            lora = torch.load(lp, map_location="cpu", weights_only=False)
            if lora.get("kind") != "lora" or lora.get("adapter") != "decoder_adapter":
                raise RuntimeError(f"{lp} is not a decoder_adapter LoRA checkpoint")
            missing, unexpected = model.decoder.load_state_dict(lora["state_dict"], strict=False)
            if unexpected:
                raise RuntimeError(f"decoder LoRA did not map onto the decoder: {unexpected[:3]}")
            model.decoder.eval()
            run_meta["system_config"].update({"path": f"B + excerpt generator ({ckpt['variant']}) + decoder LoRA",
                                              "decoder_lora": str(lp.resolve()), "decoder_lora_sha256": sha256_file(lp),
                                              "decoder_lora_frozen_block": lora.get("frozen_block"),
                                              "decoder_lora_tensors": len(lora["state_dict"])})
    elif ckpt.get("kind") in ("append", "const"):
        # Step 13: the five memories unchanged + one appended block (k+1 slots)
        from pisco_research.fusion import load_append
        appender, extra = load_append(checkpoint, device="cuda")
        appender.eval()
        fusion = None
        lora_path = Path(checkpoint).with_name("lora.pt")
        if extra.get("lora") and lora_path.exists():      # Step 14 joint training: block + decoder LoRA
            lora = torch.load(lora_path, map_location="cpu", weights_only=False)
            missing, unexpected = model.decoder.load_state_dict(lora["state_dict"], strict=False)
            if unexpected:
                raise RuntimeError(f"joint LoRA did not map onto the decoder: {unexpected[:3]}")
            model.decoder.eval()
            extra = {**extra, "lora_loaded": len(lora["state_dict"]), "lora_sha256": sha256_file(lora_path)}
        run_meta["system_config"] = {"path": f"B + appended block ({ckpt['kind']})", "compression_policy": COMPRESSION_POLICY,
                                     "checkpoint": str(Path(checkpoint).resolve()), "checkpoint_sha256": sha256_file(checkpoint),
                                     "fusion_config": appender.cfg.__dict__, "fusion_extra": extra, "slots": "k + 1",
                                     "generation": "official generation_config + decoder eos/pad fallback (greedy)"}
    elif ckpt.get("kind") == "lora":
        # Step 12 control: one of PISCO's own LoRA adapters fine-tuned; no fusion module
        missing, unexpected = model.decoder.load_state_dict(ckpt["state_dict"], strict=False)
        if unexpected or len(missing) != len(dict(model.decoder.named_parameters())) - len(ckpt["state_dict"]):
            raise RuntimeError(f"LoRA checkpoint did not map onto the decoder: unexpected={unexpected[:3]}")
        model.decoder.eval()
        fusion, extra = None, ckpt.get("extra", {})
        run_meta["system_config"] = {"path": f"B + fine-tuned {ckpt['adapter']}", "compression_policy": COMPRESSION_POLICY,
                                     "checkpoint": str(Path(checkpoint).resolve()), "checkpoint_sha256": sha256_file(checkpoint),
                                     "lora_extra": extra, "n_tensors_loaded": len(ckpt["state_dict"]),
                                     "generation": "official generation_config + decoder eos/pad fallback (greedy)"}
    else:
        fusion, extra = FusionModule.load(checkpoint, device="cuda")
        fusion.eval()
        run_meta["system_config"] = {"path": "B + fusion", "compression_policy": COMPRESSION_POLICY,
                                     "checkpoint": str(Path(checkpoint).resolve()), "checkpoint_sha256": sha256_file(checkpoint),
                                     "fusion_config": fusion.cfg.__dict__, "fusion_extra": extra,
                                     "generation": "official generation_config + decoder eos/pad fallback (greedy)"}
    run_meta["encoder_id_verification"] = verify_pisco_encoder_ids(wrapper, manifest)
    if run_meta["encoder_id_verification"]["doc_mismatches"] or not run_meta["encoder_id_verification"]["truncation_probe_equal"]:
        raise RuntimeError(f"manifest/PISCO encoder mismatch: {run_meta['encoder_id_verification']}")

    def run_one(mi):
        if excerpt_gen is not None and any("#blk" in d["doc_id"] for d in mi["documents"]):
            # Step 30: several generator inputs (doc ids "...#blk<j>") interleaved with the documents; each becomes one
            # block placed at its own position, the other documents are compressed in one call as usual
            if excerpt_gen == "textmem":
                raise RuntimeError("multi-block manifests support the text generator variant only")
            docs = mi["documents"]
            is_blk = ["#blk" in d["doc_id"] for d in docs]
            memory, _ = wrapper.compress([d["text"] for d, b in zip(docs, is_blk) if not b])
            blk_texts = [d[gen_field] for d, b in zip(docs, is_blk) if b]
            with torch.no_grad():
                blocks = generate_block_res(model, wrapper.tokenizer, [mi["question"]] * len(blk_texts), blk_texts,
                                            None, memproj, gen_max_len).to(memory.dtype)
            it_m, it_b = iter(memory), iter(blocks)
            fused = [torch.stack([next(it_b) if b else next(it_m) for b in is_blk])]
            dec = wrapper.build_decoder_inputs([mi["question"]], fused)
            out = wrapper.generate(dec, max_new_tokens=MAX_NEW_TOKENS)
            return {"prediction": out.texts[0], "output_ids": out.output_ids[0].cpu().tolist(),
                    "decoder_input_length": dec.input_ids.size(1), "n_blocks": len(blk_texts)}
        if excerpt_gen is not None:
            docs = mi["documents"]
            memory, _ = wrapper.compress([d["text"] for d in docs[:-1]])
            with torch.no_grad():
                block = generate_block_res(model, wrapper.tokenizer, [mi["question"]], [docs[-1][gen_field]],
                                           memory[None] if excerpt_gen == "textmem" else None, memproj, gen_max_len)
            fused = [torch.cat([memory, block[0].to(memory.dtype)[None]], dim=0)]
            dec = wrapper.build_decoder_inputs([mi["question"]], fused)
            out = wrapper.generate(dec, max_new_tokens=MAX_NEW_TOKENS)
            return {"prediction": out.texts[0], "output_ids": out.output_ids[0].cpu().tolist(),
                    "decoder_input_length": dec.input_ids.size(1)}
        memory, _ = wrapper.compress([d["text"] for d in mi["documents"]])
        if appender is not None:
            from pisco_research.fusion import append_block
            fused = append_block(wrapper, appender, [mi["question"]], [memory])
        else:
            fused = fuse(wrapper, fusion, [mi["question"]], [memory]) if fusion is not None else [memory]
        dec = wrapper.build_decoder_inputs([mi["question"]], fused)
        out = wrapper.generate(dec, max_new_tokens=MAX_NEW_TOKENS)
        if appender is not None:
            delta = (fused[0][-1].float() - memory.float().mean(dim=0)).norm() / memory.float().mean(dim=0).norm()
        else:
            delta = (fused[0].float() - memory.float()).norm() / memory.float().norm()
        return {"prediction": out.texts[0], "output_ids": out.output_ids[0].cpu().tolist(),
                "decoder_input_length": dec.input_ids.size(1), "memory_rel_change": round(delta.item(), 4)}
    return run_one


def predict_hybrid(manifest, run_meta):
    """Step 16 hybrid: the first k-1 documents as PISCO memory, the LAST document (the excerpt) as raw text
    placed after the memory blocks (same construction as the Step 12 mixed-context probe)."""
    import torch
    from pisco_research.reference import COMPRESSION_POLICY, PiscoReference
    step03 = import_script("03_load_reference.py", "step03")
    run_meta["load"] = {}
    model, _ = step03.load_reference_model(run_meta["load"])
    wrapper = PiscoReference(model)
    tok = wrapper.tokenizer
    mem_str = "".join(tok.mem_tokens) + tok.sep_token
    run_meta["system_config"] = {"path": "B + raw excerpt (hybrid)", "compression_policy": COMPRESSION_POLICY,
                                 "layout": "k-1 memory blocks, then the last document's used_text as raw text + SEP, then the question",
                                 "generation": "official generation_config + decoder eos/pad fallback (greedy)"}

    def run_one(mi):
        docs = mi["documents"]
        k = len(docs) - 1
        memory, _ = wrapper.compress([d["text"] for d in docs[:k]])
        model.generation_top_k = k + 1
        template = model.blend_prompt_and_memory_tokens(query=mi["question"])
        prompt = template.replace(mem_str * (k + 1), mem_str * k + docs[-1]["used_text"] + tok.sep_token, 1)
        if prompt == template:
            raise RuntimeError("placeholder block not found")
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"].to(model.decoder.device)
        row = ids[0].tolist()
        positions = [i for i, t in enumerate(row) if t in tok.mem_token_ids]
        if len(positions) != k * wrapper.n_mem:
            raise RuntimeError("unexpected number of MEM placeholders")
        with torch.no_grad():
            embeds = model.decoder.get_input_embeddings()(ids).clone()
            for j in range(k):
                first = positions[j * wrapper.n_mem]
                embeds[0, first:first + wrapper.n_mem] = memory[j]
            if "decoder_adapter" in model.adapter_keys:
                model.decoder.set_adapter("decoder_adapter")
            out = model.decoder.generate(inputs_embeds=embeds, attention_mask=enc["attention_mask"].to(ids.device),
                                         generation_config=model.generation_config, max_new_tokens=MAX_NEW_TOKENS)
        return {"prediction": tok.batch_decode(out, skip_special_tokens=True)[0], "output_ids": out[0].cpu().tolist(),
                "decoder_input_length": embeds.size(1)}
    return run_one


PREDICTORS = {"official": predict_official, "pisco": predict_pisco, "raw": predict_raw, "fusion": predict_fusion,
              "hybrid": predict_hybrid}


def cmd_predict(system, manifest_path, checkpoint=None, decoder_lora=None):
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    torch.set_num_threads(CPU_THREADS)
    if os.environ.get("CUDA_VISIBLE_DEVICES") in (None, "") or "," in os.environ["CUDA_VISIBLE_DEVICES"]:
        sys.exit("Set CUDA_VISIBLE_DEVICES to exactly one assigned GPU (e.g. 0).")
    manifest = json.loads(manifest_path.read_text())
    if (system == "fusion") != (checkpoint is not None):
        sys.exit("--checkpoint is required for --system fusion and not allowed otherwise")
    if decoder_lora is not None and system != "fusion":
        sys.exit("--decoder-lora is only allowed with --system fusion and an excerpt-generator checkpoint")
    run_prefix = system if checkpoint is None else f"fusion_{Path(checkpoint).resolve().parent.name}"
    if decoder_lora is not None:
        run_prefix += f"+{Path(decoder_lora).resolve().parent.name}"
    run_id, run_dir = new_run_dir(run_prefix, manifest)
    run_meta = {"run_id": run_id, "system": system, "started_at": now(),
                "manifest": {"path": str(manifest_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(manifest_path),
                             "manifest_id": manifest["manifest_id"]},
                "code_sha256": {str(p.relative_to(PROJECT_ROOT)): sha256_file(p) for p in
                                [Path(__file__), *sorted((PROJECT_ROOT / "src" / "pisco_research").glob("*.py"))]},
                "max_new_tokens": MAX_NEW_TOKENS, "batch_size": 1,
                **({"decoder_lora": str(decoder_lora)} if decoder_lora is not None else {}),
                "cpu_threads": {"torch": torch.get_num_threads(), "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS")}}
    run_one = PREDICTORS[system](manifest, run_meta, checkpoint) if checkpoint else PREDICTORS[system](manifest, run_meta)

    eos = 2
    records = []
    for s in manifest["samples"]:
        rec = {"run_id": run_id, "system": system, "sample_id": s["sample_id"],
               "model_input_sha256": eval_inputs.sha256_json(s["model_input"])}
        if rec["model_input_sha256"] != s["model_input_sha256"]:
            raise RuntimeError(f"{s['sample_id']}: model_input changed after manifest build")
        t0 = time.perf_counter()
        try:
            rec.update(run_one(s["model_input"]), status="ok")    # only model_input is passed
            rec["stopped_at_eos"] = eos in rec["output_ids"]
        except Exception as exc:
            rec.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        rec["seconds"] = round(time.perf_counter() - t0, 2)
        records.append(rec)
        print(f"[{system}] {s['sample_id']} {rec['status']}: {rec.get('prediction', rec.get('error'))!r}")

    with open(run_dir / "predictions.jsonl", "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    run_meta["counts"] = {"expected": len(manifest["samples"]), "attempted": len(records),
                          "succeeded": sum(r["status"] == "ok" for r in records),
                          "failed": sum(r["status"] == "failed" for r in records)}
    run_meta["finished_at"] = now()
    (run_dir / "run.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False))
    print("run:", run_dir.relative_to(PROJECT_ROOT), run_meta["counts"])


# --------------------------------------------------------------------- score

def latest_run(system, manifest_sha):
    runs = sorted(d for d in RUNS_DIR.glob(f"{system}-*") if (d / "run.json").exists())
    runs = [d for d in runs if json.loads((d / "run.json").read_text())["manifest"]["sha256"] == manifest_sha]
    return runs[-1] if runs else None


def ab_parity(manifest, preds_a, preds_b):
    """A (official) vs B (wrapper): per-sample equality of output IDs and strings."""
    a = {p["sample_id"]: p for p in preds_a}
    b = {p["sample_id"]: p for p in preds_b}
    rows = []
    for s in manifest["samples"]:
        pa, pb = a.get(s["sample_id"]), b.get(s["sample_id"])
        both_ok = pa is not None and pb is not None and pa["status"] == pb["status"] == "ok"
        rows.append({"sample_id": s["sample_id"], "both_ok": both_ok,
                     "output_ids_equal": both_ok and pa["output_ids"] == pb["output_ids"],
                     "prediction_equal": both_ok and pa["prediction"] == pb["prediction"]})
    return {"n_compared": sum(r["both_ok"] for r in rows), "n_output_ids_equal": sum(r["output_ids_equal"] for r in rows),
            "n_prediction_equal": sum(r["prediction_equal"] for r in rows),
            "mismatched_sample_ids": [r["sample_id"] for r in rows if not r["output_ids_equal"]]}


def breakdown(manifest, rows, key):
    """Match counts grouped by an annotation field (e.g. question type)."""
    groups = {}
    for s, r in zip(manifest["samples"], rows):
        g = groups.setdefault(str(s["annotation"].get(key)), {"n": 0, "matched": 0})
        g["n"] += 1
        g["matched"] += r["match"]
    return groups


def cmd_score(manifest_path):
    manifest = json.loads(manifest_path.read_text())
    manifest_sha = sha256_file(manifest_path)
    results, preds_by_system = {}, {}
    for system in PREDICTORS:
        run_dir = latest_run(system, manifest_sha)
        if run_dir is None:
            print(f"{system}: no completed run for this manifest")
            continue
        preds = [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text().splitlines()]
        preds_by_system[system] = preds
        results[system] = {"run_id": run_dir.name, **scoring.score_run(manifest, preds)}

    table = []
    for s in manifest["samples"]:
        row = {"sample_id": s["sample_id"], "reference_answers": s["evaluation"]["reference_answers"]}
        for system, res in results.items():
            r = next(x for x in res["rows"] if x["sample_id"] == s["sample_id"])
            row[system] = {"match": r["match"], "status": r["status"], "prediction": r.get("prediction")}
        table.append(row)
    out = {"scored_at": now(), "manifest": str(manifest_path.relative_to(PROJECT_ROOT)), "manifest_sha256": manifest_sha,
           "metric": "Match (normalized reference substring)",
           "summaries": {k: {"run_id": v["run_id"], **v["summary"]} for k, v in results.items()},
           "by_question_type": {k: breakdown(manifest, v["rows"], "question_type") for k, v in results.items()},
           "by_all_evidence_preserved": {k: breakdown(manifest, v["rows"], "all_evidence_preserved") for k, v in results.items()},
           "ab_parity": ab_parity(manifest, preds_by_system["official"], preds_by_system["pisco"])
           if {"official", "pisco"} <= set(preds_by_system) else None,
           "per_sample": table}
    path = RUNS_DIR.parent / f"score_{manifest['manifest_id']}_{datetime.datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({k: out[k] for k in ("manifest", "summaries", "by_question_type", "by_all_evidence_preserved", "ab_parity")},
                     indent=2, ensure_ascii=False))
    print("written:", path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-manifest")
    p = sub.add_parser("predict")
    p.add_argument("--system", choices=list(PREDICTORS), required=True)
    p.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    p.add_argument("--checkpoint", type=Path, default=None, help="fusion checkpoint (system fusion only)")
    p.add_argument("--decoder-lora", type=Path, default=None,
                   help="decoder_adapter LoRA used with an excerpt-generator checkpoint (Step 23 D_Q / D_R)")
    s = sub.add_parser("score")
    s.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    args = parser.parse_args()
    if args.cmd == "build-manifest":
        cmd_build_manifest()
    elif args.cmd == "predict":
        cmd_predict(args.system, args.manifest.resolve(), args.checkpoint, args.decoder_lora)
    else:
        cmd_score(args.manifest.resolve())
