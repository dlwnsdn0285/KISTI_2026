"""Step 03: fetch the pinned checkpoint files and load the official PISCO model.

Two stages, run separately:

  fetch  Download only the files the active path needs, at the pinned
         revisions, into the default HF cache. Every file is checked against
         the remote hash (LFS sha256, or git blob sha1 for small files).
      ~/anaconda3/envs/pisco-ref/bin/python scripts/03_load_reference.py fetch

  load   Offline load on ONE assigned GPU, then verify what ended up where.
         No forward / compression / generation / backward.
      CUDA_VISIBLE_DEVICES=0 ~/anaconda3/envs/pisco-ref/bin/python scripts/03_load_reference.py load

Why a runtime directory: the pinned loader ignores `revision` for every nested
Hub call (see notes/hf_trace.md §7). We therefore hand it a local directory
whose config points the backbone at the pinned local snapshot. Only
`decoder_model_name` differs from the original config; the source file is a
byte-identical copy of reference/upstream.
"""

import argparse
import contextlib
import datetime
import hashlib
import io
import json
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "reference" / "manifest.json"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "step03"
RUNTIME_DIR = OUTPUT_DIR / "runtime_model"

FETCH_PATTERNS = {
    # adapters + trained embed/lm_head, plus text files (re-verified against reference/upstream)
    "pisco": ["*.pth", "config.json", "modelling_pisco.py", "__init__.py", "README.md"],
    # safetensors only (the repo also has duplicate .bin weights)
    "backbone": ["config.json", "generation_config.json", "model-*.safetensors",
                 "model.safetensors.index.json", "tokenizer.json", "tokenizer.model",
                 "tokenizer_config.json", "special_tokens_map.json"],
}
EXPECTED_ADAPTERS = {"encoder_adapter", "decoder_adapter"}


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path, chunk=1 << 24):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path):
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


# ---------------------------------------------------------------- fetch stage

def fetch():
    from huggingface_hub import HfApi, snapshot_download

    manifest = json.loads(MANIFEST_PATH.read_text())
    pins = {"pisco": (manifest["repo_id"], manifest["revision_sha"]),
            "backbone": (manifest["backbone"]["repo_id"], manifest["backbone"]["revision_sha"])}
    api = HfApi()
    snapshots = manifest.setdefault("snapshots", {})

    for role, (repo_id, sha) in pins.items():
        # token=False: both repos are public; do not send stored credentials.
        local_dir = Path(snapshot_download(repo_id, revision=sha,
                                           allow_patterns=FETCH_PATTERNS[role], token=False))
        info = api.model_info(repo_id, revision=sha, files_metadata=True, token=False)
        if info.sha != sha:
            raise RuntimeError(f"{repo_id}: API returned {info.sha}, expected {sha}")
        remote = {s.rfilename: s for s in info.siblings}

        files = {}
        for path in sorted(p for p in local_dir.rglob("*") if p.is_file()):
            name = str(path.relative_to(local_dir))
            sibling = remote[name]
            entry = {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            if sibling.lfs is not None:
                entry["remote_check"] = "lfs_sha256"
                entry["remote_match"] = entry["sha256"] == sibling.lfs.sha256
            else:
                entry["remote_check"] = "git_blob_sha1"
                entry["remote_match"] = git_blob_sha1(path) == sibling.blob_id
            if role == "pisco" and name in manifest["files"]:
                entry["matches_reference_upstream"] = entry["sha256"] == manifest["files"][name]["sha256"]
            if not entry["remote_match"] or entry.get("matches_reference_upstream") is False:
                raise RuntimeError(f"{repo_id}/{name}: hash mismatch {entry}")
            files[name] = entry
            print(f"  [{role}] {entry['sha256'][:16]}  {entry['size_bytes']:>12}  {name}")

        snapshots[role] = {"repo_id": repo_id, "revision_sha": sha, "local_dir": str(local_dir),
                           "fetched_at": now(), "files": files}

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print("fetch OK; snapshots recorded in reference/manifest.json")


# ----------------------------------------------------------------- load stage

def build_runtime_dir(manifest):
    """Local model dir = pinned source copy + config with local backbone path + links to pinned .pth."""
    pisco_dir = Path(manifest["snapshots"]["pisco"]["local_dir"])
    backbone_dir = manifest["snapshots"]["backbone"]["local_dir"]
    if RUNTIME_DIR.exists():
        shutil.rmtree(RUNTIME_DIR)
    RUNTIME_DIR.mkdir(parents=True)

    for name in ("modelling_pisco.py", "__init__.py"):
        src = PROJECT_ROOT / manifest["files"][name]["local_path"]
        assert sha256_file(src) == manifest["files"][name]["sha256"], f"{name} changed on disk"
        shutil.copy2(src, RUNTIME_DIR / name)
    for name in ("adapters.pth", "decoder_first_last_layers.pth"):
        (RUNTIME_DIR / name).symlink_to((pisco_dir / name).resolve())

    original = json.loads((PROJECT_ROOT / manifest["files"]["config.json"]["local_path"]).read_text())
    runtime = dict(original, decoder_model_name=backbone_dir)
    (RUNTIME_DIR / "config.json").write_text(json.dumps(runtime, indent=2))
    return {k: {"original": original.get(k), "runtime": runtime.get(k)}
            for k in runtime if original.get(k) != runtime.get(k)}


def tensor_equal_report(expected, actual):
    """Compare a checkpoint tensor (CPU) with the loaded tensor; cast loaded to file dtype."""
    import torch
    actual_cpu = actual.detach().to("cpu")
    return {"shape": list(expected.shape), "file_dtype": str(expected.dtype),
            "loaded_dtype": str(actual.dtype), "loaded_device": str(actual.device),
            "equal_after_cast_to_file_dtype": bool(torch.equal(actual_cpu.to(expected.dtype), expected))}


def check_first_last_layers(model, manifest):
    """decoder_first_last_layers.pth vs loaded decoder, and vs the original Mistral rows."""
    import torch
    from safetensors import safe_open

    saved = torch.load(RUNTIME_DIR / "decoder_first_last_layers.pth", map_location="cpu", weights_only=True)
    state = model.decoder.state_dict()
    backbone_dir = Path(manifest["snapshots"]["backbone"]["local_dir"])
    shard_of = json.loads((backbone_dir / "model.safetensors.index.json").read_text())["weight_map"]

    report = {}
    for key, tensor in saved.items():
        entry = tensor_equal_report(tensor, state[key])
        with safe_open(backbone_dir / shard_of[key], framework="pt") as f:
            base = f.get_tensor(key)  # [32000, 4096] original Mistral rows
        n_base = base.shape[0]
        entry["base_shape"] = list(base.shape)
        # Were the original-vocab rows left untouched by PISCO training?
        entry["rows_lt_base_vocab_equal_to_mistral"] = bool(torch.equal(tensor[:n_base], base))
        entry["rows_lt_base_vocab_max_abs_diff"] = (tensor[:n_base].float() - base.float()).abs().max().item()
        # New token rows: trained values, not the mean-resize init.
        new_rows = tensor[n_base:].float()
        entry["new_rows"] = {"count": int(new_rows.shape[0]),
                             "differ_from_base_row_mean": bool(not torch.allclose(
                                 new_rows, base.float().mean(0, keepdim=True).expand_as(new_rows))),
                             "l2_norms": [round(x, 4) for x in new_rows.norm(dim=1).tolist()]}
        report[key] = entry
    return report


def check_adapters(model):
    import torch

    saved = torch.load(RUNTIME_DIR / "adapters.pth", map_location="cpu", weights_only=True)
    report = {"file_adapter_names": sorted(saved)}
    for name, file_state in saved.items():
        loaded_state = model.decoder.get_adapter_state_dict(name)
        mismatched = [k for k in file_state
                      if k not in loaded_state
                      or not tensor_equal_report(file_state[k], loaded_state[k])["equal_after_cast_to_file_dtype"]]
        cfg = model.decoder.peft_config[name]
        sample_key = sorted(file_state)[0]
        report[name] = {
            "n_tensors_file": len(file_state), "n_tensors_loaded": len(loaded_state),
            "n_params_file": sum(t.numel() for t in file_state.values()),
            "file_dtypes": sorted({str(t.dtype) for t in file_state.values()}),
            "loaded_dtypes": sorted({str(t.dtype) for t in loaded_state.values()}),
            "extra_keys_in_loaded": sorted(set(loaded_state) - set(file_state))[:5],
            "mismatched_keys": mismatched[:5], "n_mismatched": len(mismatched),
            "sample_key": sample_key, "sample_shape": list(file_state[sample_key].shape),
            "peft_config": {"r": cfg.r, "lora_alpha": cfg.lora_alpha, "lora_dropout": cfg.lora_dropout,
                            "target_modules": sorted(cfg.target_modules)},
        }
    return report


def module_states(model):
    """Training-mode flags; LoRA dropout is the one that matters for determinism."""
    import torch
    dropouts = [m for m in model.modules() if isinstance(m, torch.nn.Dropout) and m.p > 0]
    return {"model_training": model.training,
            "n_modules_training": sum(m.training for m in model.modules()),
            "n_dropout_p_gt_0": len(dropouts),
            "n_dropout_p_gt_0_training": sum(m.training for m in dropouts)}


def parameter_summary(model):
    by_dtype, by_device, trainable = {}, {}, {}
    total = 0
    for name, p in model.named_parameters():
        total += p.numel()
        by_dtype[str(p.dtype)] = by_dtype.get(str(p.dtype), 0) + p.numel()
        by_device[str(p.device)] = by_device.get(str(p.device), 0) + p.numel()
        if p.requires_grad:
            trainable[name] = p.numel()
    return {"total": total, "by_dtype": by_dtype, "by_device": by_device,
            "requires_grad_total": sum(trainable.values()),
            "requires_grad_n_tensors": len(trainable),
            "requires_grad_non_lora": [n for n in trainable if "lora_" not in n]}


def tokenizer_summary(tok):
    template = tok.chat_template or ""
    return {"len": len(tok), "padding_side": tok.padding_side, "truncation_side": tok.truncation_side,
            "bos": [tok.bos_token, tok.bos_token_id], "eos": [tok.eos_token, tok.eos_token_id],
            "pad_token_id": tok.pad_token_id,
            "added_tokens": {t: tok.convert_tokens_to_ids(t) for t in tok.additional_special_tokens},
            "mem_token_ids": tok.mem_token_ids, "sep_token_id": tok.sep_token_id,
            "ae_token_id": tok.ae_token_id, "enc_token_id": tok.convert_tokens_to_ids(tok.enc_token),
            "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest()}


def load_reference_model(report):
    """Offline load of the pinned reference on one GPU, in eval mode.

    Shared with later steps. Fills `report` with load-time facts and returns
    (model, loader_log). Raises if the imported source is not the pinned one
    or the loader reported a missing checkpoint file.
    """
    # Must be set before huggingface_hub/transformers are imported.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or "," in visible:
        sys.exit("Set CUDA_VISIBLE_DEVICES to exactly one assigned GPU (e.g. 0).")

    import torch
    import transformers
    from transformers import AutoModel

    manifest = json.loads(MANIFEST_PATH.read_text())
    report.update({"CUDA_VISIBLE_DEVICES": visible,
                   "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
                   "runtime_dir": str(RUNTIME_DIR.relative_to(PROJECT_ROOT)),
                   "runtime_config_diff": build_runtime_dir(manifest)})

    # The pinned loader reports missing files via print/warnings and keeps going,
    # so capture both and fail on them explicitly.
    captured = io.StringIO()
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(captured):
        warnings.simplefilter("always")
        model = AutoModel.from_pretrained(str(RUNTIME_DIR), trust_remote_code=True)
    report["load_cpu_seconds"] = round(time.perf_counter() - t0, 1)
    loader_log = captured.getvalue() + "".join(f"WARNING {w.category.__name__}: {w.message}\n" for w in caught)

    module_file = Path(sys.modules[type(model).__module__].__file__)
    report["model_class"] = f"{type(model).__module__}.{type(model).__name__}"
    report["model_source"] = {"path": str(module_file), "sha256": sha256_file(module_file),
                              "matches_pinned": sha256_file(module_file) == manifest["files"]["modelling_pisco.py"]["sha256"]}
    report["adapter_keys"] = list(model.adapter_keys)
    report["active_adapters_after_load"] = model.decoder.active_adapters()
    report["module_state_after_load"] = module_states(model)
    if not report["model_source"]["matches_pinned"]:
        raise RuntimeError("imported source differs from pinned modelling_pisco.py")
    if "NOT FOUND" in loader_log or "does not exist" in loader_log:
        raise RuntimeError(f"loader reported a missing checkpoint file:\n{loader_log}")

    # Move to the single assigned GPU, then switch to eval (the pinned loader never calls eval()).
    torch.cuda.reset_peak_memory_stats()
    t1 = time.perf_counter()
    model = model.to("cuda")
    model.eval()
    torch.cuda.synchronize()
    report["to_gpu_seconds"] = round(time.perf_counter() - t1, 1)
    return model, loader_log


def load():
    import torch

    manifest = json.loads(MANIFEST_PATH.read_text())
    report = {"step": "03", "started_at": now()}
    model, loader_log = load_reference_model(report)
    (OUTPUT_DIR / "loader_log.txt").write_text(loader_log)

    report["gpu_memory_mib"] = {"peak_allocated": round(torch.cuda.max_memory_allocated() / 2**20),
                                "peak_reserved": round(torch.cuda.max_memory_reserved() / 2**20),
                                "allocated_now": round(torch.cuda.memory_allocated() / 2**20)}
    report["module_state_after_eval"] = module_states(model)

    report["attn_implementation"] = model.decoder.config._attn_implementation
    report["decoder_dtype"] = str(model.decoder.dtype)
    report["n_mem_tokens"] = model.n_mem_tokens
    report["sep"] = model.sep
    report["tokenizer"] = tokenizer_summary(model.decoder_tokenizer)
    report["parameters"] = parameter_summary(model)
    report["first_last_layers"] = check_first_last_layers(model, manifest)
    report["adapters"] = check_adapters(model)
    report["generation_config"] = {
        "cocom": model.generation_config.to_diff_dict() if model.generation_config else None,
        "decoder": model.decoder.generation_config.to_diff_dict()}

    failures = []
    if not report["model_source"]["matches_pinned"]:
        failures.append("imported source differs from pinned modelling_pisco.py")
    if "NOT FOUND" in loader_log or "does not exist" in loader_log:
        failures.append("loader reported a missing checkpoint file")
    if set(report["adapter_keys"]) != EXPECTED_ADAPTERS:
        failures.append(f"adapter keys {report['adapter_keys']}")
    for name in EXPECTED_ADAPTERS & set(report["adapters"]):
        if report["adapters"][name]["n_mismatched"]:
            failures.append(f"{name}: loaded weights differ from adapters.pth")
    for key, entry in report["first_last_layers"].items():
        if not entry["equal_after_cast_to_file_dtype"]:
            failures.append(f"{key}: loaded tensor differs from decoder_first_last_layers.pth")
    if report["module_state_after_eval"]["n_dropout_p_gt_0_training"]:
        failures.append("dropout still in training mode after eval()")
    report["failures"] = failures
    report["status"] = "PASS" if not failures else "FAIL"
    report["finished_at"] = now()

    (OUTPUT_DIR / "load_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["fetch", "load"])
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fetch() if args.stage == "fetch" else load()
