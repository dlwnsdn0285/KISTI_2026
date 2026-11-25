"""Step 11c: cache the frozen PISCO memories of a manifest (per-question compression, as in evaluation).

    CUDA_VISIBLE_DEVICES=6 python scripts/11_precompute_memory.py [--manifest PATH]

Output: artifacts/step11/memory/<manifest_id>.pt  {"sample_ids": [...], "memory": bf16 [N, k, 8, H]}
"""

import argparse
import importlib.util
import json
import os
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
OUT_DIR = PROJECT_ROOT / "artifacts" / "step11" / "memory"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    args = p.parse_args()
    import torch
    torch.set_num_threads(CPU_THREADS)
    from pisco_research.reference import PiscoReference
    spec = importlib.util.spec_from_file_location("step03", PROJECT_ROOT / "scripts" / "03_load_reference.py")
    step03 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(step03)
    model, _ = step03.load_reference_model({})
    wrapper = PiscoReference(model)
    manifest = json.loads(args.manifest.read_text())
    mems, ids = [], []
    t0 = time.perf_counter()
    for i, s in enumerate(manifest["samples"]):
        memory, _ = wrapper.compress([d["text"] for d in s["model_input"]["documents"]])
        mems.append(memory.cpu())
        ids.append(s["sample_id"])
        if i % 500 == 0:
            print(f"[memory] {i}/{len(manifest['samples'])} {time.perf_counter() - t0:.0f}s", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{manifest['manifest_id']}.pt"
    torch.save({"sample_ids": ids, "memory": torch.stack(mems),
                "manifest_sha256": __import__("hashlib").sha256(args.manifest.read_bytes()).hexdigest()}, out)
    print("written:", out.relative_to(PROJECT_ROOT), tuple(torch.stack(mems).shape))


if __name__ == "__main__":
    main()
