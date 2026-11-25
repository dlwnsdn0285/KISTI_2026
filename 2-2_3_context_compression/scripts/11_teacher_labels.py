"""Step 11b: teacher labels for fusion training = the raw backbone's answers (Raw path) on the
training manifest, so the student keeps PISCO's answer style (sequence-level KD as in PISCO).

    CUDA_VISIBLE_DEVICES=6 python scripts/11_teacher_labels.py --shard 0 --n-shards 2
    CUDA_VISIBLE_DEVICES=7 python scripts/11_teacher_labels.py --shard 1 --n-shards 2

Output: artifacts/step11/teacher/<manifest_id>.shard<i>of<n>.jsonl  (sample_id, teacher, output_ids, stopped_at_eos)
"""

import argparse
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
OUT_DIR = PROJECT_ROOT / "artifacts" / "step11" / "teacher"
MAX_NEW_TOKENS = 128


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    args = p.parse_args()
    import torch
    torch.set_num_threads(CPU_THREADS)
    from pisco_research.raw import RawBackbone
    manifest = json.loads(args.manifest.read_text())
    ref = json.loads((PROJECT_ROOT / "reference" / "manifest.json").read_text())
    raw = RawBackbone(ref["snapshots"]["backbone"]["local_dir"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{manifest['manifest_id']}.shard{args.shard}of{args.n_shards}.jsonl"
    samples = manifest["samples"][args.shard::args.n_shards]
    t0 = time.perf_counter()
    with open(out, "w") as f:
        for i, s in enumerate(samples):
            mi = s["model_input"]
            res = raw.generate(mi["question"], [d["used_text"] for d in mi["documents"]], max_new_tokens=MAX_NEW_TOKENS)
            ids = res["output_ids"].tolist()
            f.write(json.dumps({"sample_id": s["sample_id"], "teacher": res["text"], "output_ids": ids,
                                "stopped_at_eos": 2 in ids}, ensure_ascii=False) + "\n")
            if i % 100 == 0:
                print(f"[teacher shard {args.shard}] {i}/{len(samples)} {time.perf_counter() - t0:.0f}s {res['text'][:80]!r}", flush=True)
    print("written:", out.relative_to(PROJECT_ROOT), len(samples))


if __name__ == "__main__":
    main()
