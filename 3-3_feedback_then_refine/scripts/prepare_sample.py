"""Stratified sample of unsupported claim-reference pairs for the FtR experiment.

- all 'contradiction' rows are kept (rare but most informative error type)
- the rest is drawn proportionally to each (model, condition) group
"""
import json
import random
from collections import Counter, defaultdict

from common import FTR_ROOT, POOL

SRC = POOL / "unsupported.jsonl"
OUT = FTR_ROOT / "data" / "sample.jsonl"
N = 1000
SEED = 42

rows = [json.loads(l) for l in open(SRC)]
random.seed(SEED)

contra = [r for r in rows if r["llm_label"]["label"] == "contradiction"]
rest = [r for r in rows if r["llm_label"]["label"] != "contradiction"]
groups = defaultdict(list)
for r in rest:
    groups[(r["model"] or "-", r["condition"])].append(r)

budget = N - len(contra)
total = len(rest)
alloc = {g: max(1, round(budget * len(v) / total)) for g, v in groups.items()}
diff = budget - sum(alloc.values())  # fix rounding on the largest group
alloc[max(groups, key=lambda g: len(groups[g]))] += diff

sample = list(contra)
for g, v in groups.items():
    sample += random.sample(v, min(alloc[g], len(v)))
random.shuffle(sample)

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    for i, r in enumerate(sample):
        r["uid"] = f"u{i:04d}"
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(len(sample))
print(Counter(r["llm_label"]["label"] for r in sample))
print(Counter((r["model"] or "-", r["condition"]) for r in sample))
