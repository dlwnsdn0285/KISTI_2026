"""문단마다 몇 개의 잠재 표현(슬롯)으로 압축할지 정한다. 압축률은 16배다.

- ratio16 (균등 압축률): 문단마다 토큰 수 / 16을 올림한 개수.
- oracle16 (적응형 배분, 오라클): 문헌당 총 슬롯 수를 정한 뒤, 문단 중요도에 소프트맥스(온도 2)를 취한 비율로 나눈다.
토큰 수는 인코더(Qwen3-Embedding) 토크나이저로 세고, 문단 하나의 슬롯 수는 1개 이상 min(토큰 수, 400)개 이하다.

입력: data/units/, data/alloc/tiers_{train,test}.json
출력: data/alloc/{ratio16,oracle16}_{train,test}.json
실행: python -m pipeline.allocate.build_alloc
"""
import math
import os
from typing import List

import numpy as np
from transformers import AutoTokenizer

from scisum.data.units import load_paper, unit_order, unit_texts
from pipeline.common.io import ALLOC_DIR, ALLOC_GROUPS, group_paper_ids, load_json, save_json

ENCODER = "Qwen/Qwen3-Embedding-0.6B"
RATIO = 16
KMAX = 400   # 부분 단위 하나의 최대 슬롯 수
TAU = 2.0    # 소프트맥스 온도


def split_budget(B: int, p, cap) -> List[int]:
    """B를 p에 비례해 나누되 각 k_i는 [1, cap_i], 합은 B (워터필링 후 최대 나머지 반올림)."""
    p = np.asarray(p, float)
    cap = np.maximum(1, np.asarray(cap, int))
    n = len(p)
    B = int(min(max(B, n), cap.sum()))
    k = np.zeros(n)
    free = np.ones(n, bool)
    rest = float(B)
    for _ in range(3 * n + 3):  # 하한이나 상한에 걸린 문단을 고정하고 남은 예산을 다시 나눈다
        idx = np.where(free)[0]
        if len(idx) == 0:
            break
        x = rest * p[idx] / max(p[idx].sum(), 1e-12)
        lo, hi = x < 1, x > cap[idx]
        if lo.any():
            k[idx[lo]] = 1
            free[idx[lo]] = False
            rest -= lo.sum()
            continue
        if hi.any():
            k[idx[hi]] = cap[idx[hi]]
            free[idx[hi]] = False
            rest -= cap[idx[hi]].sum()
            continue
        k[idx] = x
        break
    ki = np.clip(np.floor(k).astype(int), 1, cap)
    while ki.sum() != B:  # 소수부가 큰 문단부터 하나씩 더하거나(작은 것부터) 뺀다
        d = B - ki.sum()
        frac = k - ki
        room = np.where(ki < cap)[0] if d > 0 else np.where(ki > 1)[0]
        j = room[np.argmax(frac[room])] if d > 0 else room[np.argmin(frac[room])]
        ki[j] += 1 if d > 0 else -1
    return [int(x) for x in ki]


def ratio_alloc(n: np.ndarray) -> List[int]:
    cap = np.minimum(n, KMAX)
    return [int(max(1, min(c, math.ceil(x / RATIO)))) for x, c in zip(n, cap)]


def oracle_alloc(n: np.ndarray, tiers: np.ndarray) -> List[int]:
    cap = np.minimum(n, KMAX)
    B = max(len(n), math.ceil(n.sum() / RATIO))
    return split_budget(B, np.exp((tiers - tiers.max()) / TAU), cap)


def main() -> None:
    tok = AutoTokenizer.from_pretrained(ENCODER)
    for group in ALLOC_GROUPS:
        tiers = load_json(os.path.join(ALLOC_DIR, f"tiers_{group}.json"))
        out = {"ratio16": {}, "oracle16": {}}
        for pid in group_paper_ids(group):
            if pid not in tiers:
                continue
            paper = load_paper(pid)
            idxs, texts = unit_order(paper), unit_texts(paper)
            n = np.array([len(tok(texts[i])["input_ids"]) for i in idxs])
            t = np.array([tiers[pid].get(str(i), 1) for i in idxs], float)
            for name, ks in (("ratio16", ratio_alloc(n)), ("oracle16", oracle_alloc(n, t))):
                out[name][pid] = {"B": sum(ks), "units": [[i, k] for i, k in zip(idxs, ks)]}
        for name, alloc in out.items():
            save_json(alloc, os.path.join(ALLOC_DIR, f"{name}_{group}.json"))
            Bs = np.array([v["B"] for v in alloc.values()])
            print(f"[alloc] {name}_{group}: 문헌 {len(alloc)}편, 문헌당 슬롯 평균 {Bs.mean():.1f} (중앙값 {np.median(Bs):.0f})")


if __name__ == "__main__":
    main()
