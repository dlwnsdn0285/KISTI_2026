"""IM-SciQA를 논문이 서로 겹치지 않도록 학습/검증/평가 데이터로 나눈다.

논문을 점, 문항을 두 논문을 잇는 선으로 보고 Louvain 군집화로 묶은 뒤, 군집을 모아 검증/평가 데이터를 만든다.
나눈 경계에 걸린 문항은 쓸 수 없으므로, 경계의 논문을 옮겨 버려지는 문항 수를 줄인다.
--reported를 주면 실험에 쓴 분할(pipeline/parse/reported_split_ids.json)을 그대로 쓴다.

출력: data/imsciqa/{train,valid,test}.jsonl, data/imsciqa/{train,valid,test}_paper_ids.json
실행: python -m pipeline.parse.split_imsciqa --reported
"""
import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from typing import Dict, List, Set

import networkx as nx

from pipeline.common.io import IMSCIQA_DIR, append_jsonl, save_json
from pipeline.parse.build_units import bioc_title, load_bioc
from pipeline.parse.imsciqa_source import load_source_rows

SEED = 20260716          # 2단계 Louvain과 군집 순서
FIRST_SEED = 42          # 1단계
N_HELD = 300             # 1단계에서 떼어 두는 문항 수
N_TEST = N_VALID = 1000  # 2단계 split 최소 문항 수 (군집 단위로 모으므로 조금 넘을 수 있다)
RESOLUTION = 15.0
TRIALS = 12
REFINE_ROUNDS = 60
REPORTED_IDS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reported_split_ids.json")
STOP = set("the a an of and for in on with to from by using their its this that study "
           "patients analysis effects effect role based between during after among via".split())


def pair(r: dict) -> tuple:
    return r["source_paper_id"], r["target_paper_id"]


def candidate_rows(rows: List[dict]) -> List[dict]:
    """1단계: 저빈도 문헌 문항 300개를 떼고, 그 문헌을 쓰는 문항을 버린 뒤 (나머지 섞은 순서 + 떼어 둔 문항) 순서로 돌려준다."""
    cnt = Counter(p for r in rows for p in pair(r))
    rng = random.Random(FIRST_SEED)
    order = sorted(range(len(rows)), key=lambda i: (sum(cnt[p] for p in pair(rows[i])), rng.random()))
    held = order[:N_HELD]
    held_papers = {p for i in held for p in pair(rows[i])}
    held_set = set(held)
    rest = [i for i in range(len(rows)) if i not in held_set and not set(pair(rows[i])) & held_papers]
    random.Random(FIRST_SEED + 1).shuffle(rest)
    return [rows[i] for i in rest + held]


class PaperGraph:
    """문헌 = 노드, 문항 = 간선."""

    def __init__(self, rows: List[dict]):
        self.rows = rows
        self.adj: Dict[str, list] = defaultdict(list)
        self.G = nx.Graph()
        for i, r in enumerate(rows):
            x, y = pair(r)
            self.adj[x].append((y, i))
            self.adj[y].append((x, i))
            w = self.G.get_edge_data(x, y, {"weight": 0})["weight"]
            self.G.add_edge(x, y, weight=w + 1)

    def rows_of(self, S: Set[str]):
        """S 안에 온전히 들어가는 문항과 S 경계에서 잘리는 문항."""
        inside = {i for p in S for o, i in self.adj[p] if o in S}
        cut = {i for p in S for o, i in self.adj[p] if o not in S}
        return inside, cut


def title_profile(pids, titles: Dict[str, str], n: int = 12) -> set:
    """제목에 가장 많이 나온 단어 n개 (split의 주제 분포 요약)."""
    w = Counter(x for p in sorted(pids) for x in re.findall(r"[a-z][a-z-]{3,}", titles[p]) if x not in STOP)
    return {k for k, _ in sorted(w.items(), key=lambda kv: (-kv[1], kv[0]))[:n]}


def select(g: PaperGraph, comms, target: int, titles, base_profile, avoid=frozenset()) -> Set[str]:
    """군집을 무작위 순서로 모아 target 문항 이상을 만든다. TRIALS번 시도해 제목 분포가 전체와 가장 닮은 것을 고른다."""
    cand = [c for c in (set(x) - avoid for x in comms) if len(c) > 1]
    best = None
    for t in range(TRIALS):
        rng = random.Random(SEED + t)
        S: Set[str] = set()
        for ci in rng.sample(range(len(cand)), len(cand)):
            if len(g.rows_of(S)[0]) >= target:
                break
            S |= cand[ci]
        inside, cut = g.rows_of(S)
        if len(inside) < target:
            continue
        score = (len(base_profile & title_profile(S, titles)), -len(cut))
        if best is None or score > best[0]:
            best = (score, set(S))
    if best is None:
        raise SystemExit("요청한 크기의 split을 만들 수 없음")
    return best[1]


def refine(g: PaperGraph, S: Set[str], target: int) -> Set[str]:
    """경계 문헌을 하나씩 넣거나 빼서, 크기를 지키며 잘리는 문항 수가 줄어드는 동안 반복한다."""
    for _ in range(REFINE_ROUNDS):
        _, cut = g.rows_of(S)
        outside = {o for p in S for o, _ in g.adj[p] if o not in S}
        border = {p for p in S if any(o not in S for o, _ in g.adj[p])}
        moved = False
        for p in sorted(outside, key=lambda x: (-sum(1 for o, _ in g.adj[x] if o in S), x)):
            ni, nc = g.rows_of(S | {p})
            if len(nc) < len(cut) and len(ni) >= target:
                S.add(p)
                moved = True
                break
        if not moved:
            for p in sorted(border, key=lambda x: (sum(1 for o, _ in g.adj[x] if o in S), x)):
                ni, nc = g.rows_of(S - {p})
                if len(nc) < len(cut) and len(ni) >= target:
                    S.discard(p)
                    moved = True
                    break
        if not moved:
            break
    return S


def louvain_split(rows: List[dict], titles: Dict[str, str]) -> Dict[str, List[int]]:
    """2단계: {"train", "valid", "test"} -> 문항 위치 목록."""
    g = PaperGraph(rows)
    comms = nx.community.louvain_communities(g.G, weight="weight", resolution=RESOLUTION, seed=SEED)
    base = title_profile(list(g.adj), titles)
    T = refine(g, select(g, comms, N_TEST, titles, base), N_TEST)
    t_in, t_cut = g.rows_of(T)
    left = {p for p in g.adj if p not in T}
    V = refine(g, select(g, [c & left for c in comms], N_VALID, titles, base, avoid=T), N_VALID)
    v_in, v_cut = g.rows_of(V)
    dropped = (t_cut | v_cut) - t_in - v_in
    train = [i for i in range(len(rows)) if i not in t_in and i not in v_in and i not in dropped]
    return {"train": train, "valid": sorted(v_in), "test": sorted(t_in)}


def reported_split(rows: List[dict]) -> Dict[str, List[int]]:
    """보고서 실험의 split: 저장된 문항 id 목록을 문항 위치로 바꾼다."""
    pos = {r["global_id"]: i for i, r in enumerate(rows)}
    with open(REPORTED_IDS, encoding="utf-8") as f:
        return {s: [pos[q] for q in ids] for s, ids in json.load(f).items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reported", action="store_true", help="보고서 실험의 split을 그대로 쓴다")
    args = ap.parse_args()
    rows = candidate_rows(load_source_rows())
    if args.reported:
        splits = reported_split(rows)
    else:
        titles = {p: bioc_title(load_bioc(p)).lower() for p in {p for r in rows for p in pair(r)}}
        splits = louvain_split(rows, titles)
    papers = {s: sorted({p for i in idx for p in pair(rows[i])}) for s, idx in splits.items()}
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        assert not set(papers[a]) & set(papers[b]), f"{a}와 {b}가 문헌을 공유함"
    os.makedirs(IMSCIQA_DIR, exist_ok=True)
    for s, idx in splits.items():
        path = os.path.join(IMSCIQA_DIR, f"{s}.jsonl")
        if os.path.exists(path):
            os.remove(path)
        for i in idx:
            r = rows[i]
            append_jsonl(path, {"id": r["global_id"], "question": r["combined_question"],
                                "answer": r["combined_answer"], "paper_ids": list(pair(r))})
        save_json(papers[s], os.path.join(IMSCIQA_DIR, f"{s}_paper_ids.json"))
        print(f"[split] {s}: {len(idx)}문항, 문헌 {len(papers[s])}편")
    n_drop = len(rows) - sum(len(v) for v in splits.values())
    print(f"[split] 후보 {len(rows)}문항 중 split 경계에 걸려 버린 문항 {n_drop}")


if __name__ == "__main__":
    main()
