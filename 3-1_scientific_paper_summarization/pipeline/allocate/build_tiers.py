"""문단 중요도(1~5점)를 매긴다. IM-SciQA 문항의 정답 근거에 가까운 문단일수록 높다.

문항이 가리키는 근거 섹션에서 정답과 가장 비슷한 문장(BM25)을 근거 문장으로 고른다.
근거 문장이 들어 있는 문단은 5점, 나머지 문단은 근거 문장과 비슷한 정도에 따라 1~4점이다.

입력: data/imsciqa/, data/units/
출력: data/alloc/tiers_{train,test}.json
실행: python -m pipeline.allocate.build_tiers
"""
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from scisum.data.units import ABSTRACT_IDX, UNITS_DIR, load_paper, unit_order
from pipeline.common.io import ALLOC_DIR, ALLOC_GROUPS, save_json, split_rows
from pipeline.parse.imsciqa_source import load_source_rows

MARGIN = 0.05
MAX_SENTS = 3
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")

Unit = Tuple[int, str, str]  # (idx, 섹션 이름, 텍스트)


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def words(t: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", t.lower())


def sentences(text: str) -> List[str]:
    return [s.strip() for s in SENT_SPLIT.split(text) if len(s.split()) >= 4]


def bm25_scores(docs: List[str], query: str) -> np.ndarray:
    """각 문서와 질의의 BM25 점수를 문서 집합 안 최대값으로 나눈 값 (0–1)."""
    sc = np.asarray(BM25Okapi([words(d) for d in docs]).get_scores(words(query)), dtype=float)
    return sc / sc.max() if sc.max() > 0 else sc


def scoring_units(paper: dict) -> List[Unit]:
    """점수를 매길 부분 단위: 초록(idx -1, 섹션 "Abstract", 제목 없이)과 본문 문단 (섹션 접두어 없이)."""
    units = [(ABSTRACT_IDX, "Abstract", paper["abstract"])] if paper.get("abstract") else []
    return units + [(p["idx"], p.get("section") or "", p["text"]) for p in paper["paragraphs"]]


def evidence_units(units: List[Unit], section_name: str, answer: str) -> List[Tuple[int, str]]:
    """근거 섹션 안에서 정답과 가장 가까운 근거 문장들 [(문단 idx, 문장)]."""
    sec = norm(section_name)
    secs = [norm(s) for _, s, _ in units]
    cand = [i for i, s in enumerate(secs) if s == sec]
    if not cand and sec:
        cand = [i for i, s in enumerate(secs) if sec in s or s in sec]
    if not cand and sec == "abstract":
        cand = [i for i, (idx, _, _) in enumerate(units) if idx == ABSTRACT_IDX]
    if not cand:
        cand = list(range(len(units)))
    sents = [(units[i][0], s) for i in cand for s in sentences(units[i][2])]
    if not sents:
        return []
    sc = bm25_scores([s for _, s in sents], answer)
    order = np.argsort(-sc)
    top = sc[order[0]]
    return [sents[j] for j in order[:MAX_SENTS] if sc[j] >= top - MARGIN]


def paper_tiers(units: List[Unit], evidence: List[Tuple[int, str]]) -> Dict[int, int]:
    """근거 문장 목록으로 문단 점수 1–5를 매긴다."""
    texts = [t for _, _, t in units]
    sim = np.max(np.stack([bm25_scores(texts, s) for _, s in evidence]), axis=0)
    ev_idx = {i for i, _ in evidence}
    non = [j for j, (idx, _, _) in enumerate(units) if idx not in ev_idx]
    lo = float(sim[non].min()) if non else 0.0
    hi = float(sim[non].max()) if non else 1.0
    tiers = {}
    for j, (idx, _, _) in enumerate(units):
        b = 5.0 if idx in ev_idx else 1.0 + 3.0 * ((float(sim[j]) - lo) / (hi - lo) if hi > lo else 0.0)
        tiers[idx] = int(min(5, max(1, round(round(b, 3)))))
    return tiers


def main() -> None:
    source = {r["global_id"]: r for r in load_source_rows()}
    for group, splits in ALLOC_GROUPS.items():
        evidence: Dict[str, list] = defaultdict(list)
        papers: Dict[str, dict] = {}
        missing = set()
        for split in splits:
            for row in split_rows(split):
                src = source[row["id"]]
                answer = src["inter_document_answer"] or row["answer"]
                for role in ("source", "target"):
                    pid = src[f"{role}_paper_id"]
                    if pid not in papers:
                        if not os.path.exists(os.path.join(UNITS_DIR, f"{pid}.json")):
                            missing.add(pid)
                            continue
                        papers[pid] = load_paper(pid)
                    evidence[pid] += evidence_units(scoring_units(papers[pid]), src[f"{role}_section_name"], answer)
        if missing:   # 문헌 하나라도 빠지면 그 문항의 학습과 평가가 불가능하므로 여기서 멈춘다
            raise SystemExit(f"[tiers] 부분 단위 파일이 없는 문헌 {len(missing)}편 (fetch_pmc / build_units 확인): "
                             + ", ".join(sorted(missing)[:20]))
        out, hist = {}, defaultdict(int)
        for pid, paper in sorted(papers.items()):
            tiers = paper_tiers(scoring_units(paper), evidence[pid]) if evidence[pid] else {}
            out[pid] = {str(i): tiers.get(i, 1) for i in unit_order(paper)}
            for t in out[pid].values():
                hist[t] += 1
        save_json(out, os.path.join(ALLOC_DIR, f"tiers_{group}.json"))
        print(f"[tiers] {group}: 문헌 {len(out)}편, 점수 분포 {dict(sorted(hist.items()))}")


if __name__ == "__main__":
    main()
