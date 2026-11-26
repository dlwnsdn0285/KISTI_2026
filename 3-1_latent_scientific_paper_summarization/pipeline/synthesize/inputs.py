"""학습 데이터 생성에 쓸 부분 단위 목록과 결과 파일 경로.

IM-SciQA 학습 데이터의 논문만 쓴다. 생성 모델에는 문단 텍스트, 섹션 이름, 논문 제목을 따로 넘긴다.
"""
import os
from typing import List, Tuple

from scisum.data.units import ABSTRACT_IDX, UNITS_DIR, load_paper, unit_order, unit_texts
from pipeline.common.io import read_jsonl, split_paper_ids

RAW_DIR = os.path.join("data", "synth", "raw")
GEN_SPLIT = "train"

Unit = Tuple[int, str, str]  # (idx, 섹션 이름, 텍스트)


def generator_units(paper: dict) -> List[Unit]:
    texts = unit_texts(paper)
    paras = {p["idx"]: p for p in paper["paragraphs"]}
    return [(i, "Abstract", texts[i]) if i == ABSTRACT_IDX else (i, paras[i].get("section") or "", paras[i]["text"])
            for i in unit_order(paper)]


def generator_papers() -> List[str]:
    """생성 대상 문헌 = 학습 split 문헌 가운데 부분 단위 파일이 있는 것."""
    return [p for p in split_paper_ids(GEN_SPLIT) if os.path.exists(os.path.join(UNITS_DIR, f"{p}.json"))]


def paper_context(pid: str) -> Tuple[dict, List[Unit]]:
    paper = load_paper(pid)
    return paper, generator_units(paper)


def raw_path(kind: str, pid: str) -> str:
    """data/synth/raw/<kind>/<pid>.jsonl: 한 줄 = 부분 단위 하나의 최종 결과."""
    return os.path.join(RAW_DIR, kind, f"{pid}.jsonl")


def done_units(kind: str, pid: str) -> set:
    return {r["idx"] for r in read_jsonl(raw_path(kind, pid))}
