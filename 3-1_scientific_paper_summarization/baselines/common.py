"""베이스라인 입력 파일을 만드는 스크립트들이 함께 쓰는 함수.

평가 문헌 목록을 읽고, 문헌마다 {"paper_id", "text"} 한 줄을 파일에 추가한다. 이미 쓴 문헌은 다시 실행할 때 건너뛴다.
"""
import os
from typing import List

from scisum.data.units import UNITS_DIR
from pipeline.common.io import append_jsonl, read_jsonl, split_paper_ids

OUT_DIR = os.path.join("data", "baselines")


def test_papers() -> List[str]:
    return [p for p in split_paper_ids("test") if os.path.exists(os.path.join(UNITS_DIR, f"{p}.json"))]


def out_path(name: str) -> str:
    return os.path.join(OUT_DIR, f"{name}_test.jsonl")


def done_papers(name: str) -> set:
    return {r["paper_id"] for r in read_jsonl(out_path(name))}


def write(name: str, pid: str, text: str) -> None:
    append_jsonl(out_path(name), {"paper_id": pid, "text": text})
