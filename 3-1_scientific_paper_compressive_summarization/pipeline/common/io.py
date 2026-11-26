"""데이터 파일 읽기/쓰기 함수.

생성 결과는 한 건이 끝날 때마다 파일 끝에 추가하므로, 중간에 멈춰도 다시 실행하면 끝난 건은 건너뛰고 이어서 진행한다.
"""
import json
import os
from typing import Iterator

IMSCIQA_DIR = os.path.join("data", "imsciqa")
ALLOC_DIR = os.path.join("data", "alloc")


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str) -> None:
    """임시 파일에 쓴 뒤 바꿔치기해 중간에 끊겨도 깨진 파일이 남지 않게 한다."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(path + ".tmp", path)


def read_jsonl(path: str) -> Iterator[dict]:
    """한 줄에 JSON 하나. 쓰다가 끊긴 마지막 줄은 무시한다."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def append_jsonl(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def split_paper_ids(split: str) -> list:
    """IM-SciQA split의 논문 목록 (data/imsciqa/<split>_paper_ids.json)."""
    return load_json(os.path.join(IMSCIQA_DIR, f"{split}_paper_ids.json"))


def split_rows(split: str) -> list:
    return list(read_jsonl(os.path.join(IMSCIQA_DIR, f"{split}.jsonl")))


# 배분 파일의 두 묶음: _train = 학습/검증 문헌, _test = 평가 문헌
ALLOC_GROUPS = {"train": ("train", "valid"), "test": ("test",)}


def group_paper_ids(group: str) -> list:
    return sorted({p for s in ALLOC_GROUPS[group] for p in split_paper_ids(s)})
