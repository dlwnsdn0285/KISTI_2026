"""IM-SciQA 원본 데이터(Hugging Face MRAGDATASET/Pubmed-MHQA의 dev와 test)를 읽는다.

문항 하나는 논문 2편에 대해 묻는다. 원본 id는 dev와 test에서 겹치므로, 두 데이터를 이어 붙인 순서로 "mhqa-<번호>" id를 새로 붙인다.
"""
from typing import List

HF_NAME = "MRAGDATASET/Pubmed-MHQA"


def load_source_rows() -> List[dict]:
    """원본 문항 목록 (dev 뒤에 test). 각 행에 전역 id를 붙인다."""
    from datasets import load_dataset

    ds = load_dataset(HF_NAME)
    rows = [dict(r) for r in ds["dev"]] + [dict(r) for r in ds["test"]]
    for i, r in enumerate(rows):
        r["global_id"] = f"mhqa-{i}"
    return rows


def all_paper_ids(rows: List[dict]) -> List[str]:
    return sorted({r[k] for r in rows for k in ("source_paper_id", "target_paper_id")})
