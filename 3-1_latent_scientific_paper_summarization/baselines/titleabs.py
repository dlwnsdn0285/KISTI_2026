"""제목 + 초록 베이스라인 입력을 만든다 (SurveyGen 방식).

문헌마다 제목과 초록을 이어 붙인 텍스트를 쓴다. 잠재 표현 모델이 첫 부분 단위로 압축하는 텍스트와 같다.

출력: data/baselines/titleabs_test.jsonl
실행: python -m baselines.titleabs
"""
import os

from baselines.common import out_path, test_papers, write
from scisum.data.units import ABSTRACT_IDX, load_paper, unit_texts


def main() -> None:
    if os.path.exists(out_path("titleabs")):
        os.remove(out_path("titleabs"))
    pids = test_papers()
    for pid in pids:
        write("titleabs", pid, unit_texts(load_paper(pid))[ABSTRACT_IDX])
    print(f"[titleabs] {len(pids)}편 -> {out_path('titleabs')}")


if __name__ == "__main__":
    main()
