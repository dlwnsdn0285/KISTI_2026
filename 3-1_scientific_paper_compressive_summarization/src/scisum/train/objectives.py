"""학습 목표별 손실: 항목 묶음을 인코딩해 생성 모델에 넣고 정답 토큰의 손실을 돌려준다.

지시문은 학습 때 여러 문구 중 무작위로, 검증 때는 첫 번째 문구로 고정해 쓴다.
"""
import random
from typing import List

import torch

from .. import prompts


def sentence_reconstruction(model, items: List[dict]) -> torch.Tensor:
    """1단계 문장 복원 구간: 문장 하나를 잠재 표현 2개로 압축하고 그 문장을 그대로 복원한다."""
    latents = model.encode_units([it["text"] for it in items], [it["k"] for it in items])
    rows = [prompts.recon_segments(z, it["text"], model.training) for z, it in zip(latents, items)]
    return model.decoder.target_nll(rows)


def single_paragraph(model, items: List[dict], use_qa: bool = True) -> torch.Tensor:
    """1단계 (단일 문단 요약 학습): 부분 잠재 요약 하나만 보고 그 문단에 대해 다음 행을 모두 학습한다.

    - 원문 복원 1행
    - 재진술 1행 (재진술이 있으면; 학습 때는 여러 개 중 무작위 1개, 검증 때는 첫 번째)
    - 질의응답: 그 문단의 QA 쌍 전부 (use_qa=False면 생략: '원문 복원 및 재진술' 설정)
    같은 잠재 표현을 모든 행이 공유하므로 인코더는 문단마다 한 번만 돈다.
    """
    latents = model.encode_units([it["text"] for it in items], [it["k"] for it in items])
    rows = []
    for z, it in zip(latents, items):
        rows.append(prompts.recon_segments(z, it["text"], model.training))
        if it["paraphrases"]:
            para = random.choice(it["paraphrases"]) if model.training else it["paraphrases"][0]
            rows.append(prompts.paraphrase_segments(z, para, model.training))
        if use_qa:
            rows += [prompts.unit_qa_segments(z, q, a) for q, a in it["qa"]]
    return model.decoder.target_nll(rows)


def multi_paper_qa(model, items: List[dict]) -> torch.Tensor:
    """2단계 (다중 문헌 요약 활용 학습): 문헌마다 부분 잠재 요약을 이어 붙인 잠재 표현 + 질문 -> 답."""
    rows = []
    for it in items:
        papers = [model.encode_paper(p["texts"], p["counts"]) for p in it["papers"]]
        rows.append(prompts.qa_segments(it["question"], papers, it["answer"]))
    return model.decoder.target_nll(rows)
