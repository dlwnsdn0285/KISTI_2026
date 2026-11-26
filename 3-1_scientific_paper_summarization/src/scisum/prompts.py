"""생성 모델에 넣는 프롬프트 문구와 입력 순서.

입력은 세 종류의 조각으로 이루어진다: 일반 텍스트(지시문, 질문. 손실 제외), 잠재 표현(프로젝터 출력), 정답 텍스트(손실 계산).
- 다중 문헌 질의응답: Qwen 대화 형식이며, 논문 자리에 잠재 표현이나 베이스라인 텍스트가 들어간다.
- 단일 문단 요약 학습: 잠재 표현 뒤에 지시문을 주고, 원문, 재진술, 답을 정답으로 쓴다.
"""
import random
from typing import List, Sequence, Tuple, Union

import torch

Segment = Tuple[str, Union[str, torch.Tensor]]

# ---- Qwen 채팅 형식 (비사고 모드: 빈 think 블록으로 바로 답한다)
SYSTEM_OPEN, USER_OPEN, TURN_CLOSE = "<|im_start|>system\n", "<|im_start|>user\n", "<|im_end|>\n"
ASSISTANT_OPEN = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
END_OF_TURN = "<|im_end|>"          # 모든 정답 텍스트 끝에 붙이는 종료 토큰 (생성 중단 토큰)

# ---- 다중 문헌 질의응답
QA_SYSTEM = "You are a scientific question answering assistant."
QA_INSTRUCTION = (
    "Answer the following multi-hop scientific question using your own knowledge and general "
    "reasoning. You are ALSO given reference material below, which may or may not contain what "
    "the question asks; use it when it helps and rely on your own knowledge when it does not. "
    "Never state that the information is unavailable -- always commit to your single best "
    "answer. Provide a concise, complete answer in one or two sentences. Exact numbers are not "
    "required."
)
PAPERS_HEADER = "\n\nPapers:\n"
PAPER_BLOCK = "[Paper {i}]\n"
PAPER_SEP = "\n\n"
QUESTION_LEAD = "Question: {q}"

# ---- 단일 문단 요약 학습: {latent} 자리에 부분 잠재 요약이 들어간다
LATENT = "{latent}"
RECON_INSTRUCTIONS = [      # 원문 복원
    "Background: {latent} Repeat the background text above exactly, word for word:",
    "Background: {latent} Reproduce the preceding background verbatim:",
    "Background: {latent} Copy the background sentences above exactly as written:",
    "Background: {latent} The exact text of the background above is:",
]
PARAPHRASE_INSTRUCTIONS = [  # 재진술
    "Background: {latent} Rewrite the background above in different words, keeping every fact:",
    "Background: {latent} Restate all the information in the background above with new wording and sentence order:",
    "Background: {latent} Paraphrase the background above without copying its sentences:",
    "Background: {latent} Express the same content as the background above in your own words:",
]
UNIT_QA_INSTRUCTION = "{latent}\nQuestion: {q}\nAnswer:"   # 문단 질의응답


def qa_segments(question: str, papers: Sequence[Union[str, torch.Tensor]], answer: str = None) -> List[Segment]:
    """다중 문헌 질의응답 입력. papers[i]가 텐서면 잠재 표현, 문자열이면 자연어 텍스트(베이스라인)로 넣는다."""
    segs: List[Segment] = [("text", f"{SYSTEM_OPEN}{QA_SYSTEM}{TURN_CLOSE}{USER_OPEN}{QA_INSTRUCTION}{PAPERS_HEADER}")]
    for i, paper in enumerate(papers):
        segs.append(("text", PAPER_BLOCK.format(i=i + 1)))
        segs.append(("text", paper) if isinstance(paper, str) else ("latent", paper))
        segs.append(("text", PAPER_SEP))
    segs.append(("text", QUESTION_LEAD.format(q=question.strip()) + TURN_CLOSE + ASSISTANT_OPEN))
    if answer is not None:
        segs.append(("target", answer.strip() + END_OF_TURN))
    return segs


def unit_segments(latent: torch.Tensor, instruction: str, target: str) -> List[Segment]:
    """단일 문단 요약 학습 한 행: 지시문의 {latent} 자리에 부분 잠재 요약을 넣고 정답을 이어 쓴다."""
    before, after = instruction.split(LATENT)
    segs: List[Segment] = [("text", before)] if before else []
    return segs + [("latent", latent), ("text", after + " "), ("target", target.strip() + END_OF_TURN)]


def pick(options: List[str], train: bool) -> str:
    """학습 때는 지시문을 무작위로 고르고, 검증 때는 첫 번째로 고정해 손실이 결정적이 되게 한다."""
    return random.choice(options) if train else options[0]


def recon_segments(latent: torch.Tensor, text: str, train: bool) -> List[Segment]:
    """원문 복원: 잠재 표현만 보고 부분 단위를 그대로 되살린다."""
    return unit_segments(latent, pick(RECON_INSTRUCTIONS, train), text)


def paraphrase_segments(latent: torch.Tensor, paraphrase: str, train: bool) -> List[Segment]:
    """재진술: 같은 내용을 다른 문장 구조로 다시 쓴다."""
    return unit_segments(latent, pick(PARAPHRASE_INSTRUCTIONS, train), paraphrase)


def unit_qa_segments(latent: torch.Tensor, question: str, answer: str) -> List[Segment]:
    """문단 질의응답: 잠재 표현과 질문으로 답을 쓴다 (질문 토큰은 손실 제외)."""
    return unit_segments(latent, UNIT_QA_INSTRUCTION.replace("{q}", question.strip()), answer)
