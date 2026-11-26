"""손실 계산: 유효 배치 전체의 정답 토큰 평균 교차 엔트로피.

마이크로배치마다 손실의 합으로 역전파하고, 갱신 직전에 모든 GPU의 정답 토큰 수로 gradient를 나눈다.
그래서 GPU 수나 마이크로배치 크기를 바꿔도 같은 gradient가 나온다.
"""
from typing import Iterable, Tuple

import torch


def scale_to_token_mean(params: Iterable[torch.nn.Parameter], n_tokens: float, world_size: int = 1):
    """누적된 'NLL 합' gradient를 토큰 평균 gradient로 바꾼다.

    DDP는 GPU 사이 gradient를 평균(합 / world_size)하므로 world_size를 다시 곱한 뒤 전체 토큰 수로 나눈다.
    """
    scale = world_size / max(float(n_tokens), 1.0)
    for p in params:
        if p.grad is not None:
            p.grad.mul_(scale)


def reduce_sums(nll_sum: float, n_tokens: int, accelerator=None) -> Tuple[float, float]:
    """모든 GPU의 (NLL 합, 토큰 수)를 더한다. 로그와 검증 손실도 이 합으로 토큰 평균을 낸다."""
    t = torch.tensor([float(nll_sum), float(n_tokens)], dtype=torch.float64,
                     device=accelerator.device if accelerator is not None else "cpu")
    if accelerator is not None and accelerator.num_processes > 1:
        t = accelerator.reduce(t, reduction="sum")
    return float(t[0]), float(t[1])
