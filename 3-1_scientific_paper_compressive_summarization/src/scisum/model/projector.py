"""프로젝터: 인코더가 만든 잠재 표현을 생성 모델의 입력 공간으로 옮긴다."""
import torch
import torch.nn as nn


class Projector(nn.Module):
    """2층 MLP(GELU) + LayerNorm, 출력 크기를 생성 모델 토큰 임베딩 크기에 맞춘다.

    LayerNorm 출력은 차원당 표준편차가 1이라 노름이 sqrt(H)로 토큰 임베딩보다 훨씬 크다. 고정된 생성 모델이 평소 보던
    입력 크기로 읽도록 out_scale = (토큰 임베딩 평균 노름) / sqrt(H)를 곱한다.
    """

    def __init__(self, in_dim: int, out_dim: int, out_scale: float, hidden_mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_mult * out_dim), nn.GELU(), nn.Linear(hidden_mult * out_dim, out_dim))
        self.norm = nn.LayerNorm(out_dim)
        self.register_buffer("out_scale", torch.tensor(float(out_scale)))

    @staticmethod
    def scale_for(embedding: nn.Embedding) -> float:
        """생성 모델 토큰 임베딩의 평균 노름 / sqrt(H)."""
        w = embedding.weight.detach().float()
        return float(w.norm(dim=1).mean() / w.shape[1] ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(x.to(self.net[0].weight.dtype))) * self.out_scale
