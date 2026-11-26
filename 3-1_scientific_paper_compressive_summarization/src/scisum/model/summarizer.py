"""과학기술 특화 요약 모델 전체: 인코더, 프로젝터, 생성 모델.

논문마다 부분 단위를 따로 압축해 부분 잠재 요약을 만들고, 읽는 순서대로 이어 붙여 논문 하나의 잠재 표현으로 쓴다.
학습하는 파라미터는 인코더 전체, 프로젝터, 잠재 표현용 LoRA다.
"""
from typing import Callable, List, Tuple

import torch
import torch.nn as nn

from .decoder import LatentDecoder
from .encoder import ParagraphEncoder
from .projector import Projector


class LatentSummarizer(nn.Module):
    def __init__(self, encoder: ParagraphEncoder, decoder: LatentDecoder, encoder_token_budget: int = 8192):
        super().__init__()
        self.encoder, self.decoder = encoder, decoder
        self.projector = Projector(encoder.hidden_size, decoder.embed.weight.shape[1], Projector.scale_for(decoder.embed))
        self.projector.to(device=decoder.device, dtype=decoder.embed.weight.dtype)
        self.encoder_token_budget = encoder_token_budget   # 인코더 한 번에 넣는 (행 수 x 최대 길이) 상한
        self.config = {}

    @classmethod
    def build(cls, encoder_id: str, decoder_id: str, lora_rank: int = 32, encoder_max_len: int = 2048,
              dtype=torch.bfloat16, grad_checkpointing: bool = True) -> "LatentSummarizer":
        model = cls(ParagraphEncoder.from_pretrained(encoder_id, encoder_max_len, dtype, grad_checkpointing),
                    LatentDecoder.from_pretrained(decoder_id, lora_rank, dtype, grad_checkpointing))
        model.config = dict(encoder_id=encoder_id, decoder_id=decoder_id, lora_rank=lora_rank, encoder_max_len=encoder_max_len)
        return model

    def forward(self, objective: Callable, items: list) -> torch.Tensor:
        """학습 목표 하나를 실행한다. DDP가 gradient 동기화 훅을 걸 수 있도록 모든 학습 forward는 이 메서드를 거친다."""
        return objective(self, items)

    # ---- 압축
    def encode_units(self, texts: List[str], counts: List[int]) -> List[torch.Tensor]:
        """부분 단위마다 [k, H_dec] 부분 잠재 요약. 인코더 호출은 encoder_token_budget 안에서 읽는 순서대로 묶는다."""
        lens = [min(len(self.encoder.token_ids(t, k)) + k, self.encoder.max_len) for t, k in zip(texts, counts)]
        groups, cur, cur_max = [], [], 0
        for j, n in enumerate(lens):
            if cur and (len(cur) + 1) * max(cur_max, n) > self.encoder_token_budget:
                groups.append(cur); cur, cur_max = [], 0
            cur.append(j); cur_max = max(cur_max, n)
        if cur:
            groups.append(cur)
        out: List[torch.Tensor] = [None] * len(texts)
        for g in groups:
            for j, z in zip(g, self.encoder([texts[j] for j in g], [counts[j] for j in g])):
                out[j] = z
        sizes = [z.shape[0] for z in out]
        return list(torch.split(self.projector(torch.cat(out)), sizes))

    def encode_paper(self, texts: List[str], counts: List[int]) -> torch.Tensor:
        """문헌 한 편 -> [sum k, H_dec]: 부분 잠재 요약을 읽는 순서대로 이어 붙인다."""
        return torch.cat(self.encode_units(texts, counts))

    # ---- 학습 대상과 체크포인트
    def trainable_parameters(self) -> List[nn.Parameter]:
        return list(self.encoder.parameters()) + list(self.projector.parameters()) + list(self.decoder.lora_params)

    def trainable_state_dict(self) -> dict:
        cpu = lambda sd: {k: v.detach().cpu() for k, v in sd.items()}
        return {"config": self.config, "encoder": cpu(self.encoder.model.state_dict()),
                "projector": cpu(self.projector.state_dict()), "lora": self.decoder.lora_state_dict()}

    def load_trainable_state_dict(self, sd: dict):
        self.encoder.model.load_state_dict(sd["encoder"])
        self.projector.load_state_dict(sd["projector"])
        self.decoder.load_lora_state_dict(sd["lora"])


def load_checkpoint(path: str, dtype=torch.bfloat16, device: str = None) -> Tuple[LatentSummarizer, dict]:
    """체크포인트(config + 학습된 가중치)에서 모델을 다시 만든다."""
    sd = torch.load(path, map_location="cpu", weights_only=False)
    cfg = sd["config"]
    model = LatentSummarizer.build(cfg["encoder_id"], cfg["decoder_id"], cfg["lora_rank"], cfg["encoder_max_len"], dtype,
                                   grad_checkpointing=False)
    model.load_trainable_state_dict(sd)
    if device:
        model.to(device)
    return model, cfg
