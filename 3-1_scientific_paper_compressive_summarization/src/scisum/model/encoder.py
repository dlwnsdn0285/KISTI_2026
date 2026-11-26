"""인코더(Qwen3-Embedding-0.6B): 부분 단위 하나를 k개의 잠재 표현으로 압축한다.

문단 토큰 뒤에 슬롯 k개를 붙여 넣고, 각 슬롯 위치의 출력을 잠재 표현으로 쓴다.
슬롯 입력은 문단을 k개 구간으로 나눈 각 구간의 평균 임베딩이라, 슬롯마다 서로 다른 구간을 맡는다.
인과 마스크를 없애 모든 토큰과 슬롯이 서로를 참조한다(양방향 어텐션).
"""
from typing import List, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


def chunk_mean_slots(token_embeds: torch.Tensor, k: int) -> torch.Tensor:
    """[n, H] 토큰 임베딩 -> [k, H] 슬롯 입력. 경계는 linspace(0, n, k+1)의 반올림, 빈 청크(n < k)는 전체 평균으로 채운다."""
    n = token_embeds.shape[0]
    bounds = torch.linspace(0, n, k + 1).round().long().tolist()
    whole = token_embeds.mean(0)
    return torch.stack([token_embeds[a:b].mean(0) if b > a else whole for a, b in zip(bounds[:-1], bounds[1:])])


def bidirectional_mask(lengths: List[int], width: int, device=None) -> torch.Tensor:
    """[B, 1, L, L] bool 마스크: 행마다 실제 위치(토큰 + 슬롯)끼리 서로 모두 참조한다. 패딩 키는 가린다.

    4차원 마스크를 넘기면 transformers가 인과 마스크를 만들지 않고 그대로 SDPA에 전달한다.
    """
    keep = torch.arange(width, device=device)[None, :] < torch.tensor(lengths, device=device)[:, None]
    return keep[:, None, None, :].expand(len(lengths), 1, width, width)


class ParagraphEncoder(nn.Module):
    """Qwen3-Embedding 기반 요약 인코더."""

    def __init__(self, model: nn.Module, tokenizer, max_len: int = 2048):
        super().__init__()
        self.model, self.tokenizer, self.max_len = model, tokenizer, max_len
        self.hidden_size = model.config.hidden_size

    @classmethod
    def from_pretrained(cls, model_id: str, max_len: int = 2048, dtype=torch.bfloat16, grad_checkpointing: bool = True):
        model = AutoModel.from_pretrained(model_id, dtype=dtype, attn_implementation="sdpa")
        if grad_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return cls(model, AutoTokenizer.from_pretrained(model_id), max_len)

    @property
    def device(self) -> torch.device:
        return self.model.get_input_embeddings().weight.device

    def token_ids(self, text: str, k: int) -> List[int]:
        """부분 단위의 토큰 id. 슬롯 k개가 들어갈 자리를 남기고 max_len에 맞춰 자른다."""
        return self.tokenizer(text, truncation=True, max_length=self.max_len - k)["input_ids"]

    def build_inputs(self, texts: List[str], counts: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(inputs_embeds [B, L, H], 4D 양방향 마스크, 슬롯 위치 bool [B, L])."""
        ids = [self.token_ids(t, k) for t, k in zip(texts, counts)]
        emb_layer = self.model.get_input_embeddings()
        lengths = [len(x) + k for x, k in zip(ids, counts)]
        width = max(lengths)
        embeds = torch.zeros(len(texts), width, self.hidden_size, dtype=emb_layer.weight.dtype, device=self.device)
        slot_pos = torch.zeros(len(texts), width, dtype=torch.bool, device=self.device)
        for i, (x, k) in enumerate(zip(ids, counts)):
            tok = emb_layer(torch.tensor(x, device=self.device))
            n = tok.shape[0]
            embeds[i, :n] = tok
            embeds[i, n:n + k] = chunk_mean_slots(tok, k)
            slot_pos[i, n:n + k] = True
        return embeds, bidirectional_mask(lengths, width, self.device), slot_pos

    def forward(self, texts: List[str], counts: List[int]) -> List[torch.Tensor]:
        """부분 단위 i -> [counts[i], H_enc] 잠재 표현."""
        embeds, mask, slot_pos = self.build_inputs(texts, counts)
        hidden = self.model(inputs_embeds=embeds, attention_mask=mask, use_cache=False).last_hidden_state
        return [hidden[i][slot_pos[i]] for i in range(len(texts))]
