"""생성 모델(Qwen3.5-9B)과 잠재 표현용 LoRA.

생성 모델의 파라미터는 모두 고정하고, 어텐션의 Q/K/V 선형 층에 LoRA(rank 32)를 붙인다.
LoRA는 잠재 표현 위치에만 적용되며, 질문과 지시문 토큰은 원래 가중치를 그대로 쓴다.
"""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ..prompts import END_OF_TURN, Segment


class LatentPositions:
    """모든 LoRA 층이 공유하는 잠재 표현 위치 마스크 [B, L] (bool). None이면 모든 위치가 원래 가중치를 쓴다."""

    def __init__(self):
        self.mask: Optional[torch.Tensor] = None


class LatentQKVLoRA(nn.Module):
    """y = base(x) + B(A(x)), 단 잠재 표현 위치에서만. B는 0으로 초기화해 학습 시작 시 원래 모델과 같다."""

    def __init__(self, base: nn.Linear, positions: LatentPositions, rank: int):
        super().__init__()
        self.base, self.positions = base, positions
        out_dim, in_dim = base.weight.shape
        dev, dt = base.weight.device, base.weight.dtype
        self.lora_A = nn.Linear(in_dim, rank, bias=False).to(dev, dt)
        self.lora_B = nn.Linear(rank, out_dim, bias=False).to(dev, dt)
        nn.init.normal_(self.lora_A.weight, std=1e-3)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        m = self.positions.mask
        # 마스크가 없거나 모양이 다르면(생성 중 KV 캐시 단계: 새 토큰은 잠재 표현이 아님) 원래 가중치만 쓴다
        if m is None or tuple(m.shape) != tuple(x.shape[:2]):
            return out
        delta = self.lora_B(self.lora_A(x))
        return torch.where(m.to(x.device).unsqueeze(-1), out + delta, out)


def load_causal_lm(model_id: str, dtype=torch.bfloat16) -> nn.Module:
    """텍스트 생성 모델만 불러온다. Qwen3.5처럼 멀티모달 체크포인트는 텍스트 부분(model.language_model.*)만
    *ForCausalLM 이름으로 바꿔 읽어 비전 타워를 만들지 않는다."""
    config = AutoConfig.from_pretrained(model_id)
    kw = dict(dtype=dtype, attn_implementation="sdpa")
    if getattr(config, "text_config", None) is not None and getattr(config, "vision_config", None) is not None:
        kw.update(config=config.text_config, key_mapping={r"^model\.language_model\.": "model."})
    return AutoModelForCausalLM.from_pretrained(model_id, **kw)


class LatentDecoder(nn.Module):
    """세그먼트 목록을 inputs_embeds로 조립해 고정된 생성 모델에 넣고, 정답 토큰 NLL 또는 생성 결과를 돌려준다."""

    def __init__(self, model: nn.Module, tokenizer, lora_rank: int = 32):
        super().__init__()
        self.model, self.tokenizer = model, tokenizer
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.positions = LatentPositions()
        self.lora_rank = lora_rank
        self.lora_params = self._add_latent_lora(lora_rank) if lora_rank > 0 else []
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    @classmethod
    def from_pretrained(cls, model_id: str, lora_rank: int = 32, dtype=torch.bfloat16, grad_checkpointing: bool = True):
        model = load_causal_lm(model_id, dtype)
        if grad_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return cls(model, AutoTokenizer.from_pretrained(model_id), lora_rank)

    def _add_latent_lora(self, rank: int) -> List[nn.Parameter]:
        """전체 어텐션 층의 q/k/v_proj를 LatentQKVLoRA로 바꾼다. Qwen3.5의 선형 어텐션 층(q/k/v_proj 없음)은 건너뛴다."""
        params = []
        for layer in self.model.model.layers:
            attn = getattr(layer, "self_attn", None)
            if attn is None or not all(hasattr(attn, n) for n in ("q_proj", "k_proj", "v_proj")):
                continue
            for name in ("q_proj", "k_proj", "v_proj"):
                lora = LatentQKVLoRA(getattr(attn, name), self.positions, rank)
                setattr(attn, name, lora)
                params += [lora.lora_A.weight, lora.lora_B.weight]
        return params

    def lora_state_dict(self) -> dict:
        return {k: v.detach().cpu() for k, v in self.model.state_dict().items() if ".lora_" in k}

    def load_lora_state_dict(self, sd: dict):
        missing = [k for k in sd if k not in self.model.state_dict()]
        if missing:
            raise KeyError(f"LoRA 키가 모델에 없음: {missing[:3]}")
        self.model.load_state_dict(sd, strict=False)

    @property
    def device(self) -> torch.device:
        return self.model.get_input_embeddings().weight.device

    @property
    def embed(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    # ---- 입력 조립
    def embed_segments(self, segments: List[Segment]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """한 예제 -> (임베딩 [L, H], 레이블 [L] (정답 토큰 외 -100), 잠재 표현 위치 [L] bool)."""
        embs, labels, latent = [], [], []
        for kind, payload in segments:
            if kind == "latent":
                e = payload.to(self.device, self.embed.weight.dtype)
                lab = torch.full((e.shape[0],), -100, dtype=torch.long, device=self.device)
                is_latent = torch.ones(e.shape[0], dtype=torch.bool, device=self.device)
            else:
                ids = torch.tensor(self.tokenizer(payload, add_special_tokens=False)["input_ids"], dtype=torch.long, device=self.device)
                e = self.embed(ids)
                lab = ids.clone() if kind == "target" else torch.full_like(ids, -100)
                is_latent = torch.zeros(ids.shape[0], dtype=torch.bool, device=self.device)
            embs.append(e); labels.append(lab); latent.append(is_latent)
        return torch.cat(embs), torch.cat(labels), torch.cat(latent)

    def collate(self, batch: List[List[Segment]], left_pad: bool) -> Tuple[torch.Tensor, ...]:
        """(inputs_embeds, attention_mask, labels, latent_mask). 학습은 오른쪽, 생성은 왼쪽 패딩."""
        parts = [self.embed_segments(s) for s in batch]
        L, H = max(p[0].shape[0] for p in parts), self.embed.weight.shape[1]
        embeds = torch.zeros(len(batch), L, H, dtype=self.embed.weight.dtype, device=self.device)
        attn = torch.zeros(len(batch), L, dtype=torch.long, device=self.device)
        labels = torch.full((len(batch), L), -100, dtype=torch.long, device=self.device)
        latent = torch.zeros(len(batch), L, dtype=torch.bool, device=self.device)
        for i, (e, lab, lat) in enumerate(parts):
            n = e.shape[0]
            sl = slice(L - n, L) if left_pad else slice(0, n)
            embeds[i, sl], attn[i, sl], labels[i, sl], latent[i, sl] = e, 1, lab, lat
        return embeds, attn, labels, latent

    # ---- 학습: 정답 토큰 NLL
    def target_nll(self, batch: List[List[Segment]]) -> torch.Tensor:
        """정답 토큰마다의 NLL (fp32, 1차원). 로짓은 정답 토큰을 예측하는 위치에서만 계산한다(HF .loss 미사용).

        잠재 표현 위치 마스크는 backward가 끝날 때까지 그대로 둔다: gradient checkpointing이 backward 중에 forward를
        다시 계산할 때 같은 마스크를 읽어야 하기 때문이다(한 backward에 decoder forward는 한 번).
        """
        embeds, attn, labels, latent = self.collate(batch, left_pad=False)
        self.positions.mask = latent
        hidden = self.model.model(inputs_embeds=embeds, attention_mask=attn, use_cache=False).last_hidden_state
        targets = labels[:, 1:]
        sel = targets.ne(-100)
        logits = self.model.lm_head(hidden[:, :-1][sel]).float()
        return F.cross_entropy(logits, targets[sel], reduction="none")

    # ---- 추론: 탐욕적 생성
    def stop_ids(self) -> List[int]:
        ids = [self.tokenizer.convert_tokens_to_ids(t) for t in (END_OF_TURN, "<|endoftext|>")]
        return [i for i in ids if isinstance(i, int) and i >= 0 and i != self.tokenizer.unk_token_id]

    @torch.no_grad()
    def generate(self, batch: List[List[Segment]], max_new_tokens: int = 256) -> List[str]:
        embeds, attn, _, latent = self.collate(batch, left_pad=True)
        self.positions.mask = latent   # 프롬프트 전체를 처리하는 첫 단계에서만 모양이 맞아 LoRA가 적용된다
        out = self.model.generate(inputs_embeds=embeds, attention_mask=attn, max_new_tokens=max_new_tokens, do_sample=False,
                                  eos_token_id=self.stop_ids(), pad_token_id=self.pad_id)
        self.positions.mask = None
        return [self.tokenizer.decode(o, skip_special_tokens=True).strip() for o in out]
