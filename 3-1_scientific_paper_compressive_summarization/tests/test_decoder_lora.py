"""잠재 표현용 LoRA가 잠재 표현 위치에만 적용되는지 확인한다."""
import pytest
import torch

from scisum import prompts
from scisum.model.decoder import LatentDecoder, LatentPositions, LatentQKVLoRA


@pytest.fixture(scope="module")
def decoders(tiny):
    torch.manual_seed(1)
    lora = LatentDecoder.from_pretrained(tiny["decoder"], lora_rank=4, dtype=torch.float32, grad_checkpointing=False)
    for p in lora.lora_params:   # B=0 초기값이면 LoRA가 아무것도 바꾸지 않으므로 무작위 값으로 켠다
        torch.nn.init.normal_(p, std=0.5)
    base = LatentDecoder.from_pretrained(tiny["decoder"], lora_rank=0, dtype=torch.float32, grad_checkpointing=False)
    return lora, base


def test_lora_only_on_full_attention_qkv(decoders):
    lora, _ = decoders
    layers = lora.model.model.layers
    wrapped = [(i, n) for i, layer in enumerate(layers) for n, m in layer.named_modules() if isinstance(m, LatentQKVLoRA)]
    assert wrapped == [(1, "self_attn.q_proj"), (1, "self_attn.k_proj"), (1, "self_attn.v_proj")]   # 층 0은 선형 어텐션
    assert len(lora.lora_params) == 6 and all(p.requires_grad for p in lora.lora_params)
    assert not any(p.requires_grad for n, p in lora.model.named_parameters() if ".lora_" not in n)


def test_lora_module_switches_by_position():
    base = torch.nn.Linear(8, 6)
    pos = LatentPositions()
    m = LatentQKVLoRA(base, pos, rank=2)
    torch.nn.init.normal_(m.lora_B.weight)
    x = torch.randn(1, 4, 8)
    pos.mask = torch.tensor([[False, True, True, False]])
    y = m(x)
    assert torch.allclose(y[0, [0, 3]], base(x)[0, [0, 3]])
    assert not torch.allclose(y[0, 1:3], base(x)[0, 1:3])
    pos.mask = None
    assert torch.allclose(m(x), base(x))


def test_text_only_input_equals_base_model(decoders):
    lora, base = decoders
    segs = [prompts.qa_segments("What is studied?", ["A paper about lungs."], "Lung development.")]
    assert torch.allclose(lora.target_nll(segs), base.target_nll(segs), atol=1e-5)


def test_tokens_before_latents_are_unchanged(decoders):
    """잠재 표현 앞의 지시 토큰은 원래 모델과 같은 은닉 상태, 잠재 표현 위치부터는 달라진다."""
    lora, base = decoders
    z = torch.randn(3, lora.embed.weight.shape[1]) * 0.05
    segs = [prompts.qa_segments("What is studied?", [z], "Lung development.")]
    hs = []
    for d in (lora, base):
        embeds, attn, _, latent = d.collate(segs, left_pad=False)
        d.positions.mask = latent
        with torch.no_grad():
            hs.append(d.model.model(inputs_embeds=embeds, attention_mask=attn).last_hidden_state[0])
    start = int(latent[0].nonzero()[0])
    assert torch.allclose(hs[0][:start], hs[1][:start], atol=1e-5)
    assert not torch.allclose(hs[0][start:start + 3], hs[1][start:start + 3])


def test_lora_state_roundtrip(decoders, tiny):
    lora, _ = decoders
    sd = lora.lora_state_dict()
    assert len(sd) == 6 and all(".lora_" in k for k in sd)
    fresh = LatentDecoder.from_pretrained(tiny["decoder"], lora_rank=4, dtype=torch.float32, grad_checkpointing=False)
    fresh.load_lora_state_dict(sd)
    assert all(torch.equal(a, b) for a, b in zip(fresh.lora_params, lora.lora_params))
