"""테스트용 작은 무작위 인코더와 생성 모델을 만든다.

토크나이저와 모델 구조는 실제 모델(Qwen3-Embedding-0.6B, Qwen3.5-9B)과 같고, 층 수와 크기만 아주 작다.

실행: python tests/smoke/tiny_models.py <출력 폴더>   (<출력 폴더>/encoder, <출력 폴더>/decoder 생성)
"""
import os
import sys

import torch
from transformers import AutoConfig, AutoTokenizer, Qwen3Config, Qwen3Model, Qwen3_5ForConditionalGeneration


def tiny_encoder(path: str) -> None:
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, max_position_embeddings=4096)
    torch.manual_seed(0)
    Qwen3Model(cfg).save_pretrained(path)
    tok.save_pretrained(path)


def tiny_decoder(path: str) -> None:
    cfg = AutoConfig.from_pretrained("Qwen/Qwen3.5-9B")
    for k, v in dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                     head_dim=8, layer_types=["linear_attention", "full_attention"], linear_num_key_heads=2, linear_num_value_heads=2,
                     linear_key_head_dim=8, linear_value_head_dim=8).items():
        setattr(cfg.text_config, k, v)
    for k, v in dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=32, num_position_embeddings=16).items():
        setattr(cfg.vision_config, k, v)
    torch.manual_seed(0)
    Qwen3_5ForConditionalGeneration(cfg).save_pretrained(path)
    AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B").save_pretrained(path)


if __name__ == "__main__":
    out = sys.argv[1]
    tiny_encoder(os.path.join(out, "encoder"))
    tiny_decoder(os.path.join(out, "decoder"))
    print(f"[tiny] {out}/encoder, {out}/decoder")
