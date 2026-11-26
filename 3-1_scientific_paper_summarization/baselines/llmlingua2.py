"""LLMLingua-2 베이스라인 입력을 만든다.

LLMLingua-2로 문헌 전문에서 중요도가 낮은 토큰을 지워, 평가 문헌의 제목 + 초록 평균 길이(+15토큰)까지 줄인다.
결과가 비면 목표 길이를 늘려 최대 2번 다시 줄인다.

출력: data/baselines/llmlingua2_test.jsonl
실행: python -m baselines.llmlingua2
"""
import numpy as np
import torch
from transformers import AutoTokenizer

from baselines.common import done_papers, out_path, test_papers, write
from scisum.data.units import ABSTRACT_IDX, full_text, load_paper, unit_texts

COMPRESSOR = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
DECODER = "Qwen/Qwen3.5-9B"
EXTRA_TOKENS = 15


def target_tokens(tok, pids) -> int:
    lens = [len(tok(unit_texts(load_paper(p))[ABSTRACT_IDX], add_special_tokens=False)["input_ids"]) for p in pids]
    return int(round(np.mean(lens))) + EXTRA_TOKENS


def compress(comp, text: str, n_tokens: int, target: int) -> str:
    out = comp.compress_prompt(text, target_token=target)["compressed_prompt"]
    for stage in (1, 2):  # 빈 결과: 중간 길이를 거쳐 두 단계로 줄인다
        if out.strip():
            break
        mid = comp.compress_prompt(text, target_token=min(n_tokens // 2, max(4000, target * 10) * stage))["compressed_prompt"]
        if mid.strip():
            out = comp.compress_prompt(mid, target_token=target)["compressed_prompt"]
    return out


def main() -> None:
    from llmlingua import PromptCompressor

    tok = AutoTokenizer.from_pretrained(DECODER)
    pids = test_papers()
    target = target_tokens(tok, pids)
    todo = [p for p in pids if p not in done_papers("llmlingua2")]
    print(f"[llmlingua2] 목표 {target}토큰/문헌, {len(todo)}편 남음", flush=True)
    comp = PromptCompressor(model_name=COMPRESSOR, use_llmlingua2=True, device_map="cuda" if torch.cuda.is_available() else "cpu")
    for i, pid in enumerate(todo, 1):
        text = full_text(load_paper(pid))
        out = compress(comp, text, len(tok(text, add_special_tokens=False)["input_ids"]), target)
        if not out.strip():
            raise SystemExit(f"[llmlingua2] {pid}: 압축 결과가 비어 있음")
        write("llmlingua2", pid, out)
        if i % 50 == 0:
            print(f"[llmlingua2] {i}/{len(todo)}", flush=True)
    print(f"[llmlingua2] 완료 -> {out_path('llmlingua2')}")


if __name__ == "__main__":
    main()
