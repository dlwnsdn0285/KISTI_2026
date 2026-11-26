"""Keynote 베이스라인 입력을 만든다 (DeepSurvey 방식).

생성 모델(Qwen3.5-9B)이 문헌 전문을 읽고 280단어 이내의 요점 요약(Keynote)을 쓴다. 추론 기능은 끄고 greedy로 생성한다.

출력: data/baselines/keynote_test.jsonl
실행: python -m baselines.keynote
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from baselines.common import done_papers, out_path, test_papers, write
from scisum.data.units import full_text, load_paper

MODEL = "Qwen/Qwen3.5-9B"
WORD_BUDGET = 280
MAX_NEW_TOKENS = 700

KEYNOTE_SYSTEM = "You are a scientific analyst producing structured keynotes of research papers."
KEYNOTE_INSTRUCTION = (
    "Read the paper below in full and write a structured keynote of it.\n\n"
    "The keynote must cover the paper's contributions, methods, experiments, assumptions, and "
    "limitations.\n\n"
    "Do not force a fixed template. Organise the keynote with whatever headings this particular "
    "paper warrants, so the structure reflects what kind of work it is. Under each heading write "
    "compact lines rather than prose paragraphs.\n\n"
    "Preserve specifics verbatim: population and sample sizes, interventions and comparators, "
    "measured outcomes, effect sizes with their intervals and p-values, and named entities. These "
    "are the details a later reader will be asked about, and they are what a generic summary loses."
    "\n\nWrite at most {budget} words. Output only the keynote."
)
KEYNOTE_PAPER_HEADER = "\n\nPAPER:\n"


def prompt(paper_text: str) -> str:
    """Qwen 대화 형식, 추론 끔 (빈 <think> 블록)."""
    user = KEYNOTE_INSTRUCTION.format(budget=WORD_BUDGET) + KEYNOTE_PAPER_HEADER + paper_text
    return (f"<|im_start|>system\n{KEYNOTE_SYSTEM}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")


@torch.no_grad()
def main() -> None:
    todo = [p for p in test_papers() if p not in done_papers("keynote")]
    print(f"[keynote] {len(todo)}편 남음", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    for i, pid in enumerate(todo, 1):
        ids = tok(prompt(full_text(load_paper(pid))), add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        gen = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=MAX_NEW_TOKENS,
                             do_sample=False, pad_token_id=tok.eos_token_id)
        write("keynote", pid, tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True).strip())
        if i % 10 == 0:
            print(f"[keynote] {i}/{len(todo)}", flush=True)
    print(f"[keynote] 완료 -> {out_path('keynote')}")


if __name__ == "__main__":
    main()
