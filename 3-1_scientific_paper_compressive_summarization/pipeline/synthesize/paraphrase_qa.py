"""부분 단위마다 재진술, 단일 사실 QA, 복합 QA를 생성한다 (Qwen3.8-27B, CLaRa의 생성 절차).

질문을 만들고, 질문마다 답을 만든 뒤, 같은 모델이 원문과 대조해 맞는 답만 남긴다.
문단에서 더 물을 사실이 없을 때까지 보충 질문을 최대 10번 만든다.
재진술은 같은 모델이 원문과 비교해 정보가 빠지거나 더해지지 않았는지 확인하고, 통과하지 못하면 최대 3번 다시 쓴다.

입력: data/units/
출력: data/synth/raw/paraphrase_qa/<PMCID>.jsonl
실행: python -m pipeline.synthesize.paraphrase_qa --base_url http://localhost:8000/v1
"""
import argparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

from pipeline.common.io import append_jsonl
from pipeline.common.llm import THINKING_OFF, ChatClient
from pipeline.synthesize.inputs import done_units, generator_papers, paper_context, raw_path

KIND = "paraphrase_qa"
MAX_ROUNDS = 10
PARAPHRASE_RETRIES = 3
NOT_IN_DOC = "does not contain this information"
QUESTIONS = {"type": "object", "additionalProperties": {"type": ["string", "integer"]}}
JUDGE = {"type": "object", "properties": {"Judgment": {"type": "string", "enum": ["Correct", "Incorrect"]}},
         "required": ["Judgment"], "additionalProperties": False}

QA = Tuple[str, str]


class UnitGenerator:
    def __init__(self, llm: ChatClient, pool: ThreadPoolExecutor):
        self.llm = llm
        self.pool = pool  # 답과 검증 호출을 동시에 보내는 공용 풀

    def questions(self, prompt: str, **fields) -> List[str]:
        """{"Question1": ..., "Question2": ...} 응답에서 질문을 번호 순서대로 꺼낸다."""
        obj = self.llm.chat(prompt, QUESTIONS, 2048, **fields) or {}
        qs = [(int(re.sub(r"\D", "", k) or 0), v.strip()) for k, v in obj.items()
              if k.lower().startswith("question") and isinstance(v, str) and v.strip()]
        return [q for _, q in sorted(qs)]

    def answer_and_validate(self, title: str, text: str, qs: List[str]) -> List[QA]:
        answers = self.pool.map(lambda q: self.llm.chat("answer", None, 256, title=title, paragraph=text, question=q), qs)
        pairs = []
        for q, a in zip(qs, answers):
            a = re.sub(r"^\s*Answer\s*:\s*", "", a, flags=re.I).strip()
            if a and NOT_IN_DOC not in a.lower():
                pairs.append((q, a))
        verdicts = self.pool.map(lambda p: self.llm.chat("validate_answer", JUDGE, 32, title=title, paragraph=text,
                                                         question=p[0], answer=p[1]).get("Judgment"), pairs)
        return [p for p, v in zip(pairs, verdicts) if v == "Correct"]

    def qa(self, title: str, text: str) -> Tuple[List[QA], List[QA], bool]:
        """(단일 사실 QA, 복합 QA, 사실을 다 덮었는지)."""
        atomic = self.answer_and_validate(title, text, self.questions("atomic_questions", title=title, paragraph=text))
        complex_ = self.answer_and_validate(title, text, self.questions("complex_questions", title=title, paragraph=text))
        for _ in range(MAX_ROUNDS):
            qa_list = "\n".join(f"Q: {q}\nA: {a}" for q, a in atomic + complex_) or "(none)"
            more_a = self.questions("more_atomic_questions", title=title, paragraph=text, qa_list=qa_list)
            more_c = self.questions("more_complex_questions", title=title, paragraph=text, qa_list=qa_list)
            if not more_a and not more_c:
                return atomic, complex_, True
            atomic += self.answer_and_validate(title, text, more_a)
            complex_ += self.answer_and_validate(title, text, more_c)
        return atomic, complex_, False

    def paraphrase(self, text: str) -> Tuple[Optional[str], int]:
        """(검증을 통과한 첫 재진술 또는 None, 생성 횟수). 첫 시도는 greedy, 이후는 샘플링."""
        tries = [None] + [{"temperature": 0.7, "top_p": 0.95, "seed": s} for s in range(1, PARAPHRASE_RETRIES + 1)]
        for n, sampling in enumerate(tries, 1):
            para = self.llm.chat("paraphrase", None, 2048, sampling, paragraph=text).strip()
            verdict = self.llm.chat("validate_paraphrase", None, 8, paragraph=text, paraphrase=para) if para else ""
            if re.match(r"\W*(output\s*:\s*)?\W*yes\b", verdict, flags=re.I):   # "Yes", **Yes**, Output: Yes
                return para, n
        return None, len(tries)


def run_paper(gen: UnitGenerator, pid: str) -> int:
    paper, units = paper_context(pid)
    done = done_units(KIND, pid)
    n = 0
    for idx, section, text in units:
        if idx in done:
            continue
        try:
            atomic, complex_, complete = gen.qa(paper["title"], text)
            para, tries = gen.paraphrase(text)
        except Exception as e:   # 이 단위만 건너뛰고, 기록하지 않았으므로 다시 실행하면 다시 생성한다
            print(f"[paraphrase_qa] {pid} 단위 {idx} 실패: {str(e)[:150]}", flush=True)
            continue
        append_jsonl(raw_path(KIND, pid), {
            "idx": idx, "section": section,
            "atomic": [{"question": q, "answer": a} for q, a in atomic],
            "complex": [{"question": q, "answer": a} for q, a in complex_],
            "coverage_complete": complete, "paraphrase": para, "paraphrase_tries": tries})
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--workers", type=int, default=32, help="동시에 처리할 문헌 수")
    ap.add_argument("--inner", type=int, default=128, help="동시에 보낼 답과 검증 호출 수")
    ap.add_argument("--limit", type=int, default=0, help="앞에서부터 이 수의 문헌만 (확인용)")
    args = ap.parse_args()
    llm = ChatClient(args.base_url, args.model, THINKING_OFF)
    llm.wait_ready()
    pids = generator_papers()[: args.limit or None]
    print(f"[paraphrase_qa] 문헌 {len(pids)}편", flush=True)
    with ThreadPoolExecutor(args.inner) as inner, ThreadPoolExecutor(args.workers) as outer:
        gen = UnitGenerator(llm, inner)
        futs = {outer.submit(run_paper, gen, pid): pid for pid in pids}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                print(f"[paraphrase_qa] {i}/{len(pids)} {futs[f]}: 부분 단위 {f.result()}개 생성", flush=True)
            except Exception as e:  # 한 문헌의 실패가 전체를 멈추지 않게: 다시 실행하면 그 문헌을 이어서 한다
                print(f"[paraphrase_qa] {futs[f]} 실패: {e}", flush=True)


if __name__ == "__main__":
    main()
