"""부분 단위마다 요지 QA를 생성한다 (Qwen3.8-27B, 추론 기능 사용).

문단에서 요약에 남을 주장과 결과(요지)를 찾고, 요지마다 질문, 답, 그 근거가 되는 원문 문장을 만든다.
근거 문장이 원문 문단에 없거나 질문이나 답이 60단어를 넘으면 그 문단만 모아 다시 생성한다(최대 2번).

입력: data/units/
출력: data/synth/raw/gist_qa/<PMCID>.jsonl
실행: python -m pipeline.synthesize.gist_qa --base_url http://localhost:8000/v1
"""
import argparse
import difflib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

from pipeline.common.io import append_jsonl
from pipeline.common.llm import THINKING_XHIGH, ChatClient, load_schema
from pipeline.synthesize.inputs import Unit, done_units, generator_papers, paper_context, raw_path

KIND = "gist_qa"
CHUNK = 12
REGENERATE = 2
MAX_WORDS = 60
EVIDENCE_MIN = 0.85


def norm(s: str) -> str:
    return " ".join(re.sub(r"[^0-9a-z]+", " ", s.lower()).split())


def evidence_match(evidence: str, text: str) -> float:
    """근거가 원문의 (정규화한) 부분 문자열이면 1, 아니면 근거 토큰 가운데 원문과 맞는 블록이 덮는 비율."""
    e, t = norm(evidence), norm(text)
    if e in t:
        return 1.0
    et = e.split()
    if not et:
        return 0.0
    sm = difflib.SequenceMatcher(None, et, t.split(), autojunk=False)
    return sum(b.size for b in sm.get_matching_blocks()) / len(et)


def check_points(points: List[dict], text: str) -> Tuple[List[dict], int]:
    """(검사를 통과한 요지, 떨어진 요지 수)."""
    ok = []
    for pt in points:
        q, a, e = (pt.get(k, "").strip() for k in ("question", "answer", "evidence"))
        if q and a and e and len(q.split()) <= MAX_WORDS and len(a.split()) <= MAX_WORDS \
                and evidence_match(e, text) >= EVIDENCE_MIN:
            ok.append({"question": q, "answer": a, "evidence": e})
    return ok, len(points) - len(ok)


def prompt_fields(paper: dict, chunk: List[Unit]) -> dict:
    """모델에는 청크 안 번호 P0..Pn-1을 보여 주고, 응답의 idx를 부분 단위 idx로 되돌린다."""
    lines = [f"[P{j}] ({sec}) {text}" for j, (_, sec, text) in enumerate(chunk)]
    return {"title": paper["title"] or "(untitled)", "abstract": paper["abstract"],
            "n": len(chunk), "n_minus_1": len(chunk) - 1, "paragraphs": "\n\n".join(lines)}


def ask(llm: ChatClient, schema: dict, paper: dict, chunk: List[Unit], attempt: int = 0) -> Dict[int, Tuple[List[dict], int]]:
    """청크 하나를 요청해 {부분 단위 idx: (통과 요지, 떨어진 수)}. 응답이 깨지면 빈 dict (청크 전체를 다시 요청)."""
    try:
        # 첫 요청은 greedy, 다시 생성할 때는 샘플링해 다른 응답을 받는다
        sampling = None if attempt == 0 else {"temperature": 0.6, "top_p": 0.95, "seed": attempt}
        reply = llm.chat("gist_qa", schema, sampling=sampling, **prompt_fields(paper, chunk))
        got = {p["idx"]: p["points"] for p in reply["paragraphs"]}
    except Exception as e:
        print(f"[gist_qa] {paper['paper_id']} 응답 오류: {str(e)[:150]}", flush=True)
        return {}
    if sorted(got) != list(range(len(chunk))):
        return {}
    return {idx: check_points(got[j], text) for j, (idx, _, text) in enumerate(chunk)}


def run_paper(llm: ChatClient, schema: dict, pool: ThreadPoolExecutor, pid: str) -> int:
    paper, units = paper_context(pid)
    done = done_units(KIND, pid)
    pending = [u for u in units if u[0] not in done]
    best: Dict[int, List[dict]] = {}
    requests: Dict[int, int] = {}

    def finish(unit: Unit) -> None:
        append_jsonl(raw_path(KIND, pid), {"idx": unit[0], "section": unit[1],
                                           "points": best.get(unit[0], []), "requests": requests[unit[0]]})

    n = len(pending)
    for attempt in range(1 + REGENERATE):
        chunks = [pending[c:c + CHUNK] for c in range(0, len(pending), CHUNK)]
        results = list(pool.map(lambda ch: ask(llm, schema, paper, ch, attempt), chunks))
        retry = []
        for chunk, res in zip(chunks, results):
            for unit in chunk:
                idx = unit[0]
                requests[idx] = attempt + 1
                ok, failed = res.get(idx, ([], 1))
                if len(ok) > len(best.get(idx, [])):
                    best[idx] = ok
                if ok and not failed:
                    finish(unit)
                elif attempt == REGENERATE:
                    finish(unit)  # 다시 요청할 횟수를 다 씀: 지금까지 가장 나은 응답을 쓴다
                else:
                    retry.append(unit)
        pending = retry
        if not pending:
            break
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--workers", type=int, default=16, help="동시에 처리할 문헌 수")
    ap.add_argument("--chunk_workers", type=int, default=64, help="동시에 보낼 청크 요청 수")
    ap.add_argument("--limit", type=int, default=0, help="앞에서부터 이 수의 문헌만 (확인용)")
    args = ap.parse_args()
    llm = ChatClient(args.base_url, args.model, THINKING_XHIGH, timeout=14400)
    llm.wait_ready()
    schema = load_schema("gist_qa")
    pids = generator_papers()[: args.limit or None]
    print(f"[gist_qa] 문헌 {len(pids)}편", flush=True)
    with ThreadPoolExecutor(args.chunk_workers) as inner, ThreadPoolExecutor(args.workers) as outer:
        futs = {outer.submit(run_paper, llm, schema, inner, pid): pid for pid in pids}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                print(f"[gist_qa] {i}/{len(pids)} {futs[f]}: 부분 단위 {f.result()}개 생성", flush=True)
            except Exception as e:  # 다시 실행하면 그 문헌을 이어서 한다
                print(f"[gist_qa] {futs[f]} 실패: {e}", flush=True)


if __name__ == "__main__":
    main()
