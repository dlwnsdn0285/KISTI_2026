"""생성한 답을 M3SciQA 평가 프롬프트로 채점한다.

평가 모델(gpt-oss-20b)이 답과 정답을 비교해 같은 의미면 1점, 부분적으로 맞으면 0.5점, 틀리면 0점을 주고, 평균을 정답률로 보고한다.
채점 결과는 한 문항씩 저장하며, 다시 실행하면 채점한 문항은 건너뛴다.

입력: 답 파일 (scisum.infer.answer 출력)
출력: <답 파일>.judged.jsonl, <답 파일>.judged.summary.json
실행: python -m scisum.eval.judge --preds runs/eval/ours.jsonl --base_url http://localhost:8000/v1
"""
import argparse
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple

# M3SciQA (Li et al., 2024) Appendix Table 16, 0-0.5-1 설정의 프롬프트 원문 그대로 ({{ }}는 format용 이스케이프)
M3SCIQA_JUDGE_PROMPT = (
    "I am testing a model's performance on open-ended questions. I want you to help me in checking to see "
    "if the candidate answer has the same meaning as the reference answer for a given question. If you think "
    "the reference answer and the candidate answer have the same meaning, respond {{\"selection\": \"1\"}}; "
    "otherwise, respond by {{\"selection\": \"0\"}}. If you think the candidate is partially correct, respond "
    "by {{\"selection\": \"0.5\"}}. If the answer is \"I don't know,\" rate it to 0.\n"
    "<QUESTION> {q} </QUESTION>\n<REFERENCE> {a} </REFERENCE>\n<CANDIDATE> {p} </CANDIDATE>"
)
SCORES = {"1": 1.0, "1.0": 1.0, "0.5": 0.5, "0": 0.0, "0.0": 0.0}
RETRIES = 4


def parse_score(reply: str) -> Optional[float]:
    """응답에서 점수를 읽는다: 마지막 {"selection": "x"}, 없으면 점수만 있는 응답. 읽을 수 없으면 None (judge_one이 다시 묻는다)."""
    reply = reply.strip()
    if not reply:
        return None
    sel = re.findall(r"\"selection\"\s*:\s*\"?\s*(1(?:\.0)?|0\.5|0(?:\.0)?)\s*\"?", reply)
    if sel:
        return SCORES[sel[-1]]
    bare = re.match(r"^\s*(1(?:\.0)?|0\.5|0(?:\.0)?)\s*$", reply)
    if bare:
        return SCORES[bare.group(1)]
    return None


def judge_one(client, model: str, row: dict, temperature: float, extra_body: dict) -> Tuple[Optional[float], str]:
    """한 문항 채점. 호출이 실패하거나 점수를 읽을 수 없는 응답은 다시 시도하고, 끝내 실패하면 점수 None (평균에서 빠지고 실패로 센다)."""
    msg = M3SCIQA_JUDGE_PROMPT.format(q=row["question"], a=row["answer"], p=row.get("pred", ""))
    last = ""
    for _ in range(RETRIES):
        try:
            out = client.chat.completions.create(model=model, messages=[{"role": "user", "content": msg}],
                                                 temperature=temperature, extra_body=extra_body)
            last = out.choices[0].message.content or ""
            score = parse_score(last)
            if score is not None:
                return score, last
        except Exception as e:   # 서버 오류는 재시도
            last = f"{type(e).__name__}: {e}"
    return None, last[:500]


def read_preds(path: str) -> dict:
    """답 파일 {id: 행}. 중간에 끊긴 줄은 건너뛰고, 같은 id가 여러 번 있으면 마지막 것을 쓴다."""
    rows = {}
    for line in open(path, encoding="utf-8"):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        rows[d["id"]] = d
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", required=True, help="infer.answer 출력 jsonl (id, question, answer, pred)")
    ap.add_argument("--base_url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--reasoning_effort", default="medium")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--out", default=None, help="채점 jsonl (기본 <preds>.judged.jsonl)")
    ap.add_argument("--summary", default=None, help="요약 json (기본 <preds>.judged.summary.json)")
    args = ap.parse_args()
    from openai import OpenAI
    out = args.out or args.preds.replace(".jsonl", "") + ".judged.jsonl"
    summary = args.summary or out.replace(".jsonl", "") + ".summary.json"

    rows = list(read_preds(args.preds).values())
    done = {}
    if os.path.exists(out):
        for line in open(out, encoding="utf-8"):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("score") is not None:
                done[d["id"]] = d
    todo = [r for r in rows if r["id"] not in done]
    client = OpenAI(api_key="EMPTY", base_url=args.base_url)
    # gpt-oss 추론 강도: vLLM 버전에 따라 요청 최상위 필드 또는 채팅 템플릿 인자로 읽으므로 둘 다 넘긴다
    extra = {"reasoning_effort": args.reasoning_effort, "chat_template_kwargs": {"reasoning_effort": args.reasoning_effort}}
    lock = threading.Lock()
    with open(out, "a", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(judge_one, client, args.model, r, args.temperature, extra): r for r in todo}
        for fut in as_completed(futs):
            r = futs[fut]
            score, reply = fut.result()
            d = {**r, "score": score, "judge_reply": reply}
            with lock:
                f.write(json.dumps(d, ensure_ascii=False) + "\n"); f.flush()
            if score is not None:
                done[r["id"]] = d

    scores = [done[r["id"]]["score"] for r in rows if r["id"] in done]
    res = {"preds": args.preds, "judge": args.model, "n": len(rows), "n_scored": len(scores), "n_failed": len(rows) - len(scores),
           "accuracy": sum(scores) / len(scores) if scores else None}
    with open(summary, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res))
    if res["n_failed"]:
        print(f"[judge] 채점 실패 {res['n_failed']}문항은 평균에서 빠졌다. 다시 실행하면 그 문항만 다시 채점한다.")


if __name__ == "__main__":
    main()
