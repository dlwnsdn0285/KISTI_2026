"""스모크 테스트용 모의 LLM 서버 (OpenAI 호환 API).

학습 데이터 생성(Qwen3.8-27B)과 채점(gpt-oss-20b) 서버 대신, 각 프롬프트에 형식만 맞춘 응답을 돌려준다.
다시 생성하는 경로도 확인되도록 일부 응답은 일부러 틀리게 준다.
- 보충 질문: 문단마다 첫 번째 보충 라운드에서만 질문 1개
- 요지 QA: 문단 4개 중 1개는 처음 요청에 원문에 없는 근거 문장
- 재진술: 첫 생성 4개 중 1개는 검증에서 "No"

실행: python tests/smoke/mock_llm_server.py --port 8099
"""
import argparse
import hashlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SEEN = set()
LOCK = threading.Lock()


def first_time(key: str) -> bool:
    with LOCK:
        if key in SEEN:
            return False
        SEEN.add(key)
        return True


def bucket(text: str, n: int) -> int:
    return int(hashlib.md5(text.encode()).hexdigest(), 16) % n


def gist_reply(user: str) -> dict:
    """[P0] (섹션) 문단 ... 형식의 문단마다 요지 1개, 근거는 문단 앞 12단어."""
    paras = []
    for m in re.finditer(r"\[P(\d+)\] \((.*?)\) (.*?)(?=\n\n\[P\d+\] |\Z)", user, flags=re.S):
        j, text = int(m.group(1)), m.group(3).strip()
        evidence = " ".join(text.split()[:12])
        if bucket(text, 4) == 0 and first_time("gist:" + text[:200]):
            evidence = "this sentence is not in the paragraph at all"
        paras.append({"idx": j, "points": [{"question": f"What does paragraph P{j} report?",
                                            "answer": "It reports " + " ".join(text.split()[:6]) + ".", "evidence": evidence}]})
    return {"paragraphs": paras}


def reply_for(body: dict) -> str:
    user = body["messages"][-1]["content"]
    rf = body.get("response_format") or {}
    name = (rf.get("json_schema") or {}).get("name", "")
    if name in ("atomic_questions", "complex_questions"):
        return json.dumps({"Question1": f"What is the main finding ({name.split('_')[0]})?", "Question2": "Which method was used?"})
    if name.startswith("more_"):                      # 보충 라운드: 문단마다 첫 라운드에만 질문 1개, 그다음은 없음 -> 종료
        para = user.split("Q: ")[0][-300:]
        return json.dumps({"Question1": "What else does the paragraph state?"}) if first_time(name + para) else "{}"
    if name == "validate_answer":
        return json.dumps({"Judgment": "Correct"})
    if name == "gist_qa":
        return json.dumps(gist_reply(user))
    if "selection" in user:                           # M3SciQA 채점
        return '{"selection": "1"}' if bucket(user, 3) else '{"selection": "0.5"}'
    if "Paraphrased Document" in user:                # 재진술 검증
        return "No" if "[greedy-rejected]" in user else "Yes"
    if "paraphrase the document" in user:            # 재진술
        tag = "[greedy-rejected] " if body.get("temperature", 0) == 0 and bucket(user, 4) == 0 else ""
        return tag + "In summary, the document states the following. " + user[-300:]
    if "factual answering assistant" in user:         # 질문에 대한 답
        return "A short answer from the document."
    return "OK"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, obj: dict) -> None:
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send({"object": "list", "data": [{"id": "mock", "object": "model"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        content = reply_for(body)
        self._send({"id": "mock", "object": "chat.completion", "model": body.get("model", "mock"),
                    "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8099)
    args = ap.parse_args()
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
