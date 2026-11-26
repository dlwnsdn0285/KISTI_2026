"""vLLM 서버(OpenAI 호환 API)에 프롬프트를 보내고 응답을 받는다.

프롬프트는 pipeline/prompts/<이름>.md에 있으며, "## SYSTEM" 아래는 시스템 프롬프트, "## USER" 아래는 사용자 프롬프트다.
사용자 프롬프트의 {필드} 자리를 값으로 채워 보내고, JSON 스키마를 주면 응답을 dict로 바꿔 돌려준다.
"""
import json
import os
import re
import time
import urllib.request
from typing import Optional, Tuple, Union

PROMPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts")

THINKING_OFF = {"chat_template_kwargs": {"enable_thinking": False}}
THINKING_XHIGH = {"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "xhigh"}}


def load_prompt(name: str) -> Tuple[str, str]:
    """(system, user 틀). system이 비어 있으면 시스템 메시지 없이 보낸다."""
    with open(os.path.join(PROMPT_DIR, f"{name}.md"), encoding="utf-8") as f:
        text = f.read()
    system = text.split("## SYSTEM", 1)[1].split("## USER", 1)[0].strip()
    return system, text.split("## USER", 1)[1].strip()


def load_schema(name: str) -> dict:
    with open(os.path.join(PROMPT_DIR, f"{name}.schema.json"), encoding="utf-8") as f:
        return json.load(f)


def fill(template: str, **fields) -> str:
    for k, v in fields.items():
        template = template.replace("{" + k + "}", str(v))
    return template


class ChatClient:
    """vLLM 서버 하나에 대한 호출기. 실패하면 retries번 다시 시도한다."""

    def __init__(self, base_url: str, model: str, extra: Optional[dict] = None,
                 timeout: int = 3600, retries: int = 2):
        self.url = base_url.rstrip("/")
        self.model = model
        self.extra = extra or {}
        self.timeout = timeout
        self.retries = retries

    def wait_ready(self, max_wait: int = 5400) -> None:
        """서버가 뜰 때까지 30초마다 확인한다."""
        for _ in range(max_wait // 30):
            try:
                urllib.request.urlopen(self.url + "/models", timeout=5)
                return
            except Exception:
                print("[llm] 서버 대기 중", flush=True)
                time.sleep(30)
        raise SystemExit(f"[llm] 서버 응답 없음: {self.url}")

    def chat(self, prompt: str, schema: Optional[dict] = None, max_tokens: Optional[int] = None,
             sampling: Optional[dict] = None, **fields) -> Union[str, dict]:
        """prompt 파일 하나로 한 번 호출한다. 기본은 greedy(temperature 0), sampling으로 바꿀 수 있다.
        max_tokens가 None이면 서버 한도까지 생성한다 (추론을 켠 호출)."""
        system, user = load_prompt(prompt)
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": fill(user, **fields)})
        body = {"model": self.model, "temperature": 0, "messages": messages, **self.extra, **(sampling or {})}
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": prompt, "schema": schema}}
        last = None
        for _ in range(self.retries + 1):
            try:
                text = self._post(body)
                return text if schema is None else parse_json(text)
            except Exception as e:  # 네트워크 오류나 잘린 JSON: 잠시 뒤 다시
                last = e
                time.sleep(3)
        raise RuntimeError(f"{prompt}: {type(last).__name__}: {str(last)[:200]}")

    def _post(self, body: dict) -> str:
        req = urllib.request.Request(self.url + "/chat/completions", data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'EMPTY')}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            out = json.load(r)
        choice = out["choices"][0]
        if choice.get("finish_reason") == "length":   # 토큰 한도에서 잘린 응답은 실패로 보고 다시 요청한다
            raise ValueError("응답이 max_tokens에서 잘림")
        content = choice["message"].get("content") or ""
        # 서버가 추론 부분을 따로 떼지 않았으면 마지막 </think> 뒤만 쓴다 (여는 태그가 프롬프트에 있는 경우 포함)
        return content.rsplit("</think>", 1)[-1].strip()


def parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, flags=re.S)
    return json.loads(m.group(0) if m else text)
