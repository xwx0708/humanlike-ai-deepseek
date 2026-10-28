"""潜意识模型：本地 Ollama 上的 Qwen。"""

import re
import time
from typing import Any

from contracts.llm import Message, Reply
from llm._http import check_not_truncated, check_required, parse_json_object, post_json

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


class OllamaModel:
    def __init__(
        self,
        model: str = "qwen3.8:27b-q4_K_M",
        base_url: str = "http://127.0.0.1:11434",
        keep_alive: str = "30m",
        num_ctx: int | None = 16384,
        think: bool = False,
        timeout: float = 600,
        retries: int = 1,
    ):
        self.name = model
        self._url = base_url.rstrip("/") + "/api/chat"
        self._keep_alive = keep_alive
        self._num_ctx = num_ctx
        self._think = think  # 潜意识的任务以结构化判断为主，默认关掉思考过程，省时间
        self._timeout = timeout  # 首次加载模型约 1.5 分钟，超时给足
        self._retries = retries

    def chat(
        self,
        messages: list[Message],
        *,
        schema: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Reply:
        options: dict[str, Any] = {}
        if self._num_ctx:
            options["num_ctx"] = self._num_ctx
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        body: dict[str, Any] = {
            "model": self.name,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "think": self._think,
            "keep_alive": self._keep_alive,
            "options": options,
        }
        if schema is not None:
            body["format"] = schema  # Ollama 按 schema 约束输出

        start = time.monotonic()
        raw = post_json(self._url, body, timeout=self._timeout, retries=self._retries)
        message = raw.get("message", {})
        text = _THINK.sub("", message.get("content", "")).strip()
        data = None
        if schema is not None:
            check_not_truncated(raw.get("done_reason"), max_tokens)
            data = parse_json_object(text)
            check_required(data, schema)
        return Reply(
            text=text,
            model=raw.get("model", self.name),
            data=data,
            reasoning=message.get("thinking") or "",
            prompt_tokens=raw.get("prompt_eval_count", 0),
            completion_tokens=raw.get("eval_count", 0),
            seconds=time.monotonic() - start,
        )
