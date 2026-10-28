"""任何 OpenAI 兼容接口（/chat/completions）：DeepSeek、通义、Kimi、智谱、OpenRouter、本地 vLLM 等。

只要填地址、密钥和模型名就能用，不绑定某一家（D28）。某家有自己的额外参数时，传 extra_body。
"""

import json
import time
from typing import Any

from contracts.llm import Message, Reply
from llm._http import check_not_truncated, check_required, parse_json_object, post_json


class OpenAICompatModel:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        extra_body: dict[str, Any] | None = None,
        timeout: float = 120,
        retries: int = 2,
    ):
        self.name = model
        self._key = api_key
        self._extra = dict(extra_body or {})
        # 地址可以写到 /v1 为止，也可以不带；只补 /chat/completions
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._timeout = timeout
        self._retries = retries

    def chat(
        self,
        messages: list[Message],
        *,
        schema: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Reply:
        msgs = [{"role": m.role, "content": m.content} for m in messages]
        body: dict[str, Any] = {"model": self.name, "messages": msgs, "stream": False, **self._extra}
        if schema is not None:
            # 各家对 json_schema 的支持参差不齐；只要求合法 JSON，把 schema 写进提示里
            body["response_format"] = {"type": "json_object"}
            msgs.insert(
                0,
                {
                    "role": "system",
                    "content": "只输出一个符合以下 JSON Schema 的 json 对象，不要输出其他内容：\n"
                    + json.dumps(schema, ensure_ascii=False),
                },
            )
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens

        start = time.monotonic()
        headers = {"Authorization": f"Bearer {self._key}"} if self._key else {}
        raw = post_json(self._url, body, headers=headers, timeout=self._timeout, retries=self._retries)
        choice = raw["choices"][0]
        text = choice["message"].get("content") or ""
        data = None
        if schema is not None:
            check_not_truncated(choice.get("finish_reason"), max_tokens)
            data = parse_json_object(text)
            check_required(data, schema)
        usage = raw.get("usage") or {}
        return Reply(
            text=text,
            model=raw.get("model", self.name),
            data=data,
            reasoning=choice["message"].get("reasoning_content") or "",
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            seconds=time.monotonic() - start,
        )
