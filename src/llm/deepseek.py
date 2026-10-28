"""思考模型：DeepSeek API（OpenAI 兼容格式）。"""

import json
import time
from typing import Any

from contracts.llm import Message, Reply
from llm._http import check_not_truncated, check_required, parse_json_object, post_json


class DeepSeekModel:
    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-flash",
        base_url: str = "https://api.deepseek.com",
        thinking: bool = False,
        timeout: float = 120,
        retries: int = 2,
    ):
        self.name = model
        self._key = api_key
        # deepseek-flash 默认先思考再回答，思考内容也占 max_tokens，还会拖慢回复；默认关掉
        self._thinking = thinking
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
        body: dict[str, Any] = {
            "model": self.name,
            "messages": msgs,
            "stream": False,
            "thinking": {"type": "enabled" if self._thinking else "disabled"},
        }
        if schema is not None:
            # DeepSeek 的 JSON 模式不接受 schema，只保证输出合法 JSON，所以把 schema 写进提示里
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
        raw = post_json(
            self._url,
            body,
            headers={"Authorization": f"Bearer {self._key}"},
            timeout=self._timeout,
            retries=self._retries,
        )
        choice = raw["choices"][0]
        text = choice["message"].get("content") or ""
        data = None
        if schema is not None:
            check_not_truncated(choice.get("finish_reason"), max_tokens)
            data = parse_json_object(text)
            check_required(data, schema)
        usage = raw.get("usage", {})
        return Reply(
            text=text,
            model=raw.get("model", self.name),
            data=data,
            reasoning=choice["message"].get("reasoning_content") or "",
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            seconds=time.monotonic() - start,
        )
