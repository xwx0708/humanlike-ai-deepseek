"""测试用的假模型：按顺序返回预设的回答，并记下每次收到的请求。"""

import json
from collections.abc import Callable
from typing import Any

from contracts.llm import LLMError, Message, Reply
from llm._http import check_required


class FakeModel:
    def __init__(
        self,
        replies: list[str | dict[str, Any]] | Callable[[list[Message]], str | dict[str, Any]],
        name: str = "fake",
    ):
        """replies：预设回答的列表（dict 表示 JSON 回答），或一个根据消息生成回答的函数。"""
        self.name = name
        self._replies = replies
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[Message],
        *,
        schema: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Reply:
        self.calls.append({"messages": list(messages), "schema": schema})
        if callable(self._replies):
            answer = self._replies(messages)
        elif self._replies:
            answer = self._replies.pop(0)
        else:
            raise LLMError("FakeModel 的预设回答用完了")
        text = answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)
        data = None
        if schema is not None:
            if not isinstance(answer, dict):
                raise LLMError("要求 JSON 输出，但预设回答不是 dict")
            check_required(answer, schema)
            data = answer
        return Reply(text=text, model=self.name, data=data)
