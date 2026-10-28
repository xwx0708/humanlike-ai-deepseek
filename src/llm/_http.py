"""带重试的 JSON POST，只用标准库。"""

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from contracts.llm import LLMError

RETRY_STATUS = {408, 429, 500, 502, 503, 504}


def post_json(
    url: str,
    body: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 120,
    retries: int = 2,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """网络错误、超时、限流和服务端错误会重试（间隔 1、2、4… 秒）；鉴权等客户端错误直接失败。"""
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req_headers = {"Content-Type": "application/json", **(headers or {})}
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, data=data, headers=req_headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            if e.code not in RETRY_STATUS or attempt == retries:
                raise LLMError(f"HTTP {e.code}：{detail}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt == retries:
                raise LLMError(f"连接失败：{e}") from e
        sleep(2**attempt)
    raise AssertionError("unreachable")


def parse_json_object(text: str) -> dict[str, Any]:
    """解析模型输出的 JSON 对象。容忍外面包了 ``` 代码块。"""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else ""
        s = s.rsplit("```", 1)[0]
    try:
        value = json.loads(s)
    except json.JSONDecodeError as e:
        raise LLMError(f"输出不是合法 JSON：{text[:200]}") from e
    if not isinstance(value, dict):
        raise LLMError(f"输出不是 JSON 对象：{text[:200]}")
    return value


def check_not_truncated(finish_reason: str | None, max_tokens: int | None) -> None:
    if finish_reason == "length":
        raise LLMError(
            f"输出被长度上限截断（max_tokens={max_tokens}），JSON 不完整；调大 max_tokens 或关掉思考"
        )


def check_required(data: dict[str, Any], schema: dict[str, Any]) -> None:
    missing = [k for k in schema.get("required", []) if k not in data]
    if missing:
        raise LLMError(f"JSON 缺少字段：{missing}")
