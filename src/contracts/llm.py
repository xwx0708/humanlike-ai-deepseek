"""模型调用的统一接口。

思考模型（DeepSeek API）和潜意识模型（本地 Ollama 上的 Qwen）都实现 ChatModel，
测试时用 llm.fake.FakeModel 代替。各模块只依赖这里的定义，不直接依赖某家的 API。
"""

from dataclasses import dataclass
from typing import Any, Literal, Protocol

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True)
class Reply:
    text: str  # 模型输出的正文（已去掉思考过程）
    model: str  # 实际回应的模型名
    data: dict[str, Any] | None = None  # 要求 JSON 输出时，解析后的结果
    reasoning: str = ""  # 开启思考时的思考过程，仅供调试，不要给用户看
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0


class LLMError(RuntimeError):
    """调用失败：网络、鉴权、超时，或要求 JSON 时输出无法解析。"""


class ChatModel(Protocol):
    name: str

    def chat(
        self,
        messages: list[Message],
        *,
        schema: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Reply:
        """发送一轮对话。给了 schema（JSON Schema）时，要求模型只输出符合它的 JSON，结果放在 Reply.data。"""
        ...


def user(content: str) -> Message:
    return Message("user", content)


def system(content: str) -> Message:
    return Message("system", content)
