"""对话平台的接口（D44）：回应模块组织好每轮的内容，交给某个平台上的思考模型去回答。

不和 AstrBot 绑死：AstrBot 插件是一种平台（思考模型在 AstrBot 里选，能用它的工具和看图）；
命令行是另一种（response.cli，思考模型用本项目 .env 里的配置）。以后别的 QQ 框架、Telegram 也按这个接。
"""

from collections.abc import Sequence
from typing import Protocol


class ThinkingBackend(Protocol):
    """把一轮内容交给思考模型，拿回它说的话。"""

    def reply(
        self, system: str, contexts: list[dict[str, str]], prompt: str, image_paths: Sequence[str] = ()
    ) -> tuple[str, int | None]:
        """system：系统提示词；contexts：之前的对话（只在熟悉期有）；prompt：这一条；image_paths：对方发来的图。
        返回 (回复正文, 这一轮实际用掉的输入 token 数或 None)。"""
