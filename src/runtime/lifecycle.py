"""全局生命周期：熟悉期 → 启动 → 运行期（设想文档"启动时机：熟悉期"；D24、D26、D27）。

- 熟悉期：不启动记忆系统，在上下文限制内正常聊天，消息照常存档。
- 启动：上下文用到约 6 万 token 时（D43；原为上限的 70%，D26），依次调用各模块登记的启动步骤
  （记忆：整理全部对话、建立概率海；以后情绪：用全部数据算基线）。
- 运行期：遗忘、浮现、整理循环。

状态存在一个小 JSON 文件里（默认 data/lifecycle.json）；path 为 None 时只在内存里（测试用）。
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from runtime import config

FAMILIAR = "熟悉期"
RUNNING = "运行期"
config.section_info(
    "lifecycle",
    "启动机制",
    "熟悉期 → 启动 → 运行期。熟悉期按普通方式带全部聊天记录；用量达到启动 token 数时，一次性整理全部对话、建立概率海。",
)
START_TOKENS = config.value(
    "lifecycle",
    "start_tokens",
    60000,
    "启动 token 数",
    "上下文用量达到这么多 token 就启动记忆系统（D43，用户定：默认六万，可自行调整）。"
    "不按上限的比例算：上限 100 万的模型若等到 70% 才启动，熟悉期会长到这套系统起不了作用。启动只发生一次。",
    lo=5000,
    hi=1000000,
    step=5000,
    unit="token",
)

DEFAULT_PATH = config.data_dir() / "lifecycle.json"


@dataclass
class Lifecycle:
    path: Path | None = DEFAULT_PATH
    phase: str = FAMILIAR
    started_ts: float | None = None
    hooks: list[tuple[str, Callable[[float], object]]] = field(default_factory=list)
    start_tokens: int = field(default_factory=lambda: START_TOKENS)  # 用到这么多 token 就启动（D43）

    def __post_init__(self):
        if self.path is not None and Path(self.path).is_file():
            data = json.loads(Path(self.path).read_text(encoding="utf-8"))
            self.phase, self.started_ts = data["phase"], data.get("started_ts")

    def on_start(self, module: str, fn: Callable[[float], object]) -> None:
        """模块登记自己在"启动"时要做的事，按登记顺序执行。"""
        self.hooks.append((module, fn))

    def should_start(self, context_used: int, context_limit: int = 0) -> bool:
        """上下文用量达到启动 token 数就启动（D43）。context_limit 不再参与判断（D24 的"上限的 70%"已取消），
        参数留着是为了让调用方不用改。"""
        return self.phase == FAMILIAR and context_used >= self.start_tokens

    def start(self, now_ts: float) -> dict[str, object]:
        """执行各模块的启动步骤，然后进入运行期。返回每个模块的结果。"""
        results = {module: fn(now_ts) for module, fn in self.hooks}
        self.phase, self.started_ts = RUNNING, now_ts
        self._save()
        return results

    def maybe_start(self, now_ts: float, context_used: int, context_limit: int) -> dict[str, object] | None:
        if not self.should_start(context_used, context_limit):
            return None
        return self.start(now_ts)

    def _save(self) -> None:
        if self.path is None:
            return
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        data = {"phase": self.phase, "started_ts": self.started_ts}
        Path(self.path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
