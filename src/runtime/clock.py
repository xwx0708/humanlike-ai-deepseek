"""时钟：统一的时间来源。代码里不直接调用 time.time()，而是用注入的 Clock，测试时换成 FakeClock。

注意：这是现实时间。记忆模块的"记忆时间"（对话时走得快、不对话就停，D21）在 memory.clock，基于这里的现实时间计算。
"""

import time
from typing import Protocol


class Clock(Protocol):
    def now(self) -> float:
        """当前时间，Unix 时间戳（秒）。"""
        ...


class SystemClock:
    def now(self) -> float:
        return time.time()


class FakeClock:
    """测试和模拟用：时间只在调用 advance / set 时前进。"""

    def __init__(self, start: float = 0.0):
        self._now = start

    def now(self) -> float:
        return self._now

    def advance(self, seconds: float) -> float:
        self._now += seconds
        return self._now

    def set(self, ts: float) -> None:
        self._now = ts
