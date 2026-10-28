"""后台调度：各模块把要定期做的事登记进来，由同一个循环按时调用（决策 D27：时钟谁都要）。

例如记忆模块登记"每分钟 tick 一次"（memory.runner），以后情绪模块登记基线更新。
一个任务出错只记下来，不影响别的任务。
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from runtime.clock import Clock, SystemClock


@dataclass
class Job:
    name: str
    every: float  # 秒
    fn: Callable[[float], object]  # 参数是当前时间
    next_ts: float | None = None


@dataclass
class Scheduler:
    clock: Clock = field(default_factory=SystemClock)
    jobs: list[Job] = field(default_factory=list)
    errors: list[tuple[str, float, str]] = field(default_factory=list)  # (任务, 时间, 错误)

    def add(self, name: str, every: float, fn: Callable[[float], object]) -> None:
        self.jobs.append(Job(name, every, fn))

    def run_pending(self) -> list[str]:
        """把到点的任务各跑一次，返回跑了哪些。"""
        now, ran = self.clock.now(), []
        for job in self.jobs:
            if job.next_ts is None or now >= job.next_ts:
                try:
                    job.fn(now)
                except Exception as e:  # noqa: BLE001  一个模块出错不拖垮别的模块，错误记下来
                    self.errors.append((job.name, now, f"{type(e).__name__}: {e}"))
                job.next_ts = now + job.every
                ran.append(job.name)
        return ran

    def run_forever(self, poll: float = 5.0) -> None:
        while True:
            self.run_pending()
            time.sleep(poll)
