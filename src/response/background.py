"""和 AstrBot 合成一个进程（D29）：各模块的后台任务登记进共用调度（D27），由插件每分钟跑一次。

记忆的后台任务由记忆模块自己提供（memory.background_job），这里只负责登记。
"""

from collections.abc import Callable

from runtime.scheduler import Scheduler


def make_scheduler(memory_job: Callable[[float], object]) -> Scheduler:
    scheduler = Scheduler()
    scheduler.add("记忆", 60, memory_job)
    return scheduler
