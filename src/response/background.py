"""和 AstrBot 合成一个进程（D29）：记忆的后台任务登记进共用调度（D27），由插件每分钟跑一次。

后台任务在另一个线程里跑（整理要调用本地 Qwen，可能要几分钟，不能卡住收发消息），
SQLite 连接不能跨线程，所以每次自己开一个、用完关掉。
"""

from collections.abc import Callable
from pathlib import Path

from contracts.llm import ChatModel
from memory.lifecycle import register
from memory.runner import TickReport, tick
from memory.store import DEFAULT_PATH, MemoryStore
from runtime.lifecycle import DEFAULT_PATH as LIFECYCLE_PATH
from runtime.lifecycle import Lifecycle
from runtime.scheduler import Scheduler


def memory_job(
    model: ChatModel,
    context_limit: int,
    db_path: str | Path = DEFAULT_PATH,
    lifecycle_path: Path | None = LIFECYCLE_PATH,
) -> Callable[[float], TickReport]:
    """记忆模块每分钟要做的事：熟悉期检查是否该启动；运行期停顿后整理、对话中途分批整理、夜间合并。"""

    def run(now_ts: float) -> TickReport:
        store = MemoryStore(db_path)
        try:
            life = Lifecycle(path=lifecycle_path)
            register(life, store, model)
            return tick(store, model, now_ts, life, context_limit)
        finally:
            store.close()

    return run


def make_scheduler(model: ChatModel, context_limit: int, **paths) -> Scheduler:
    scheduler = Scheduler()
    scheduler.add("记忆", 60, memory_job(model, context_limit, **paths))
    return scheduler
