"""记忆模块的后台任务：由全局调度（runtime.scheduler）每分钟调用一次 tick()（D27：时钟谁都要）。

- 熟悉期：只检查全局生命周期是否该启动（上下文用到约 70%，D24、D26）；启动时各模块的启动步骤由生命周期依次执行。
- 运行期：对话停顿后整理（设想文档：整理在对话结束后或空闲时运行，避免边读边改）。
  停顿多久算"停顿"，暂定与记忆时钟的减速点相同（25 分钟，D21）。
- 运行期：对话中途也分批整理（D31）：最早的话快要掉出新鲜记忆时，先把较早的一部分切段放进概率海，
  避免长时间不停顿时出现"不新鲜、又没整理"的空档。见 memory.fresh.split_point。
- 夜间：凌晨首次 tick 时合并近期重复的事件线（事件切分.md 第三层第 5 条）。
"""

import argparse
import random
import sys
import time
from dataclasses import dataclass, field

from contracts.llm import ChatModel
from memory.boundaries import conversation_closed
from memory.clock import DEFAULT_CLOCK, clock_of
from memory.fresh import split_point
from memory.lifecycle import context_used, register
from memory.merge import MergeReport, merge_recent_threads
from memory.store import MemoryStore
from memory.subconscious import ConsolidationReport, consolidate
from runtime import config
from runtime.lifecycle import RUNNING, Lifecycle


@dataclass(frozen=True)
class RunnerKnobs:
    pause: float = DEFAULT_CLOCK.idle_after  # 秒：对话停顿多久后整理（暂定，与记忆时钟减速点相同）
    night_hour: int = 3  # 几点以后算"夜间"（暂定）


DEFAULT_KNOBS = config.knobs(RunnerKnobs(), "runner")


@dataclass
class TickReport:
    started: dict[str, object] | None = None  # 全局启动时各模块的结果
    consolidated: dict[int, ConsolidationReport] = field(default_factory=dict)
    merged: MergeReport | None = None


def tick(
    store: MemoryStore,
    model: ChatModel,
    now_ts: float,
    lifecycle: Lifecycle,
    context_limit: int,
    k: RunnerKnobs = DEFAULT_KNOBS,
    rng: random.Random | None = None,
) -> TickReport:
    report = TickReport()
    if lifecycle.phase != RUNNING:
        report.started = lifecycle.maybe_start(now_ts, context_used(store), context_limit)
        return report
    clock = clock_of(store)
    for cid in store.conversation_ids():
        turns = store.unsegmented_turns(cid)
        paused = turns and now_ts - turns[-1].ts >= k.pause
        unsummarized = not turns and store.segments(cid) and not store.get_conversation(cid).summary
        if (
            store.pending_consolidations(cid)
            or paused
            or (unsummarized and conversation_closed(store, cid, now_ts))
        ):
            report.consolidated[cid] = consolidate(store, model, cid, now_ts, rng)
        elif (
            turns
            and config.enabled("mid_chat_split")
            and (until := split_point(store, cid, now_ts, clock)) is not None
        ):
            report.consolidated[cid] = consolidate(store, model, cid, now_ts, rng, until_ts=until)
    today = time.strftime("%Y-%m-%d", time.localtime(now_ts))
    if (
        config.enabled("nightly_merge")
        and time.localtime(now_ts).tm_hour >= k.night_hour
        and store.get_meta("last_merge_day") != today
    ):
        report.merged = merge_recent_threads(store, model, now_ts)
        store.set_meta("last_merge_day", today)
    return report


def main(argv: list[str] | None = None) -> None:
    """暂行的独立入口：只跑记忆模块的后台任务。以后各模块的任务都登记到同一个全局调度里一起跑。"""
    from llm import subconscious_model
    from llm.settings import load_settings
    from runtime.scheduler import Scheduler

    ap = argparse.ArgumentParser(description="记忆模块的后台任务：每分钟 tick 一次")
    ap.add_argument("--once", action="store_true", help="只 tick 一次")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    store, model, lifecycle = MemoryStore(), subconscious_model(), Lifecycle()
    register(lifecycle, store, model)
    limit = load_settings().thinking_context_limit

    def job(now_ts: float) -> None:
        r = tick(store, model, now_ts, lifecycle, limit)
        if r.started or r.consolidated or r.merged:
            print(
                time.strftime("%H:%M"), "启动" if r.started else "", f"整理 {list(r.consolidated)}", r.merged
            )

    scheduler = Scheduler()
    scheduler.add("记忆", 60, job)
    if args.once:
        scheduler.run_pending()
    else:
        scheduler.run_forever()


if __name__ == "__main__":
    main()
