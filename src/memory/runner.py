"""后台调度：什么时候启动、什么时候整理、什么时候夜间合并。由后台时钟定期调用 tick()。

- 熟悉期：上下文用到约 70% 时启动（D24）。
- 运行期：对话停顿后整理（设想文档：整理在对话结束后或空闲时运行，避免边读边改）。
  停顿多久算"停顿"，暂定与记忆时钟的减速点相同（25 分钟，D21）。
- 夜间：凌晨首次 tick 时合并近期重复的事件线（事件切分.md 第三层第 5 条）。

暂行：后台时钟归哪个模块还没定（架构总图边界问题），先提供 tick() 和一个简单的循环入口
`python -m memory.runner`（每分钟 tick 一次）。
"""

import argparse
import random
import sys
import time
from dataclasses import dataclass, field

from contracts.llm import ChatModel
from memory.boundaries import conversation_closed
from memory.clock import DEFAULT_CLOCK
from memory.lifecycle import RUNNING, StartReport, maybe_start, phase
from memory.merge import MergeReport, merge_recent_threads
from memory.store import MemoryStore
from memory.subconscious import ConsolidationReport, consolidate


@dataclass(frozen=True)
class RunnerKnobs:
    pause: float = DEFAULT_CLOCK.idle_after  # 秒：对话停顿多久后整理（暂定，与记忆时钟减速点相同）
    night_hour: int = 3  # 几点以后算"夜间"（暂定）


DEFAULT_KNOBS = RunnerKnobs()


@dataclass
class TickReport:
    started: StartReport | None = None
    consolidated: dict[int, ConsolidationReport] = field(default_factory=dict)
    merged: MergeReport | None = None


def tick(
    store: MemoryStore,
    model: ChatModel,
    now_ts: float,
    context_limit: int,
    k: RunnerKnobs = DEFAULT_KNOBS,
    rng: random.Random | None = None,
) -> TickReport:
    report = TickReport()
    if phase(store) != RUNNING:
        report.started = maybe_start(store, model, now_ts, context_limit, rng=rng)
        return report
    for cid in store.conversation_ids():
        turns = store.unsegmented_turns(cid)
        paused = turns and now_ts - turns[-1].ts >= k.pause
        unsummarized = not turns and store.segments(cid) and not store.get_conversation(cid).summary
        if paused or (unsummarized and conversation_closed(store, cid, now_ts)):
            report.consolidated[cid] = consolidate(store, model, cid, now_ts, rng)
    today = time.strftime("%Y-%m-%d", time.localtime(now_ts))
    if time.localtime(now_ts).tm_hour >= k.night_hour and store.get_meta("last_merge_day") != today:
        report.merged = merge_recent_threads(store, model, now_ts)
        store.set_meta("last_merge_day", today)
    return report


def main(argv: list[str] | None = None) -> None:
    from llm import subconscious_model
    from llm.settings import load_settings

    ap = argparse.ArgumentParser(description="记忆模块的后台调度：每分钟 tick 一次")
    ap.add_argument("--once", action="store_true", help="只 tick 一次")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    store, model = MemoryStore(), subconscious_model()
    limit = load_settings().thinking_context_limit
    while True:
        r = tick(store, model, time.time(), limit)
        if r.started or r.consolidated or r.merged:
            print(
                time.strftime("%H:%M"), "启动" if r.started else "", f"整理 {list(r.consolidated)}", r.merged
            )
        if args.once:
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
