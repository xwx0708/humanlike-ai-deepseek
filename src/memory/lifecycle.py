"""记忆系统的生命周期：熟悉期 → 启动 → 运行期（设想文档"启动时机：熟悉期"一节；决策 D24）。

- 熟悉期：不启动记忆系统，在上下文限制内正常聊天；但每条消息照常存档（ingest_turn），防止中途离开丢数据。
- 启动：上下文用到约 70%（用户 2026-10-28 定）时，一次性整理熟悉期的全部对话：切段、打分、挂事件线、
  建立概率海。整理途中不批量遗忘；整理完把所有片段的强度设为满格（设想文档：平滑过渡，用户不会感到 AI 突然失忆）。
- 运行期：遗忘、浮现、整理循环。

暂行：情绪基线还没实现（偏离记录 #2），启动时还不能"用全部数据算基线"。
"""

import random
from dataclasses import dataclass, field

from contracts.llm import ChatModel
from memory.resident import refresh_residents
from memory.store import MemoryStore
from memory.subconscious import ConsolidationReport, consolidate
from memory.tokens import estimate_tokens

FAMILIAR = "熟悉期"
RUNNING = "运行期"


@dataclass(frozen=True)
class LifecycleKnobs:
    start_ratio: float = 0.7  # 上下文用到多少比例时启动（用户定：约 70%）


DEFAULT_KNOBS = LifecycleKnobs()


@dataclass
class StartReport:
    context_used: int  # 启动时估算的上下文用量（token）
    reports: list[ConsolidationReport] = field(default_factory=list)


def phase(store: MemoryStore) -> str:
    return store.get_meta("phase", FAMILIAR)


def context_used(store: MemoryStore) -> int:
    """熟悉期里全部对话都在上下文中：估算它们一共占多少 token。"""
    return sum(estimate_tokens(t) + 4 for t in store.all_turn_texts())


def should_start(store: MemoryStore, context_limit: int, k: LifecycleKnobs = DEFAULT_KNOBS) -> bool:
    return phase(store) == FAMILIAR and context_used(store) >= k.start_ratio * context_limit


def start_up(
    store: MemoryStore, model: ChatModel, now_ts: float, rng: random.Random | None = None
) -> StartReport:
    """启动：整理熟悉期的全部对话，建立概率海，然后进入运行期。"""
    report = StartReport(context_used(store))
    for cid in store.conversation_ids():
        report.reports.append(consolidate(store, model, cid, now_ts, rng, sweep_forgotten=False))
    for seg in store.segments(state="active"):  # 初始强度满格：遗忘从启动这一刻开始算
        seg.last_recall_ts = now_ts
        store.save_segment_state(seg)
    refresh_residents(store, now_ts)
    store.set_meta("phase", RUNNING)
    store.set_meta("started_ts", str(now_ts))
    return report


def maybe_start(
    store: MemoryStore,
    model: ChatModel,
    now_ts: float,
    context_limit: int,
    k: LifecycleKnobs = DEFAULT_KNOBS,
    rng: random.Random | None = None,
) -> StartReport | None:
    """每条消息存档后可以调用一次：到了启动条件就启动，否则什么都不做。"""
    if not should_start(store, context_limit, k):
        return None
    return start_up(store, model, now_ts, rng)
