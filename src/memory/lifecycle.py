"""记忆模块在全局生命周期里的部分（生命周期本身是全局的，见 runtime.lifecycle；D24、D27）。

- 熟悉期：记忆系统不启动，消息照常存档（ingest_turn），防止中途离开丢数据。
- 启动：记忆模块要做的是一次性整理熟悉期的全部对话：切段、打分、挂事件线、建立概率海。
  整理途中不批量遗忘；整理完把所有片段的强度设为满格（设想文档：平滑过渡，用户不会感到 AI 突然失忆）。
- 何时启动由全局生命周期判断（上下文用到约 70%）；上下文用量用回应模块回报的实际用量，没有时按字数粗估（context_used）。
"""

import random
from dataclasses import dataclass, field

from contracts.emotion import EmotionBaseline, EmotionScorer
from contracts.llm import ChatModel
from memory.resident import refresh_residents
from memory.store import MemoryStore
from memory.subconscious import ConsolidationReport, consolidate
from memory.tokens import estimate_tokens
from runtime.lifecycle import Lifecycle


@dataclass
class StartReport:
    context_used: int  # 启动时估算的上下文用量（token）
    reports: list[ConsolidationReport] = field(default_factory=list)


USED_KEY = "context_used"


def report_context_used(store: MemoryStore, prompt_tokens: int) -> None:
    """回应模块每轮回报思考模型实际用掉的输入 token（D24：接入后改用实际用量）。"""
    store.set_meta(USED_KEY, str(int(prompt_tokens)))


def context_used(store: MemoryStore) -> int:
    """熟悉期里全部对话都在上下文中：一共占多少 token。有回应模块回报的实际用量就用它（取较大者，
    因为回报的是上一轮的，之后又来了新消息）；没有时按字数估算。"""
    estimate = sum(estimate_tokens(t) + 4 for t in store.all_turn_texts())
    reported = int(store.get_meta(USED_KEY, "0"))
    return max(reported, estimate) if reported else estimate


def start_up(
    store: MemoryStore,
    model: ChatModel,
    now_ts: float,
    rng: random.Random | None = None,
    scorer: EmotionScorer | None = None,
    baseline: EmotionBaseline | None = None,
) -> StartReport:
    """记忆模块的启动步骤：整理熟悉期的全部对话，建立概率海。"""
    report = StartReport(context_used(store))
    for cid in store.conversation_ids():
        report.reports.append(
            consolidate(
                store, model, cid, now_ts, rng, sweep_forgotten=False, scorer=scorer, baseline=baseline
            )
        )
    for seg in store.segments(state="active"):  # 初始强度满格：遗忘从启动这一刻开始算
        seg.last_recall_ts = now_ts
        store.save_segment_state(seg)
    refresh_residents(store, now_ts)
    return report


def register(
    lifecycle: Lifecycle, store: MemoryStore, model: ChatModel, rng: random.Random | None = None
) -> None:
    """把记忆模块的启动步骤登记到全局生命周期。"""
    lifecycle.on_start("记忆", lambda now_ts: start_up(store, model, now_ts, rng))
