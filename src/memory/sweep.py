"""遗忘之后：批量清理（决策 D22，用户 2026-10-28 定）。

强度掉到遗忘线以下的片段先攒着；数量达到 batch_size 时，把这一批（固定 batch_size 条，按到达遗忘线的先后）一起处理：
- 95% 移进档案（state = "archived"），不再出现在记忆系统里。原文一直在档案里，以后"翻档案"还能找回。
- 剩下 5% 抽象化：只提取简要关键词保留（state = "abstract"，关键词存在 trace），并重新开始一条新的遗忘曲线。
不分类别，整批一起算（用户 2026-10-28 确认）。攒得多时分成几批，每批都是 batch_size 条，余下的继续攒。
常驻栏里的线不遗忘（D17），不参与。留下的 5% 随机挑（用户定）。一批多少条是暂定的旋钮。
"""

import math
import random
from dataclasses import dataclass

from contracts.llm import ChatModel
from contracts.memory import SECONDS_PER_DAY, Segment
from memory.clock import MemoryClock, clock_of
from memory.forgetting import initial_half_life, relative_arousal
from memory.prompts import KEYWORDS_SCHEMA, build_keywords_prompt, parse_keywords
from memory.resident import effective_strength
from memory.store import MemoryStore

IN_MEMORY = ("active", "abstract")  # 还在记忆系统里的状态


@dataclass(frozen=True)
class SweepKnobs:
    batch_size: int = 20  # 一批固定多少条（用户说"一定数量"，数值暂定）
    keep_ratio: float = 0.05  # 留下抽象化的比例（用户定）
    excerpt_chars: int = 300  # 给模型看的原文节选长度


DEFAULT_KNOBS = SweepKnobs()


@dataclass
class SweepReport:
    archived: list[int]
    kept: dict[int, list[str]]  # 片段 → 关键词


def forgotten(store: MemoryStore, now_ts: float, clock: MemoryClock | None = None) -> list[Segment]:
    """还在记忆系统里、但强度已经低于遗忘线的片段。"""
    clock = clock or clock_of(store)
    line = store.load_global_knobs().forget_line
    return [
        s
        for state in IN_MEMORY
        for s in store.segments(state=state)
        if effective_strength(store, s, now_ts, clock) < line
    ]


def _crossed_at(seg: Segment, clock: MemoryClock, forget_line: float) -> float:
    """到达遗忘线的记忆时间（天）：最近一次被提及时的记忆时间 + 半衰期 × log2(1 / 遗忘线)。"""
    return clock.seconds_at(seg.last_recall_ts) / SECONDS_PER_DAY + seg.half_life * math.log2(1 / forget_line)


def sweep(
    store: MemoryStore,
    model: ChatModel,
    now_ts: float,
    clock: MemoryClock | None = None,
    k: SweepKnobs = DEFAULT_KNOBS,
    rng: random.Random | None = None,
) -> SweepReport | None:
    """攒够了就按批处理，返回处理结果；一批都不够返回 None。rng：随机挑留下的 5%，测试时传固定种子。"""
    clock, rng = clock or clock_of(store), rng or random.Random()
    line = store.load_global_knobs().forget_line
    due = sorted(forgotten(store, now_ts, clock), key=lambda x: _crossed_at(x, clock, line))
    if len(due) < k.batch_size:
        return None
    report = SweepReport([], {})
    for i in range(len(due) // k.batch_size):
        _sweep_batch(store, model, now_ts, due[i * k.batch_size : (i + 1) * k.batch_size], k, rng, report)
    return report


def _sweep_batch(
    store: MemoryStore,
    model: ChatModel,
    now_ts: float,
    batch: list[Segment],
    k: SweepKnobs,
    rng: random.Random,
    report: SweepReport,
) -> None:
    """一批：95% 移进档案，5% 只留关键词、重新开始遗忘曲线。"""
    n_keep = max(1, round(len(batch) * k.keep_ratio))
    keep = rng.sample(batch, n_keep)  # 随机留 5%（用户定）
    drop = [s for s in batch if s not in keep]

    r = model.chat(
        build_keywords_prompt(
            [(s.id, s.title, store.turn_text(s.message_ids)[: k.excerpt_chars]) for s in keep]
        ),
        schema=KEYWORDS_SCHEMA,
        temperature=0,
    )
    words = parse_keywords(r.data, [s.id for s in keep])

    knobs = store.load_global_knobs()
    kept = {}
    for s in keep:
        s.trace = "、".join(words.get(s.id) or [s.title[:12]])  # 模型漏掉的，兜底用标题开头
        e = s.emotion
        # 新的遗忘曲线：从现在开始，半衰期按情绪和重要性重新算（不再带首尾加成和以前的调整）
        s.half_life = (
            initial_half_life(relative_arousal(e.arousal, 0.0), e.importance, knobs)
            if e
            else knobs.base_half_life
        )
        s.adjustment, s.at_edge, s.last_recall_ts, s.state = 1.0, False, now_ts, "abstract"
        store.save_segment_state(s)
        kept[s.id] = s.trace.split("、")
    for s in drop:
        s.state = "archived"
        store.save_segment_state(s)
    archived = [s.id for s in drop]
    store.log_sweep(now_ts, archived, kept)
    report.archived += archived
    report.kept.update(kept)
