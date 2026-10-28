"""扫描器与浮现区：记忆自己冒头（设想文档"记忆自己冒头：分区扫描与浮现区"一节；待办 #9）。

设想文档定下的部分：
- 不是去"挑选"信息，而是让概率自己冒头；潜意识慢慢地算，算完这个分区再算下一个分区；数值运算用 Python。
- 后台循环像时钟指针一样轮流扫描分区，对每条记忆：衰减强度、加上当下话题的"启动"加成、按概率掷骰子，
  命中就放进浮现区；p = sigmoid((强度 + 启动加成 - 阈值) / 温度)。
- 浮现区容量很小（3～5 条），旧的会被挤掉；思考模型每轮看一眼，自己决定要不要提起。
- 温度：低温专注，高温放空。保留兜底：思考模型仍可主动检索（search_memory）。
- 检索要留下痕迹：被真正用到一次就回报一次（use_memory），用于加固（D20）。

暂定（待用户确认，见 00-总纲/待你确认.md）：
- 启动加成不用嵌入模型（D10）：沿用挂事件线的候选规则，人物重合 + jieba/BM25 关键词重合。
- 浮现概率按单位时间算，与扫描快慢脱钩（待议问题第 2 条）：p 是"每分钟的浮现强度"，一次扫描的命中概率
  = 1 - exp(-rate × p × 距上次扫描这条记忆的分钟数)。
- 效价：负面记忆在话题不相关时浮现概率打折（待议问题第 3 条）。
- 分区按片段所在的月份；常驻栏里的不参与（它们一直在场）；移进档案的不参与。
"""

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field

from contracts.memory import Segment
from memory.candidates import bm25, tokenize
from memory.clock import MemoryClock, clock_of
from memory.forgetting import Trace, recall
from memory.resident import effective_strength, is_resident_segment
from memory.store import MemoryStore

IN_MEMORY = ("active", "abstract")


@dataclass(frozen=True)
class ScannerKnobs:
    buffer_size: int = 4  # 浮现区容量（设想文档 3～5，取中间，暂定）
    threshold: float = 0.6  # sigmoid 的阈值（暂定）
    temperature: float = 0.15  # 低温专注、高温放空（暂定默认值；由谁、按什么调，待定）
    rate: float = 0.02  # 每分钟的浮现强度上限：最强、最相关的记忆约每 50 分钟冒头一次（暂定）
    priming_max: float = 0.5  # 启动加成的上限（暂定）
    person_bonus: float = 0.3  # 人物重合给的启动加成（暂定）
    negative_damp: float = 0.6  # 负面且话题无关时，浮现概率最多打几折（暂定）


DEFAULT_KNOBS = ScannerKnobs()


@dataclass
class Surfaced:
    segment_id: int
    text: str  # 标题；只剩关键词的给关键词
    clarity: float  # 清晰度 = 当前强度；让回应模块能说"有点印象但记不清"
    valence: float
    ts: float


@dataclass
class Scanner:
    store: MemoryStore
    k: ScannerKnobs = DEFAULT_KNOBS
    rng: random.Random = field(default_factory=random.Random)
    buffer: list[Surfaced] = field(default_factory=list)
    priming: dict[int, float] = field(default_factory=dict)  # 片段 → 启动加成；话题变化时算一次
    _cursor: int = 0
    _last_scan: dict[int, float] = field(default_factory=dict)

    def set_topic(self, text: str, person_ids: list[int] = ()) -> None:
        """当下话题变化时调用一次：给相关记忆"充电"。"""
        segs = [s for state in IN_MEMORY for s in self.store.segments(state=state)]
        docs = {s.id: tokenize(_text(s)) for s in segs}
        scores = dict(bm25(tokenize(text), docs))
        top = max(scores.values(), default=0.0) or 1.0
        people = set(person_ids)
        self.priming = {}
        for s in segs:
            p = 0.5 * self.k.priming_max * scores.get(s.id, 0.0) / top
            if people & set(s.person_ids):
                p += self.k.person_bonus
            if p > 0:
                self.priming[s.id] = min(self.k.priming_max, p)

    def partitions(self) -> list[list[Segment]]:
        """按片段开始的月份分区（暂定），常驻栏里的不参与。"""
        parts: dict[str, list[Segment]] = defaultdict(list)
        for state in IN_MEMORY:
            for s in self.store.segments(state=state):
                if not is_resident_segment(self.store, s):
                    parts[_month(s.start_ts)].append(s)
        return [parts[m] for m in sorted(parts)]

    def probability(self, seg: Segment, strength: float) -> float:
        """每分钟的浮现强度（0～rate）。"""
        prim = self.priming.get(seg.id, 0.0)
        x = (strength + prim - self.k.threshold) / self.k.temperature
        p = 1 / (1 + math.exp(-max(-50.0, min(50.0, x))))
        valence = seg.emotion.valence if seg.emotion else 0.0
        if valence < 0:  # 负面记忆：话题不相关时少冒头，相关时照常
            related = min(1.0, prim / (0.5 * self.k.priming_max))  # 关键词最相关或人物重合即算完全相关
            p *= 1 - self.k.negative_damp * (-valence) * (1 - related)
        return self.k.rate * p

    def tick(self, now_ts: float, clock: MemoryClock | None = None) -> list[Surfaced]:
        """扫描下一个分区（时钟指针走一格），返回这一格新冒头的记忆。"""
        parts = self.partitions()
        if not parts:
            return []
        part = parts[self._cursor % len(parts)]
        self._cursor += 1
        clock = clock or clock_of(self.store)
        new = []
        for s in part:
            minutes = (now_ts - self._last_scan.get(s.id, now_ts - 60.0)) / 60.0
            self._last_scan[s.id] = now_ts
            strength = effective_strength(self.store, s, now_ts, clock)
            if strength < self.store.load_global_knobs().forget_line:
                continue  # 低于遗忘线的交给批量遗忘（D22），不再冒头
            hit = 1 - math.exp(-self.probability(s, strength) * max(0.0, minutes))
            if self.rng.random() < hit and all(b.segment_id != s.id for b in self.buffer):
                v = s.emotion.valence if s.emotion else 0.0
                new.append(Surfaced(s.id, _text(s), strength, v, now_ts))
        self.buffer = (self.buffer + new)[-self.k.buffer_size :]  # 旧的被挤掉
        return new


def _text(s: Segment) -> str:
    return s.trace if s.state == "abstract" else s.title


def _month(ts: float) -> str:
    import time

    return time.strftime("%Y-%m", time.localtime(ts))


def search_memory(store: MemoryStore, query: str, now_ts: float, limit: int = 5) -> list[Surfaced]:
    """兜底：思考模型主动检索记忆系统（不含档案）。BM25 相关度 × 当前强度。"""
    clock = clock_of(store)
    segs = {s.id: s for state in IN_MEMORY for s in store.segments(state=state)}
    hits = []
    for sid, score in bm25(tokenize(query), {i: tokenize(_text(s)) for i, s in segs.items()}):
        s = segs[sid]
        strength = effective_strength(store, s, now_ts, clock)
        if strength >= store.load_global_knobs().forget_line:
            v = s.emotion.valence if s.emotion else 0.0
            hits.append((score * strength, Surfaced(sid, _text(s), strength, v, now_ts)))
    return [h for _, h in sorted(hits, key=lambda x: -x[0])[:limit]]


def use_memory(store: MemoryStore, segment_id: int, ts: float) -> None:
    """记忆使用回执（回应 → 记忆）：思考模型真正提起或用到了这条记忆。强度回满、半衰期按"忘了多少"加长（D20）；
    回忆次数加一，常驻得分里的"回忆频率"由此而来（D17）。仅仅浮现不算。"""
    seg = store.get_segment(segment_id)
    elapsed = clock_of(store).elapsed_days(seg.last_recall_ts, ts)
    after = recall(Trace(seg.half_life, 0.0, seg.recall_count), elapsed, store.load_global_knobs())
    seg.half_life, seg.recall_count, seg.last_recall_ts = after.half_life, after.recall_count, ts
    store.save_segment_state(seg)
