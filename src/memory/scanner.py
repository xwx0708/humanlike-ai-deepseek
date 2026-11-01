"""扫描器与浮现区：记忆自己冒头（设想文档"记忆自己冒头：分区扫描与浮现区"一节；待办 #9）。

设想文档定下的部分：
- 不是去"挑选"信息，而是让概率自己冒头；潜意识慢慢地算，算完这个分区再算下一个分区；数值运算用 Python。
- 后台循环像时钟指针一样轮流扫描分区，对每条记忆：衰减强度、加上当下话题的"启动"加成、按概率掷骰子，
  命中就放进浮现区；p = sigmoid((强度 + 启动加成 - 阈值) / 温度)。
- 浮现区容量很小（3～5 条），旧的会被挤掉；思考模型每轮看一眼，自己决定要不要提起。
- 温度：低温专注，高温放空；由话题而定：正专注于一个话题就是专注，在闲聊就是发散（用户定，D26）。保留兜底：思考模型仍可主动检索（search_memory）。
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
from dataclasses import dataclass, field, replace

from contracts.knobs import knob
from contracts.memory import Segment
from contracts.plugins import Candidate
from memory.candidates import bm25, tokenize
from memory.clock import MemoryClock, clock_of
from memory.forgetting import Trace, recall
from memory.resident import effective_strength, is_resident_segment
from memory.store import MemoryStore
from runtime import config, plugins

IN_MEMORY = ("active", "abstract")


@dataclass(frozen=True)
class ScannerKnobs:
    buffer_size: int = knob(
        4, "浮现区容量", "浮现区可同时容纳的片段数，FIFO 置换。浮现本身不触发巩固。", lo=1, hi=12
    )  # 浮现区容量（设想文档 3～5，取中间，暂定）
    threshold: float = knob(
        0.6, "浮现阈值", "sigmoid 的中心：S + 启动加成等于它时，p 为速率上限的一半。", lo=0, hi=1.5, step=0.05
    )  # sigmoid 的阈值（暂定）
    # 温度由话题而定（用户定，D26）：正专注于一个话题就是专注（低温），在闲聊就是发散（高温）
    temp_focus: float = knob(
        0.1, "专注温度 T", "专注于同一话题时的温度：低温使采样更确定，强记忆更集中地浮现。", lo=0.01, hi=1
    )  # 专注时的温度（暂定）
    temp_diffuse: float = knob(
        0.3, "发散温度 T", "闲聊时的温度：高温使采样更随机，弱记忆也有机会浮现。", lo=0.01, hi=1
    )  # 闲聊时的温度（暂定）
    focus_turns: int = knob(
        6, "专注判定", "同一话题连续多少条用户消息算完全专注（温度降到专注温度）。", lo=1, hi=30
    )  # 同一个话题连续聊到几条用户消息，算完全专注（暂定）
    rate: float = knob(
        0.02,
        "速率上限",
        "每分钟浮现强度的上限。0.02 时最强、最相关的记忆约每 50 分钟浮现一次。",
        lo=0.001,
        hi=0.5,
        step=0.001,
    )  # 每分钟的浮现强度上限：最强、最相关的记忆约每 50 分钟冒头一次（暂定）
    priming_max: float = knob(
        0.5,
        "启动加成上限",
        "启动效应（priming）：与当前话题关键词相关的片段获得的最大加成。",
        lo=0,
        hi=1.5,
        step=0.05,
    )  # 启动加成的上限（暂定）
    person_bonus: float = knob(
        0.3, "人物加成", "片段涉及的人物与当前话题重合时的加成。", lo=0, hi=1.5, step=0.05
    )  # 人物重合给的启动加成（暂定）
    negative_damp: float = knob(
        0.6, "负面抑制", "负面片段在话题无关时，浮现概率最多下降的比例。", lo=0, hi=1, step=0.05
    )  # 负面且话题无关时，浮现概率最多打几折（暂定）


DEFAULT_KNOBS = config.knobs(
    ScannerKnobs(),
    "scanner",
    "扫描与浮现",
    "扫描器按月份分区轮询，每分钟一个分区。对每个片段做伯努利采样：P = 1 − e^(−p·Δt)，p = 速率上限 · σ((S + 启动加成 − 阈值) / T)。命中者进入浮现区。",
)


# ---------- 浮现策略的可替换点（改进三，D42）：contracts.plugins.SurfacePolicy ----------


class ProbabilisticSurfacing:
    """默认：设想文档的伯努利采样。p = rate · σ((S + 启动加成 − 阈值) / T)，P = 1 − e^(−p·Δt)；
    负面记忆在话题无关时少冒头。"""

    name = "概率浮现"

    def probability(self, c: Candidate, temperature: float, k: "ScannerKnobs") -> float:
        """每分钟的浮现强度（0～rate）。"""
        x = (c.strength + c.priming - k.threshold) / temperature
        p = 1 / (1 + math.exp(-max(-50.0, min(50.0, x))))
        valence = c.segment.emotion.valence if c.segment.emotion else 0.0
        if valence < 0:  # 负面记忆：话题不相关时少冒头，相关时照常
            related = min(1.0, c.priming / (0.5 * k.priming_max))  # 关键词最相关或人物重合即算完全相关
            p *= 1 - k.negative_damp * (-valence) * (1 - related)
        return k.rate * p

    def surface(
        self, candidates: list[Candidate], temperature: float, k: "ScannerKnobs", rng: random.Random
    ) -> list[Segment]:
        out = []
        for c in candidates:
            hit = 1 - math.exp(-self.probability(c, temperature, k) * max(0.0, c.minutes))
            if rng.random() < hit:
                out.append(c.segment)
        return out


class RelevanceSurfacing:
    """纯相关度检索：不抽样，只把和当下话题最相关的几条（启动加成 × 强度最高）拿出来；话题无关的不冒头。"""

    name = "相关度检索"

    def surface(
        self, candidates: list[Candidate], temperature: float, k: "ScannerKnobs", rng: random.Random
    ) -> list[Segment]:
        related = [c for c in candidates if c.priming > 0]
        related.sort(key=lambda c: -(c.priming * c.strength))
        return [c.segment for c in related[: k.buffer_size]]


class NoSurfacing:
    """关闭浮现：记忆只在新鲜记忆、常驻栏和用户要求翻档案时出现。"""

    name = "关闭浮现"

    def surface(
        self, candidates: list[Candidate], temperature: float, k: "ScannerKnobs", rng: random.Random
    ) -> list[Segment]:
        return []


SURFACE_POLICIES = {p.name: p for p in (ProbabilisticSurfacing(), RelevanceSurfacing(), NoSurfacing())}
DEFAULT_POLICY = plugins.choose(
    "surface",
    config.value(
        "scanner",
        "surface_policy",
        ProbabilisticSurfacing.name,
        "浮现策略",
        "哪些记忆冒头进浮现区。概率浮现是默认（伯努利采样）；相关度检索只取和话题最相关的几条、不抽样；"
        "关闭浮现则记忆只在新鲜记忆、常驻栏和翻档案时出现。也可以把自己写的策略放进数据文件夹的 plugins/。",
        kind="select",
        options=plugins.names("surface", SURFACE_POLICIES),
    ),
    SURFACE_POLICIES,
)


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
    focus: float = 0.5  # 0 = 闲聊发散，1 = 专注于一个话题；由 set_focus / focus_of 给出
    policy: object = DEFAULT_POLICY  # contracts.plugins.SurfacePolicy：按设置选用

    @property
    def temperature(self) -> float:
        return self.k.temp_diffuse + (self.k.temp_focus - self.k.temp_diffuse) * self.focus

    def set_focus(self, focus: float) -> None:
        self.focus = min(1.0, max(0.0, focus))

    def refresh_buffer(self, now_ts: float) -> None:
        """缓存只保留仍可见的片段，并重新读取清晰度与文本，防止删除或抽象化后泄露旧内容。"""
        clock = clock_of(self.store)
        visible = {s.id: s for state in IN_MEMORY for s in self.store.segments(state=state)}
        refreshed = []
        for item in self.buffer:
            seg = visible.get(item.segment_id)
            if seg is None:
                continue
            clarity = effective_strength(self.store, seg, now_ts, clock)
            if clarity >= self.store.load_global_knobs().forget_line:
                refreshed.append(replace(item, text=_text(seg), clarity=min(item.clarity, clarity)))
        self.buffer = refreshed
        self.priming = {sid: value for sid, value in self.priming.items() if sid in visible}
        self._last_scan = {sid: value for sid, value in self._last_scan.items() if sid in visible}

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
        """每分钟的浮现强度（0～rate），按默认的概率浮现算；供测试和可视化看。"""
        c = Candidate(seg, strength, self.priming.get(seg.id, 0.0), 0.0)
        return ProbabilisticSurfacing().probability(c, self.temperature, self.k)

    def tick(self, now_ts: float, clock: MemoryClock | None = None) -> list[Surfaced]:
        """扫描下一个分区（时钟指针走一格），返回这一格新冒头的记忆。"""
        self.refresh_buffer(now_ts)
        parts = self.partitions()
        if not parts:
            return []
        part = parts[self._cursor % len(parts)]
        self._cursor += 1
        clock = clock or clock_of(self.store)
        line = self.store.load_global_knobs().forget_line
        candidates, strengths = [], {}
        for s in part:
            minutes = (now_ts - self._last_scan.get(s.id, now_ts - 60.0)) / 60.0
            self._last_scan[s.id] = now_ts
            strength = effective_strength(self.store, s, now_ts, clock)
            if strength < line:
                continue  # 低于遗忘线的交给批量遗忘（D22），不再冒头
            strengths[s.id] = strength
            candidates.append(Candidate(s, strength, self.priming.get(s.id, 0.0), minutes))
        new = []
        for s in self.policy.surface(candidates, self.temperature, self.k, self.rng):
            if all(b.segment_id != s.id for b in self.buffer) and all(n.segment_id != s.id for n in new):
                v = s.emotion.valence if s.emotion else 0.0
                new.append(Surfaced(s.id, _text(s), strengths[s.id], v, now_ts))
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


def focus_of(store: MemoryStore, conversation_id: int, k: ScannerKnobs = DEFAULT_KNOBS) -> float:
    """这场对话现在有多专注：从最近一个候选边界（换话题的信号）起，用户连续聊了几条，满 focus_turns 条算 1。
    暂定：用切段第一层的候选边界判断"还在不在同一个话题"，不调用模型；话题频繁跳来跳去就是闲聊、发散。"""
    turns = [t for t in store.turns(conversation_id) if t.role == "user"]
    marks = store.boundaries(conversation_id)
    n = 0
    for t in reversed(turns):
        n += 1
        if t.id in marks:
            break
    return min(1.0, n / k.focus_turns)
