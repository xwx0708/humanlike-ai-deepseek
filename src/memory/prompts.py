"""潜意识模型（本地 Qwen）的三个提示词：切段第二层、判断事件线归属、修改函数参数。

每个任务由三部分组成：build_*（拼提示词）、*_SCHEMA（约束输出的 JSON 格式）、parse_*（校验结果）。
模型的输出一律不直接信任：越界的数值收回范围内，不存在的 id 丢掉，漏掉的消息补回去。
设计依据见 01-记忆/事件切分.md、01-记忆/潜意识.md。
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from contracts.llm import Message, system, user
from contracts.memory import ChatTurn, EmotionScore
from memory.forgetting import ForgettingKnobs
from memory.tuning import TUNABLE


def _clamp(x: Any, lo: float, hi: float, default: float) -> float:
    try:
        return min(hi, max(lo, float(x)))
    except (TypeError, ValueError):
        return default


def _hm(ts: float) -> str:
    return time.strftime("%m-%d %H:%M", time.localtime(ts))


# ====================================================================================
# 一、切段第二层：确认片段
# ====================================================================================

SEGMENT_SYSTEM = """你是一个聊天 AI 的"潜意识"，在后台整理它和用户的聊天记录。现在要把一段对话切成"片段"。

## 什么是片段
- 片段是对话中围绕同一件事的一个子话题，必须能用一句话概括。概括不了就说明切得太细，和相邻片段合并。
- 话题明显转了（例如从妈妈生病转到周末爬山），就是新片段；只是顺嘴插一句又马上绕回来，不算新片段。
- 话题慢慢漂远（例如从工作累聊到想换工作再聊到想去哪个城市），漂到和开头不是一件事时，切开。
- 允许话题交错：一条消息如果同时在说两件事，可以同时放进两个片段。
- 每条消息至少属于一个片段。
- 标 [候选边界] 的消息是规则提示可能换话题的位置，只供参考，你自己判断。

## 每个片段要给出
- messages：包含的消息编号。
- title：一句话标题，用第三人称概括发生了什么，例如"他妈妈住院了，他很担心"。
- people：片段里提到的具体的人。name 用最常用的叫法，aliases 列出这段里出现的其他叫法（"我妈""老妈""母亲"是同一个人）。不要列"我""你"、AI 自己和泛指的人。
- 情绪评分：只评用户的情绪，不评 AI。看情绪最强烈的那一刻和片段结尾（峰终定律），不要取平均。
  - arousal 唤醒度 0～1：多强烈。闲聊 0.1；有点烦或有点开心 0.3～0.5；明显着急、难过、兴奋 0.6～0.8；崩溃、狂喜、惊吓 0.9 以上。
  - valence 效价 -1～1：负面为负，正面为正，中性为 0。
  - importance 重要性 0～1：和情绪无关的"以后用得上、不能忘"。不要因为情绪强就给高分，情绪已经由 arousal 体现。锚点：
    - 0～0.2：闲聊、日常琐事（"今天好累""午饭吃了面"）。
    - 0.2～0.4：随口提到的近况（"我妈最近老咳嗽""最近在追一部剧"）。
    - 0.5～0.7：实质性的事件或计划（"我妈住院了""在考虑辞职"）。
    - 0.8～0.9：长期稳定的事实（过敏、家人的名字和关系、生日、职业）。
    - 1：用户明确说"记住这个"。
  - emotion_reason：一句话说明打分理由。
- 反讽和含蓄表达容易判错：说"太好了又加班"是负面；语气突然变得简短冷淡，可能比字面更严重。

只输出 JSON。"""

SEGMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "messages": {"type": "array", "items": {"type": "integer"}},
                    "title": {"type": "string"},
                    "people": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "aliases": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["name", "aliases"],
                        },
                    },
                    "arousal": {"type": "number"},
                    "valence": {"type": "number"},
                    "importance": {"type": "number"},
                    "emotion_reason": {"type": "string"},
                },
                "required": [
                    "messages",
                    "title",
                    "people",
                    "arousal",
                    "valence",
                    "importance",
                    "emotion_reason",
                ],
            },
        }
    },
    "required": ["segments"],
}


@dataclass
class SegmentDraft:
    """Qwen 切出、经校验的片段，还没有存进数据库。"""

    turn_ids: list[int]
    title: str
    people: list[tuple[str, list[str]]]  # (常用叫法, 其他叫法)
    emotion: EmotionScore


def build_segment_prompt(
    turns: list[ChatTurn], boundaries: dict[int, list[str]], previous_title: str | None = None
) -> list[Message]:
    lines = []
    for i, t in enumerate(turns, 1):
        who = "用户" if t.role == "user" else "AI"
        mark = f" [候选边界：{'、'.join(boundaries[t.id])}]" if t.id in boundaries else ""
        lines.append(f"{i}. [{_hm(t.ts)}] {who}：{t.content}{mark}")
    head = f"上一批最后一个片段的标题：{previous_title}\n\n" if previous_title else ""
    return [system(SEGMENT_SYSTEM), user(head + "对话如下：\n" + "\n".join(lines))]


def parse_segments(data: dict[str, Any], turns: list[ChatTurn]) -> list[SegmentDraft]:
    n = len(turns)
    drafts = []
    for raw in data.get("segments", []):
        idx = sorted({i for i in raw.get("messages", []) if isinstance(i, int) and 1 <= i <= n})
        title = str(raw.get("title", "")).strip()
        if not idx or not title:
            continue
        people = []
        for p in raw.get("people", []):
            name = str(p.get("name", "")).strip()
            if name:
                aliases = [
                    a.strip() for a in p.get("aliases", []) if isinstance(a, str) and a.strip() and a != name
                ]
                people.append((name, aliases))
        emotion = EmotionScore(
            arousal=_clamp(raw.get("arousal"), 0, 1, 0.1),
            valence=_clamp(raw.get("valence"), -1, 1, 0.0),
            importance=_clamp(raw.get("importance"), 0, 1, 0.1),
            reason=str(raw.get("emotion_reason", "")).strip(),
        )
        drafts.append(SegmentDraft([turns[i - 1].id for i in idx], title, people, emotion))
    if not drafts:
        raise ValueError("Qwen 没有切出任何有效片段")

    # 漏掉的消息并到前面最近的片段（没有前面的就并到第一个）
    covered = {t for d in drafts for t in d.turn_ids}
    order = {t.id: i for i, t in enumerate(turns)}
    for t in turns:
        if t.id not in covered:
            before = [d for d in drafts if min(order[x] for x in d.turn_ids) <= order[t.id]]
            target = max(before, key=lambda d: min(order[x] for x in d.turn_ids)) if before else drafts[0]
            target.turn_ids = sorted({*target.turn_ids, t.id}, key=order.__getitem__)
    drafts.sort(key=lambda d: min(order[x] for x in d.turn_ids))
    return drafts


# ====================================================================================
# 二、判断事件线归属
# ====================================================================================

THREAD_SYSTEM = """你是一个聊天 AI 的"潜意识"。用户的经历按"事件线"组织：一条事件线是跨越多天的同一件事，
例如"妈妈的健康：咳嗽 → 住院 → 出院"。现在有一个新片段，判断它属于哪条已有的事件线。

## 规则
- 是同一件事的延续（同一个人的同一件事、同一个计划的后续），才算属于。只是沾边（都和工作有关、都提到了妈妈但说的是两回事）不算。
- 可以同时属于多条（例如"因为妈妈住院，他想辞职回老家"同时属于妈妈的健康和工作两条线）。
- 片段的主体是一件新的事时，即使它是由已有的事引起的（例如因为妈妈住院而想辞职），也要新建，
  并把引起它的那条事件线放进 related。
- 都不属于，就选"新建"，并给新事件线起一个短标题，概括这件事而不是这个片段，例如"妈妈的健康""工作去留""周末爬山"。
- 没有候选事件线时，必须新建，只需要给出标题。
- confidence 是你的把握，0～1。拿不准时给低分，系统会先新建，晚上再检查要不要合并。错误地合并比错误地新建更难发现。
- related 列出"有关但不是同一件事"的事件线编号，用来建立关联，可以为空。

只输出 JSON。"""

THREAD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "thread_ids": {"type": "array", "items": {"type": "integer"}},
        "new_thread": {"type": "boolean"},
        "new_thread_title": {"type": "string"},
        "related": {"type": "array", "items": {"type": "integer"}},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["thread_ids", "new_thread", "new_thread_title", "related", "confidence", "reason"],
}

THREAD_CONFIDENCE_MIN = 0.6  # 把握低于此值就新建（暂定，待用真实数据校准）


@dataclass
class ThreadDecision:
    attach_to: list[int]  # 挂到这些事件线；为空表示新建
    new_title: str
    related: list[int]
    confidence: float
    reason: str
    low_confidence: bool = False  # 模型想挂到已有线，但把握不够，被改成了新建

    @property
    def create(self) -> bool:
        return not self.attach_to


def build_thread_prompt(segment: dict[str, Any], candidates: list[dict[str, Any]]) -> list[Message]:
    """segment / candidates 都是卡片：标题、一句话摘要（片段可无）、最近片段标题、涉及人物。"""
    cards = "\n".join(
        f"[{c['id']}] {c['title']}｜摘要：{c.get('summary', '')}｜最近：{c.get('latest_segment', '')}"
        f"｜人物：{'、'.join(c.get('people', [])) or '无'}"
        for c in candidates
    )
    seg = f"标题：{segment['title']}｜人物：{'、'.join(segment.get('people', [])) or '无'}"
    if segment.get("context"):
        seg += f"｜前后文：{segment['context']}"
    return [system(THREAD_SYSTEM), user(f"新片段：\n{seg}\n\n候选事件线：\n{cards or '（没有候选）'}")]


def parse_thread(
    data: dict[str, Any], candidate_ids: list[int], threshold: float = THREAD_CONFIDENCE_MIN
) -> ThreadDecision:
    valid = set(candidate_ids)
    ids = [i for i in dict.fromkeys(data.get("thread_ids", [])) if i in valid]
    related = [i for i in dict.fromkeys(data.get("related", [])) if i in valid and i not in ids]
    confidence = _clamp(data.get("confidence"), 0, 1, 0.0)
    title = str(data.get("new_thread_title", "")).strip()
    low = bool(ids) and not data.get("new_thread") and confidence < threshold
    if data.get("new_thread") or low:
        related = list(dict.fromkeys([*related, *ids]))  # 没把握挂上去的，至少记成"有关"
        ids = []
    return ThreadDecision(ids, title, related, confidence, str(data.get("reason", "")).strip(), low)


# ====================================================================================
# 三、修改函数参数
# ====================================================================================

TUNING_SYSTEM = """你是一个聊天 AI 的"潜意识"，负责调节它的记忆：哪些事该记得更久，哪些可以忘得快一些。
记忆按遗忘曲线衰减：强度 = 0.5 ^ (经过天数 / 半衰期)，强度低于遗忘线就被遗忘。你不能改公式，只能提出调整参数。

## 可以提出的调整
1. 单条记忆（scope = "memory"，target = 记忆编号）：factor 是半衰期的倍数，大于 1 记得更久，小于 1 忘得更快。
   最典型的情况：新信息让一件旧事变重要了。例如半年前随口说"我妈最近老咳嗽"，今天说"我妈住院了"，
   那条旧记忆应该拉长（factor 2～3）。
2. 全局参数（scope = "global"，target = 参数名）：只有在长期、反复的证据表明整体记性该调时才提，单凭一个片段几乎不该提。
   可选参数：{knobs}

## 规则
- 没有需要调的就返回空列表。大多数片段都不需要调整。
- 每条调整都要写 reason，说明依据的是新片段里的什么信息。
- 系统会限幅：单条记忆单次最多 ×3，全局参数单次最多 ±10%，所以不用给极端值。

只输出 JSON。"""

TUNING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "adjustments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "scope": {"type": "string", "enum": ["memory", "global"]},
                    "target": {"type": "string"},
                    "factor": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["scope", "target", "factor", "reason"],
            },
        }
    },
    "required": ["adjustments"],
}


@dataclass
class Adjustment:
    scope: str  # "memory" | "global"
    target: str  # 记忆编号（字符串）或全局参数名
    factor: float
    reason: str


@dataclass
class MemoryCard:
    """给 Qwen 看的旧记忆卡片。"""

    id: int
    title: str
    age_days: float
    strength: float
    half_life: float
    people: list[str] = field(default_factory=list)


def build_tuning_prompt(
    new_segment: dict[str, Any], related: list[MemoryCard], knobs: ForgettingKnobs
) -> list[Message]:
    knob_desc = "、".join(f"{k}（当前 {getattr(knobs, k):g}）" for k in TUNABLE)
    mems = "\n".join(
        f"[{m.id}] {m.title}｜{m.age_days:.0f} 天前｜强度 {m.strength:.2f}｜半衰期 {m.half_life:.0f} 天"
        f"｜人物：{'、'.join(m.people) or '无'}"
        for m in related
    )
    seg = json.dumps(new_segment, ensure_ascii=False)
    return [
        system(TUNING_SYSTEM.replace("{knobs}", knob_desc)),
        user(f"新片段：{seg}\n\n相关的旧记忆：\n{mems or '（没有）'}"),
    ]


def parse_tuning(data: dict[str, Any], memory_ids: list[int]) -> list[Adjustment]:
    valid_memory = {str(i) for i in memory_ids}
    out = []
    for raw in data.get("adjustments", []):
        scope, target = raw.get("scope"), str(raw.get("target", "")).strip()
        if scope == "memory":  # 模型有时会把整张卡片抄进来，例如 "[1] 他妈妈老咳嗽｜180 天前"，只取开头的编号
            m = re.match(r"\[?\s*(\d+)", target)
            target = m.group(1) if m else target
        try:
            factor = float(raw.get("factor"))
        except (TypeError, ValueError):
            continue
        if factor <= 0 or factor == 1:
            continue
        if (scope == "memory" and target in valid_memory) or (scope == "global" and target in TUNABLE):
            out.append(Adjustment(scope, target, factor, str(raw.get("reason", "")).strip()))
    return out
