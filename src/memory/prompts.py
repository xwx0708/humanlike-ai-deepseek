"""潜意识模型的提示词：切段第二层、判断事件线归属、修改函数参数、更新事件线摘要、画像矛盾、遗忘后提取关键词。

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
from contracts.memory import PROFILE_CATEGORIES, ChatTurn, EmotionScore
from memory.forgetting import ForgettingKnobs
from memory.tuning import TUNABLE


def _clamp(x: Any, lo: float, hi: float, default: float) -> float:
    try:
        return min(hi, max(lo, float(x)))
    except (TypeError, ValueError):
        return default


# 画像事实里不该出现的相对时间词：画像永远不会被遗忘，过一阵子这类话就不对了
_RELATIVE_TIME = re.compile(
    r"近期|最近|近来|这几天|这阵子|目前|现在|刚刚|刚才|今天|明天|昨天|后天|本周|这周|下周|上周|本月|这个月|下个月|上个月|今年|明年|去年"
)


def _hm(ts: float) -> str:
    return time.strftime("%m-%d %H:%M", time.localtime(ts))


# ====================================================================================
# 一、切段第二层：确认片段
# ====================================================================================

SEGMENT_SYSTEM = """你是一个聊天 AI 的"潜意识"，在后台整理它和用户的聊天记录。现在要把一段对话切成"片段"。

## 什么是片段
- 片段是对话中围绕同一件事的一个子话题，必须能用一句话概括。
- 太细和太粗都不行：一个片段撑不起一句话，就和相邻片段合并；标题需要用"并""同时""还""另外"把几件事串起来，
  就说明切得太粗，要拆开（例如"妈妈住院"和"老板对请假不满"是两个片段，即使后者是前者引起的）。
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
- facts：片段里出现的**长期稳定的事实**，会存进用户画像，永远不会被遗忘，所以宁缺毋滥。
  - 只收不会很快改变的事：过敏和忌口、长期病史、家人和关系（"他妹妹叫小雨"）、生日和纪念日、职业和学校、
    住在哪里、长期的喜好。
  - **不收**一时的状态和事件："妈妈住院了""今天很累""周末要去爬山"都不是。
  - **不收**带"明天""下周""下个月""最近"这类相对时间的事：画像永远不会被遗忘，过一阵子这类话就不对了。
    即将发生的事（"妹妹下个月结婚"）是事件，不是事实；只有事情本身变成长期事实时才收（"妹妹已婚"）。
  - about：说的是谁。用户本人写"用户"，其他人写 people 里的常用叫法。
  - category：从 {categories} 中选一个。
  - fact：一句简短的陈述，例如"对花生过敏""在杭州工作"。
  - 没有就给空列表。大多数片段都没有。
- remember_requested：用户是否**明确要求**记住这个片段里的事（"记住这个""你别忘了""帮我记一下"）。
  只看用户有没有这样说，不看事情重不重要。

只输出 JSON。""".replace("{categories}", "、".join(PROFILE_CATEGORIES))

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
                    "facts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "about": {"type": "string"},
                                "category": {"type": "string", "enum": list(PROFILE_CATEGORIES)},
                                "fact": {"type": "string"},
                            },
                            "required": ["about", "category", "fact"],
                        },
                    },
                    "remember_requested": {"type": "boolean"},
                },
                "required": [
                    "messages",
                    "title",
                    "people",
                    "arousal",
                    "valence",
                    "importance",
                    "emotion_reason",
                    "facts",
                    "remember_requested",
                ],
            },
        }
    },
    "required": ["segments"],
}


@dataclass
class SegmentDraft:
    """潜意识模型切出、经校验的片段，还没有存进数据库。"""

    turn_ids: list[int]
    title: str
    people: list[tuple[str, list[str]]]  # (常用叫法, 其他叫法)
    emotion: EmotionScore
    facts: list[tuple[str, str, str]] = field(
        default_factory=list
    )  # (说的是谁, 类别, 事实)；"用户"指用户本人
    remember_requested: bool = False  # 用户明确要求记住（常驻栏免竞争，D17）


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
        facts = []
        for f in raw.get("facts", []) or []:
            text = str(f.get("fact", "")).strip()
            if text and not _RELATIVE_TIME.search(text):  # 带相对时间的不是稳定事实，模型漏判时在这里拦住
                category = f.get("category") if f.get("category") in PROFILE_CATEGORIES else "其他"
                facts.append((str(f.get("about", "")).strip() or "用户", category, text))
        drafts.append(
            SegmentDraft(
                [turns[i - 1].id for i in idx],
                title,
                people,
                emotion,
                facts,
                raw.get("remember_requested") is True,
            )
        )
    if not drafts:
        raise ValueError("潜意识模型没有切出任何有效片段")

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
- 候选分两栏："常驻事件线"是一直记得的事，"其他候选"是这次检索出来的。两栏一样对待，都可以选。
- 两栏都没有时，必须新建，只需要给出标题。
- confidence 是你的把握，0～1。拿不准时给低分，系统会先新建，晚上再检查要不要合并。错误地合并比错误地新建更难发现。
- related 列出"有关但不是同一件事"的事件线编号，用来建立关联，可以为空。
- new_thread_ongoing：只在新建时有意义。这件事现在是否还在进行（还没有结局、用户还在为它操心为 true；一次性告知的事实、已经结束的事为 false）。

只输出 JSON。"""

THREAD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "thread_ids": {"type": "array", "items": {"type": "integer"}},
        "new_thread": {"type": "boolean"},
        "new_thread_title": {"type": "string"},
        "new_thread_ongoing": {"type": "boolean"},
        "related": {"type": "array", "items": {"type": "integer"}},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": [
        "thread_ids",
        "new_thread",
        "new_thread_title",
        "new_thread_ongoing",
        "related",
        "confidence",
        "reason",
    ],
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
    new_ongoing: bool = True  # 新建时：这件事是否还在进行（D17）

    @property
    def create(self) -> bool:
        return not self.attach_to


def _thread_cards(cards: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"[{c['id']}] {c['title']}｜摘要：{c.get('summary', '')}｜最近：{c.get('latest_segment', '')}"
        f"｜人物：{'、'.join(c.get('people', [])) or '无'}"
        for c in cards
    )


def build_thread_prompt(
    segment: dict[str, Any], candidates: list[dict[str, Any]], residents: list[dict[str, Any]] = ()
) -> list[Message]:
    """segment / candidates / residents 都是卡片：标题、一句话摘要（片段可无）、最近片段标题、涉及人物。
    residents 是常驻栏（D17），单独一栏给出，不占 candidates 的名额。"""
    seg = f"标题：{segment['title']}｜人物：{'、'.join(segment.get('people', [])) or '无'}"
    if segment.get("context"):
        seg += f"｜前后文：{segment['context']}"
    body = (
        f"新片段：\n{seg}\n\n常驻事件线：\n{_thread_cards(list(residents)) or '（没有）'}"
        f"\n\n其他候选：\n{_thread_cards(candidates) or '（没有）'}"
    )
    return [system(THREAD_SYSTEM), user(body)]


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
    return ThreadDecision(
        ids,
        title,
        related,
        confidence,
        str(data.get("reason", "")).strip(),
        low,
        data.get("new_thread_ongoing") is not False,
    )


# ====================================================================================
# 三、修改函数参数
# ====================================================================================

TUNING_SYSTEM = """你是一个聊天 AI 的"潜意识"，负责调节它的记忆：哪些事该记得更久，哪些可以忘得快一些。
记忆按遗忘曲线衰减：强度 = 0.5 ^ (经过天数 / 半衰期)，强度低于遗忘线就被遗忘。你不能改公式，只能提出调整参数。

## 可以提出的调整
1. 单条记忆（scope = "memory"，target = 记忆编号）：factor 是半衰期的倍数，大于 1 记得更久，小于 1 忘得更快。
   最典型的情况：新信息让一件旧事变重要了。例如半年前随口说"我妈最近老咳嗽"，今天说"我妈住院了"，
   那条旧记忆应该拉长（factor 2～3）。
   **只有标着"事件已完结"的记忆才可以缩短（factor 小于 1）**，例如事情已经解决，其中的细节可以忘得快一些；
   标着"事件进行中"的只能拉长或不动。
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
    """给潜意识模型看的旧记忆卡片。"""

    id: int
    title: str
    age_days: float
    strength: float
    half_life: float
    people: list[str] = field(default_factory=list)
    finished: bool = False  # 所在事件线都已完结（D19：只有完结后才允许把系数调到小于 1）


def build_tuning_prompt(
    new_segment: dict[str, Any], related: list[MemoryCard], knobs: ForgettingKnobs
) -> list[Message]:
    knob_desc = "、".join(f"{k}（当前 {getattr(knobs, k):g}）" for k in TUNABLE)
    mems = "\n".join(
        f"[{m.id}] {m.title}｜{m.age_days:.0f} 天前｜强度 {m.strength:.2f}｜半衰期 {m.half_life:.0f} 天"
        f"｜人物：{'、'.join(m.people) or '无'}｜{'事件已完结' if m.finished else '事件进行中'}"
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


# ====================================================================================
# 四、更新事件线摘要
# ====================================================================================

SUMMARY_SYSTEM = """你是一个聊天 AI 的"潜意识"。一条事件线（跨越多天的同一件事）新增了片段，请更新它的摘要。

## 规则
- summary：一句话，按时间顺序概括这件事的来龙去脉和最新进展，不超过 60 个字。例如"他妈妈先是咳嗽，后来查出肺炎住院，现在已出院"。
- 用第三人称"他"指用户。只写片段里有的事，不要推测。
- new_title：标题一般不改，留空。只有原标题明显概括不了这件事时，才给出新的短标题（不超过 12 个字）。
- ongoing：这件事**现在是否还在进行**。还没有结局、还在发展、用户还在为它操心，就是 true（"妈妈住院中""在考虑辞职"）；
  已经有了结局或告一段落，就是 false（"妈妈已出院、恢复良好""已经换了工作"）。

只输出 JSON。"""

SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "new_title": {"type": "string"},
        "ongoing": {"type": "boolean"},
    },
    "required": ["summary", "new_title", "ongoing"],
}


def build_summary_prompt(title: str, summary: str, segments: list) -> list[Message]:
    """segments：这条线上的片段（按时间排序），只用到 start_ts 和 title。"""
    lines = "\n".join(
        f"- {time.strftime('%Y-%m-%d', time.localtime(s.start_ts))}：{s.title}" for s in segments
    )
    return [
        system(SUMMARY_SYSTEM),
        user(f"事件线标题：{title}\n原摘要：{summary}\n\n各个片段（按时间）：\n{lines}"),
    ]


def parse_summary(data: dict[str, Any], old_title: str) -> tuple[str, str, bool]:
    """返回 (新摘要, 标题, 是否仍在进行)。摘要为空时报错；新标题为空或过长时沿用原标题；
    没给出 ongoing 时按"仍在进行"处理。"""
    summary = str(data.get("summary", "")).strip()
    if not summary:
        raise ValueError("潜意识模型没有给出摘要")
    title = str(data.get("new_title", "")).strip()
    return summary[:120], (title if 0 < len(title) <= 20 else old_title), data.get("ongoing") is not False


# ====================================================================================
# 五、判断画像事实是否矛盾（决策 D16）
# ====================================================================================

CONFLICT_SYSTEM = """你是一个聊天 AI 的"潜意识"，在维护用户画像（长期稳定的事实）。现在有一条新事实，
判断它和哪些旧事实矛盾。

## 规则
- 矛盾：两条说的是同一件事，但不能同时成立。例如"在杭州工作"和"在上海工作"；"对花生过敏"和"不对花生过敏"；
  "妹妹叫小雨"和"妹妹叫小雪"（如果明显说的是同一个妹妹）。
- 不算矛盾：只是补充或更具体（"在杭州工作"和"在杭州做前端"）；说的是不同的事（"对花生过敏"和"对芒果过敏"）；
  两条可以同时成立（"喜欢跑步"和"喜欢游泳"）。
- 拿不准就不算矛盾。

只输出 JSON。"""

CONFLICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "conflicts": {"type": "array", "items": {"type": "integer"}},
        "reason": {"type": "string"},
    },
    "required": ["conflicts", "reason"],
}


def build_conflict_prompt(new_fact: str, old_facts: list[tuple[int, str]], about: str) -> list[Message]:
    olds = "\n".join(f"[{i}] {f}" for i, f in old_facts)
    return [system(CONFLICT_SYSTEM), user(f"说的是：{about}\n新事实：{new_fact}\n\n旧事实：\n{olds}")]


def parse_conflicts(data: dict[str, Any], old_ids: list[int]) -> tuple[list[int], str]:
    valid = set(old_ids)
    ids = [i for i in dict.fromkeys(data.get("conflicts", [])) if i in valid]
    return ids, str(data.get("reason", "")).strip()


# ====================================================================================
# 六、遗忘后抽象化：只提取简要关键词（决策 D22）
# ====================================================================================

KEYWORDS_SYSTEM = """你是一个聊天 AI 的"潜意识"。下面这些记忆已经快忘了，每条只留几个简要的关键词，
细节都不要了。以后它们只以这几个词的样子留在记忆里。

## 规则
- 每条 2～4 个关键词，每个不超过 6 个字。
- 留下最能让人隐约想起这件事的词：涉及的人、事情的核心、当时的情绪。例如"妈妈、住院、担心"。
- 不要写成句子，不要编造原文没有的东西。

只输出 JSON。"""

KEYWORDS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "keywords": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "keywords"],
            },
        }
    },
    "required": ["items"],
}


def build_keywords_prompt(memories: list[tuple[int, str, str]]) -> list[Message]:
    """memories：(编号, 标题, 原文节选)。"""
    body = "\n\n".join(f"[{i}] {title}\n原文：{text}" for i, title, text in memories)
    return [system(KEYWORDS_SYSTEM), user(body)]


def parse_keywords(data: dict[str, Any], ids: list[int]) -> dict[int, list[str]]:
    """每条最多留 4 个、每个最多 6 个字；模型漏掉的条目不在结果里，由调用方兜底。"""
    valid, out = set(ids), {}
    for raw in data.get("items", []):
        i = raw.get("id")
        words = [str(w).strip()[:6] for w in raw.get("keywords", []) if str(w).strip()]
        if i in valid and words:
            out[i] = list(dict.fromkeys(words))[:4]
    return out


# ====================================================================================
# 七、一场对话的摘要：随子片段遗忘重新概括（事件切分.md"遗忘效果"；待办 #7）
# ====================================================================================

CONVERSATION_SYSTEM = """你是一个聊天 AI 的"潜意识"。请用一句话概括它和用户的一场聊天。

## 规则
- 不超过 40 个字，用第三人称"他"指用户，例如"那晚从他妈妈住院一直聊到他想换工作"。
- 给你的是这场聊天里**现在还记得**的部分：完整的片段标题，或者只剩几个关键词的模糊片段。
- 记得的越少，概括越抽象、越模糊；只剩关键词时，只写出大概，不要补出细节，不要编造。

只输出 JSON。"""

CONVERSATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
}


def build_conversation_prompt(date: str, parts: list[str]) -> list[Message]:
    """parts：还记得的片段，按时间顺序；只剩关键词的以"（模糊）"开头。"""
    body = "\n".join(f"- {p}" for p in parts)
    return [system(CONVERSATION_SYSTEM), user(f"日期：{date}\n还记得的部分：\n{body}")]


def parse_conversation(data: dict[str, Any]) -> str:
    summary = str(data.get("summary", "")).strip()
    if not summary:
        raise ValueError("潜意识模型没有给出对话摘要")
    return summary[:80]
