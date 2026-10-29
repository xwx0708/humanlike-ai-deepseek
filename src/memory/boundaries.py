"""切段第一层：实时规则（只用 Python，不调用模型）。

每条消息进来时：
- 距上一条超过时间间隔 → 开一场新对话（硬边界，不需要第二层确认）。
- 命中长度上限、时长上限、跨天、语言信号 → 标一个候选边界，交给第二层的潜意识模型确认。
话题跳变和漂移需要理解语义，不在这一层判断（决策 D10）。见 01-记忆/事件切分.md。
"""

import re
import time
from dataclasses import dataclass

from contracts.memory import ChatTurn
from memory.store import MemoryStore
from runtime import config


@dataclass(frozen=True)
class BoundaryRules:
    """阈值都是暂定值，待用真实聊天记录校准。"""

    conversation_gap_minutes: float = 30  # 超过这么久没有消息，就是新的一场对话
    max_turns: int = 30  # 距上一个候选边界（或对话开头）满这么多条消息，强制标一次
    max_minutes: float = 60  # 距上一个候选边界满这么久，强制标一次


DEFAULT_RULES = config.knobs(BoundaryRules(), "boundaries")

# 出现在句首才算的信号："对了"在句中常常不是换话题
_LEADING = (
    "对了",
    "话说",
    "另外",
    "还有个事",
    "还有件事",
    "顺便问",
    "顺便说",
    "说个事",
    "说件事",
    "问你个事",
    "跟你说",
)
# 出现在哪里都算的信号
_ANYWHERE = ("换个话题", "不聊这个了", "不说这个了", "不提了", "说回", "回到刚才", "言归正传", "扯远了")
_PUNCT = re.compile(r"^[\s，。！？、,.!?…~～：:；;“”\"'（）()]+")


def language_signal(text: str) -> str | None:
    """返回命中的信号词；没有命中返回 None。"""
    head = _PUNCT.sub("", text)
    for w in _LEADING:
        if head.startswith(w):
            return w
    for w in _ANYWHERE:
        if w in text:
            return w
    return None


def _day(ts: float) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(ts))


def ingest_turn(
    store: MemoryStore, role: str, content: str, ts: float, rules: BoundaryRules = DEFAULT_RULES
) -> ChatTurn:
    """存入一条消息，并按规则分对话、标候选边界。返回存好的消息。"""
    conv = store.last_conversation()
    if conv is None or ts - conv.end_ts > rules.conversation_gap_minutes * 60:
        conv = store.new_conversation(ts)
        return store.add_turn(conv.id, role, content, ts)  # 新对话的第一条本身就是边界，不用再标

    previous = store.turns(conv.id)
    marked = store.boundaries(conv.id)
    turn = store.add_turn(conv.id, role, content, ts)

    reasons = []
    if role == "user" and (w := language_signal(content)):
        reasons.append(f"语言信号：{w}")
    if _day(ts) != _day(previous[-1].ts):
        reasons.append("跨天")
    if not reasons:  # 这条已经是边界的话，长度和时长从这里重新算，不用再检查
        since = [t for t in previous if t.id >= max(marked, default=0)]  # 从上一个候选边界（或对话开头）算起
        if len(since) >= rules.max_turns:
            reasons.append("长度上限")
        if ts - since[0].ts >= rules.max_minutes * 60:
            reasons.append("时长上限")
    for r in reasons:
        store.add_boundary(turn.id, r)
    return turn


def conversation_closed(
    store: MemoryStore, conversation_id: int, now_ts: float, rules: BoundaryRules = DEFAULT_RULES
) -> bool:
    """这场对话是否已经结束：后面已经开了新对话，或者已经安静超过时间间隔。"""
    conv = store.get_conversation(conversation_id)
    last = store.last_conversation()
    return (
        last is not None and last.id != conversation_id
    ) or now_ts - conv.end_ts > rules.conversation_gap_minutes * 60
