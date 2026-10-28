"""新鲜记忆：刚说过的话（决策 D30，用户 2026-10-28 定）。

- 运行期的上下文里没有"聊天记录"这一项。用户原话："所谓的本场对话就是几乎未衰减的记忆。"
  只在运行期用；启动之前上下文就是普通的聊天记录（用户说明）。
- 还没整理（没切进片段）的消息就是新鲜记忆：强度从说出那一刻的满格开始，按记忆时钟（D21）衰减。
  强度不低于 floor 的稳定在场，不掷骰子；低于的不再原样出现，等整理成片段后只能靠浮现想起。
- 还没打分，半衰期用平淡小事的基础值（base_half_life）。floor 是暂定旋钮。
- 和浮现区不会重复：浮现区里都是片段，新鲜记忆都是还没切进片段的消息。
"""

from dataclasses import dataclass

from contracts.memory import ChatTurn
from memory.clock import MemoryClock, clock_of
from memory.forgetting import ForgettingKnobs, strength
from memory.store import MemoryStore


@dataclass(frozen=True)
class FreshKnobs:
    floor: float = 0.9  # 强度不低于它的新鲜记忆稳定在场（暂定）。半衰期 7 天、对话中流速 10 倍时，约等于最近 2.5 小时的话
    half_life: float = ForgettingKnobs().base_half_life  # 天：还没打分，按平淡小事算


DEFAULT_KNOBS = FreshKnobs()


@dataclass(frozen=True)
class FreshMemory:
    turn: ChatTurn
    strength: float


def fresh_memories(
    store: MemoryStore,
    now_ts: float,
    clock: MemoryClock | None = None,
    k: FreshKnobs = DEFAULT_KNOBS,
) -> list[FreshMemory]:
    """此刻在场的新鲜记忆，按说出的先后排列。"""
    clock = clock or clock_of(store)
    out = []
    for cid in store.conversation_ids():
        for t in store.unsegmented_turns(cid):
            s = strength(k.half_life, clock.elapsed_days(t.ts, now_ts))
            if s >= k.floor:
                out.append(FreshMemory(t, s))
    out.sort(key=lambda m: (m.turn.ts, m.turn.id or 0))
    return out


def fresh_block(items: list[FreshMemory]) -> str:
    """给回应模块放进提示词的一段文字。"""
    if not items:
        return ""
    lines = ["刚才的对话（记得很清楚）："]
    for m in items:
        who = "对方" if m.turn.role == "user" else "我"
        lines.append(f"{who}：{m.turn.content}")
    return "\n".join(lines)
