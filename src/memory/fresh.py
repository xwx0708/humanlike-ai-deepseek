"""新鲜记忆：刚说过的话（决策 D30、D31，用户 2026-10-28 定）。

- 运行期的上下文里没有"聊天记录"这一项。用户原话："所谓的本场对话就是几乎未衰减的记忆。"
  只在运行期用；启动之前上下文就是普通的聊天记录（用户说明）。
- 每条消息从说出那一刻的满格开始，按记忆时钟（D21）衰减。强度不低于 floor 的稳定在场，不掷骰子；
  低于的不再原样出现，只能靠浮现、检索想起。还没打分，半衰期用平淡小事的基础值。
- 对话中途分批整理（D31）：最早一条还没整理的消息强度降到 split_at 时，把强度低于 split_upto 的那部分
  先切段放进概率海，免得它们掉出新鲜记忆时还没整理、两头不靠。切了段的消息只要强度不低于 floor 仍然在场；
  此时它所在的片段若也浮现，用 drop_fresh 去掉重复。
- floor、split_at、split_upto 都是暂定旋钮。
"""

from dataclasses import dataclass, replace

from contracts.knobs import knob
from contracts.memory import ChatTurn
from memory.budget import DEFAULT_KNOBS as BUDGET
from memory.budget import truncate
from memory.clock import MemoryClock, clock_of
from memory.forgetting import ForgettingKnobs, strength
from memory.store import MemoryStore
from memory.tokens import estimate_tokens
from runtime import config

FRESH_HEADER = "刚才的对话（记得很清楚）："


@dataclass(frozen=True)
class FreshKnobs:
    # 半衰期 7 天、对话中流速 10 倍时：0.9 约是 2.5 小时前的话，0.92 约 2 小时，0.96 约 1 小时
    floor: float = knob(
        0.9, "在场下限", "强度不低于此值的消息稳定在场，不经采样。", lo=0.5, hi=0.99
    )  # 强度不低于它的稳定在场（暂定）
    half_life: float = knob(
        ForgettingKnobs().base_half_life,
        "新鲜记忆半衰期",
        "尚未打分的消息按此半衰期衰减。",
        lo=0.5,
        hi=60,
        step=0.5,
        unit="天",
    )  # 天：还没打分，按平淡小事算
    split_at: float = knob(
        0.92,
        "中途整理触发",
        "最早一条未整理消息的强度降到此值时，对话中途先整理一批。须高于在场下限。",
        lo=0.5,
        hi=0.999,
        step=0.005,
    )  # 最早一条未整理的消息降到这里，就中途整理一批（暂定，要早于 floor）
    split_upto: float = knob(
        0.96,
        "中途整理范围",
        "这一批整理到强度低于此值的消息为止，更新的留到下一批。须高于触发值。",
        lo=0.5,
        hi=0.999,
        step=0.005,
    )  # 这一批整理到强度低于它的消息为止，最近约 1 小时的话留到下一批（暂定）


DEFAULT_KNOBS = config.knobs(
    FreshKnobs(),
    "fresh",
    "新鲜记忆",
    "运行期不另带聊天记录：刚说过的话作为几乎未衰减的记忆在场（D30、D31）。",
)


@dataclass(frozen=True)
class FreshMemory:
    turn: ChatTurn
    strength: float


def turn_strength(turn: ChatTurn, now_ts: float, clock: MemoryClock, k: FreshKnobs = DEFAULT_KNOBS) -> float:
    return strength(k.half_life, clock.elapsed_days(turn.ts, now_ts))


def fresh_memories(
    store: MemoryStore,
    now_ts: float,
    clock: MemoryClock | None = None,
    k: FreshKnobs = DEFAULT_KNOBS,
) -> list[FreshMemory]:
    """此刻在场的新鲜记忆，按说出的先后排列。"""
    clock = clock or clock_of(store)
    out = []
    for t in store.iter_turns_newest_first():
        s = turn_strength(t, now_ts, clock, k)
        if s < k.floor:
            break  # 越往前越弱，后面的都不新鲜了
        out.append(FreshMemory(t, s))
    out.reverse()
    return out


def fit_fresh(items: list[FreshMemory], budget: int | None = None) -> list[FreshMemory]:
    """放进新鲜记忆预算（memory.budget）：从最近一条往前放，越早的越先让出位置；最近一条单独就超出时截短。
    让出位置的消息不丢，照常整理进概率海，之后可能浮现。"""
    budget = BUDGET.fresh if budget is None else budget
    budget -= estimate_tokens(FRESH_HEADER) + 1
    out, used = [], 0
    for m in reversed(items):
        cost = estimate_tokens(m.turn.content) + 4  # 加上"对方："和换行
        if used + cost > budget:
            if not out:
                out.append(
                    FreshMemory(replace(m.turn, content=truncate(m.turn.content, budget - 4)), m.strength)
                )
            break
        out.append(m)
        used += cost
    out.reverse()
    return out


def split_point(
    store: MemoryStore,
    conversation_id: int,
    now_ts: float,
    clock: MemoryClock | None = None,
    k: FreshKnobs = DEFAULT_KNOBS,
) -> float | None:
    """对话中途该不该整理一批：该的话返回这一批整理到哪个时刻（交给 consolidate 的 until_ts），不该返回 None。"""
    turns = store.unsegmented_turns(conversation_id)
    if not turns:
        return None
    clock = clock or clock_of(store)
    if turn_strength(turns[0], now_ts, clock, k) >= k.split_at:
        return None
    older = [t.ts for t in turns if turn_strength(t, now_ts, clock, k) < k.split_upto]
    return max(older)


def drop_fresh(store: MemoryStore, surfaced: list, fresh: list[FreshMemory]) -> list:
    """去掉浮现区里和新鲜记忆重复的：片段的消息全都还在新鲜记忆里，就不必再浮现一次。"""
    ids = {m.turn.id for m in fresh}
    out = []
    for s in surfaced:
        try:
            seg = store.get_segment(s.segment_id)
        except KeyError:
            continue
        if seg.state not in ("active", "abstract"):
            continue
        own = set(seg.message_ids)
        if not (own and own <= ids):
            out.append(s)
    return out


def fresh_block(items: list[FreshMemory]) -> str:
    """给回应模块放进提示词的一段文字。"""
    if not items:
        return ""
    lines = [FRESH_HEADER]
    for m in items:
        who = "对方" if m.turn.role == "user" else "我"
        lines.append(f"{who}：{m.turn.content}")
    return "\n".join(lines)
