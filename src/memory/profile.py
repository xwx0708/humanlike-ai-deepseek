"""用户画像的矛盾处理（决策 D16，用户 2026-10-28 定）。

新说法和画像里的旧记录矛盾时，按旧记录的牢固度掷一次骰子，三选一：
- 唤起：旧记录被重新想起（交给浮现区，思考模型可能会问"你之前不是说……"），两条都保留；
- 覆盖：以新说法为准，旧记录移进档案（不删除，可以找回）；
- 都不：两条都保留，暂不处理。
旧记录越牢固，越容易被唤起、越难被覆盖。具体数值是暂定的旋钮。
"""

import random
from dataclasses import dataclass

from contracts.knobs import knob
from contracts.memory import ProfileFact
from runtime import config

EVOKE, OVERWRITE, NOTHING = "唤起", "覆盖", "都不"


@dataclass(frozen=True)
class ConflictKnobs:
    firm_half_life: float = knob(
        180.0, "牢固度半衰期", "旧事实多久未被提及，牢固度减半。", lo=1, hi=3650, step=1, unit="天"
    )  # 天：旧记录多久没被提到，牢固度减半
    evoke_min: float = knob(
        0.2, "最低唤起概率", "最不牢固的旧事实被唤起的概率。", lo=0, hi=1, step=0.05
    )  # 最不牢固的旧记录被唤起的概率
    evoke_max: float = knob(
        0.8, "最高唤起概率", "最牢固的旧事实被唤起的概率。", lo=0, hi=1, step=0.05
    )  # 最牢固的旧记录被唤起的概率
    overwrite_max: float = knob(
        0.8, "最高覆盖概率", "最不牢固的旧事实被覆盖（移入档案）的概率。", lo=0, hi=1, step=0.05
    )  # 最不牢固的旧记录被覆盖的概率


DEFAULT_KNOBS = config.knobs(
    ConflictKnobs(),
    "profile",
    "画像冲突",
    "新事实与画像中的旧事实矛盾时，按旧事实的牢固度掷骰子：唤起 / 覆盖 / 都不（D16）。",
)


def firmness(fact: ProfileFact, idle_days: float, k: ConflictKnobs = DEFAULT_KNOBS) -> float:
    """0～1。被提到的次数越多、最近一次越近，越牢固。idle_days：最近一次被提到后经过的记忆时间（D21）。"""
    by_mentions = 1 - 0.5**fact.mentions  # 1 次 0.5，2 次 0.75，3 次 0.875
    by_recency = 0.5 ** (max(0.0, idle_days) / k.firm_half_life)
    return by_mentions * by_recency


def outcome_probabilities(firm: float, k: ConflictKnobs = DEFAULT_KNOBS) -> dict[str, float]:
    evoke = k.evoke_min + (k.evoke_max - k.evoke_min) * firm
    overwrite = k.overwrite_max * (1 - firm)
    total = evoke + overwrite
    if total > 1:
        evoke, overwrite = evoke / total, overwrite / total
    return {EVOKE: evoke, OVERWRITE: overwrite, NOTHING: max(0.0, 1 - evoke - overwrite)}


@dataclass(frozen=True)
class ConflictResult:
    outcome: str  # 唤起 / 覆盖 / 都不
    firmness: float
    probabilities: dict[str, float]
    roll: float  # 掷出的随机数，记进日志便于复查


def resolve_conflict(
    old: ProfileFact, idle_days: float, rng: random.Random, k: ConflictKnobs = DEFAULT_KNOBS
) -> ConflictResult:
    firm = firmness(old, idle_days, k)
    probs = outcome_probabilities(firm, k)
    roll = rng.random()
    if roll < probs[EVOKE]:
        outcome = EVOKE
    elif roll < probs[EVOKE] + probs[OVERWRITE]:
        outcome = OVERWRITE
    else:
        outcome = NOTHING
    return ConflictResult(outcome, firm, probs, roll)


def take_evocations(store) -> list[str]:
    """被唤起、等待浮现的旧事实（D16"唤起"：两条都留，旧的等待浮现）。返回给思考模型看的句子，并标为已浮现，
    所以每条只浮现一次。由回应模块在运行期每轮调用，放进浮现区。"""
    out = []
    for e in store.pending_evocations():
        old = store.fact_text(e["fact_id"])
        new = store.fact_text(e["trigger_fact_id"]) if e["trigger_fact_id"] else None
        if old:
            out.append(f"你想起他以前说过：{old}" + (f"（和他后来说的「{new}」不一样）" if new else ""))
        store.consume_evocation(e["id"])
    return out
