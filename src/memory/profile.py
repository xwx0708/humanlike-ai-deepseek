"""用户画像的矛盾处理（决策 D16，用户 2026-10-28 定）。

新说法和画像里的旧记录矛盾时，按旧记录的牢固度掷一次骰子，三选一：
- 唤起：旧记录被重新想起（交给浮现区，思考模型可能会问"你之前不是说……"），两条都保留；
- 覆盖：以新说法为准，旧记录移进档案（不删除，可以找回）；
- 都不：两条都保留，暂不处理。
旧记录越牢固，越容易被唤起、越难被覆盖。具体数值是暂定的旋钮。
"""

import random
from dataclasses import dataclass

from contracts.memory import ProfileFact, days

EVOKE, OVERWRITE, NOTHING = "唤起", "覆盖", "都不"


@dataclass(frozen=True)
class ConflictKnobs:
    firm_half_life: float = 180.0  # 天：旧记录多久没被提到，牢固度减半
    evoke_min: float = 0.2  # 最不牢固的旧记录被唤起的概率
    evoke_max: float = 0.8  # 最牢固的旧记录被唤起的概率
    overwrite_max: float = 0.8  # 最不牢固的旧记录被覆盖的概率


DEFAULT_KNOBS = ConflictKnobs()


def firmness(fact: ProfileFact, now_ts: float, k: ConflictKnobs = DEFAULT_KNOBS) -> float:
    """0～1。被提到的次数越多、最近一次越近，越牢固。"""
    by_mentions = 1 - 0.5**fact.mentions  # 1 次 0.5，2 次 0.75，3 次 0.875
    by_recency = 0.5 ** (max(0.0, days(now_ts - fact.last_ts)) / k.firm_half_life)
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
    old: ProfileFact, now_ts: float, rng: random.Random, k: ConflictKnobs = DEFAULT_KNOBS
) -> ConflictResult:
    firm = firmness(old, now_ts, k)
    probs = outcome_probabilities(firm, k)
    roll = rng.random()
    if roll < probs[EVOKE]:
        outcome = EVOKE
    elif roll < probs[EVOKE] + probs[OVERWRITE]:
        outcome = OVERWRITE
    else:
        outcome = NOTHING
    return ConflictResult(outcome, firm, probs, roll)
