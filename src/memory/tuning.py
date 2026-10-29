"""潜意识修改全局参数时的护栏：窄幅、慢速、留日志。

说明见 01-记忆/潜意识.md。单条记忆的调整护栏在 forgetting.clamp_adjustment。
"""

from dataclasses import dataclass, fields, replace

from memory.forgetting import ForgettingKnobs
from runtime import config

# 允许潜意识修改的全局参数。小模型调整自身上限的参数（adjust_*）不在其中，避免它给自己松绑。
TUNABLE = ("base_half_life", "emotion_max", "importance_max", "reinforce", "position_bonus", "forget_line")


@dataclass(frozen=True)
class GlobalLimits:
    step_max: float = 1.1  # 窄幅：单次最多变化 ±10%
    range_max: float = 1.5  # 窄幅：始终保持在默认值的 1/1.5～1.5 倍之间
    cooldown_days: float = 7.0  # 慢速：同一参数两次修改至少间隔 7 天


DEFAULT_LIMITS = GlobalLimits()
DEFAULT_KNOBS = config.knobs(ForgettingKnobs(), "forgetting")  # "允许范围"以这组默认值为中心


@dataclass(frozen=True)
class GlobalChange:
    """一条全局参数调整日志。被拒绝的请求也记录。"""

    knob: str
    at: float  # 时间（天）
    old: float
    requested: float  # 小模型请求的新值
    new: float  # 限幅后实际采用的值；被拒绝时等于 old
    applied: bool
    reason: str  # 小模型给出的理由
    note: str = ""  # 护栏的处理说明


def adjust_global(
    knobs: ForgettingKnobs,
    knob: str,
    factor: float,
    now: float,
    reason: str,
    log: list[GlobalChange],
    limits: GlobalLimits = DEFAULT_LIMITS,
    defaults: ForgettingKnobs = DEFAULT_KNOBS,
) -> tuple[ForgettingKnobs, GlobalChange]:
    """按 factor 倍调整一个全局参数。返回新的参数和日志条目；调用方负责把条目追加到日志里。"""
    if knob not in TUNABLE:
        raise ValueError(f"{knob} 不允许潜意识修改")
    if factor <= 0:
        raise ValueError("factor 必须大于 0")

    old = getattr(knobs, knob)
    requested = old * factor
    last = max((c.at for c in log if c.knob == knob and c.applied), default=None)
    if last is not None and now - last < limits.cooldown_days:
        wait = limits.cooldown_days - (now - last)
        return knobs, GlobalChange(
            knob, now, old, requested, old, False, reason, f"冷却中，还需 {wait:.1f} 天"
        )

    notes = []
    step = min(limits.step_max, max(1.0 / limits.step_max, factor))
    if step != factor:
        notes.append(f"单次限幅为 ×{step:.3g}")
    default = getattr(defaults, knob)
    new = min(default * limits.range_max, max(default / limits.range_max, old * step))
    if new != old * step:
        notes.append("已到允许范围的边界")
    if new == old:
        return knobs, GlobalChange(
            knob, now, old, requested, old, False, reason, "；".join(notes) or "没有变化"
        )
    return replace(knobs, **{knob: new}), GlobalChange(
        knob, now, old, requested, new, True, reason, "；".join(notes)
    )


def knob_names() -> list[str]:
    return [f.name for f in fields(ForgettingKnobs)]
