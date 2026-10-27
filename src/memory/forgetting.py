"""遗忘曲线：半衰期怎么算、强度怎么随时间衰减、回忆怎么加固。

公式说明见 01-记忆/遗忘曲线.md。时间单位统一为"天"（浮点数），由调用方从注入的时钟换算，
本模块不读取系统时间。
"""

import math
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ForgettingKnobs:
    """可调旋钮。默认值让平淡小事约三周被遗忘、极强烈的事约七个半月。"""

    base_half_life: float = 7.0  # 平淡小事的半衰期（天）
    emotion_max: float = 10.0  # 相对唤醒度为 1 时，半衰期放大的倍数
    importance_max: float = 5.0  # 重要性为 1 时，半衰期放大的倍数
    reinforce: float = 1.0  # 回忆加固力度：快忘光时被想起，半衰期最多变为 (1 + reinforce) 倍
    position_bonus: float = 1.5  # 一场对话的首个、末个片段的半衰期倍数
    forget_line: float = 0.1  # 强度低于此值即被遗忘
    adjust_step_max: float = 3.0  # 小模型单次调整的上限（倍），下限为其倒数
    adjust_total_max: float = 5.0  # 小模型累积调整的上限（倍），下限为其倒数


def relative_arousal(arousal: float, baseline: float) -> float:
    """本片段唤醒度相对长期基线高出多少，归一到 0～1。低于基线记为 0（另由重要性处理）。"""
    if baseline >= 1.0:
        return 0.0
    return min(1.0, max(0.0, (arousal - baseline) / (1.0 - baseline)))


def emotion_factor(rel_arousal: float, k: ForgettingKnobs) -> float:
    """指数形：越往高处放大得越厉害。0 → 1 倍，0.5 → 约 3.2 倍，1 → emotion_max 倍。"""
    return k.emotion_max**rel_arousal


def importance_factor(importance: float, k: ForgettingKnobs) -> float:
    return k.importance_max ** min(1.0, max(0.0, importance))


def clamp_adjustment(total: float, step: float, k: ForgettingKnobs) -> float:
    """把小模型的一次调整并入累积值，单次和累积都限幅。"""
    step = min(k.adjust_step_max, max(1.0 / k.adjust_step_max, step))
    return min(k.adjust_total_max, max(1.0 / k.adjust_total_max, total * step))


def initial_half_life(
    rel_arousal: float,
    importance: float,
    k: ForgettingKnobs,
    *,
    at_edge: bool = False,
    adjustment: float = 1.0,
) -> float:
    """新记忆的半衰期（天）。at_edge：是否为一场对话的首个或末个片段。"""
    position = k.position_bonus if at_edge else 1.0
    return (
        k.base_half_life
        * emotion_factor(rel_arousal, k)
        * importance_factor(importance, k)
        * position
        * adjustment
    )


def strength(half_life: float, days_since_last_recall: float) -> float:
    """当前强度：刚记下或刚被想起时为 1，每过一个半衰期减半。"""
    return 0.5 ** (max(0.0, days_since_last_recall) / half_life)


def days_until_forgotten(half_life: float, k: ForgettingKnobs) -> float:
    """从满强度衰减到遗忘线需要多少天。"""
    return half_life * math.log2(1.0 / k.forget_line)


@dataclass(frozen=True)
class Trace:
    """一条记忆在遗忘曲线上的状态。时间都以"天"计。"""

    half_life: float
    last_recall: float  # 最近一次记下或被想起的时间
    recall_count: int = 0

    def strength_at(self, now: float) -> float:
        return strength(self.half_life, now - self.last_recall)

    def is_forgotten(self, now: float, k: ForgettingKnobs) -> bool:
        return self.strength_at(now) < k.forget_line


def recall(trace: Trace, now: float, k: ForgettingKnobs) -> Trace:
    """被真正用到一次（不是仅仅浮现）：强度回满，半衰期按"忘了多少"加长。

    刚聊过又提（强度接近 1）几乎不加固；快忘光时被想起，加固最多，半衰期最多变为 (1 + reinforce) 倍。
    """
    gain = 1.0 + k.reinforce * (1.0 - trace.strength_at(now))
    return replace(
        trace, half_life=trace.half_life * gain, last_recall=now, recall_count=trace.recall_count + 1
    )
