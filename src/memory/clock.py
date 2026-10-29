"""记忆时钟：遗忘按"记忆时间"算，不按日历时间算（决策 D21，用户 2026-10-28 定）。

- 以对话产生的时间戳为准、以现实时间为标准，但流速比现实快（speed 倍）：
  实际使用是"聊几天、空一阵、再聊几天"，不会持续一整年。
- 不对话就不遗忘：最后一条消息之后（且智能体不在处理用户布置的任务时），
  过 idle_after（20～30 分钟）流速降为 idle_rate（0.01 倍），再过 freeze_after 降为 0，直到再次开始对话。

记忆时间用"天"表示，可以直接放进遗忘曲线 0.5 ^ (经过天数 / 半衰期)。
除 idle_rate 外的数值都是暂定的旋钮。
"""

from bisect import bisect_right
from dataclasses import dataclass

from contracts.memory import SECONDS_PER_DAY
from runtime import config


@dataclass(frozen=True)
class ClockKnobs:
    speed: float = 10.0  # 对话期间的流速（倍）。暂定：每天聊 2 小时左右时，一天的使用约等于一天的记忆时间
    idle_after: float = 25 * 60  # 秒：对话终止多久后开始减速（用户定 20～30 分钟）
    idle_rate: float = 0.01  # 减速后的流速，相对正常流速（用户定）
    freeze_after: float = 24 * 3600  # 秒：减速多久后停止流动（用户说"超过某个时限"，数值暂定）


DEFAULT_CLOCK = config.knobs(ClockKnobs(), "clock")


def clock_of(store, k: ClockKnobs = DEFAULT_CLOCK) -> "MemoryClock":
    """用存储里的全部消息时间和任务时段建一个记忆时钟。"""
    return MemoryClock(store.activity(), k)


class MemoryClock:
    """由"活动区间"（每条消息是一个点，任务是一段）算出任意时刻的记忆时间。"""

    def __init__(self, activity: list[tuple[float, float]], k: ClockKnobs = DEFAULT_CLOCK):
        self.k = k
        spans: list[list[float]] = []
        for s, e in sorted(activity):
            if spans and s <= spans[-1][1]:
                spans[-1][1] = max(spans[-1][1], e)
            else:
                spans.append([s, e])
        self.starts = [s for s, _ in spans]
        self.ends = [e for _, e in spans]
        self.cum = []  # 每个活动区间开始时的记忆时间（秒）
        acc = 0.0
        for i, (s, e) in enumerate(spans):
            if i:
                acc += self._gap(s - self.ends[i - 1])
            self.cum.append(acc)
            acc += (e - s) * k.speed

    def _gap(self, g: float) -> float:
        """活动结束后经过 g 秒，记忆时间走了多少秒。"""
        k = self.k
        full = min(g, k.idle_after)
        slow = min(max(0.0, g - k.idle_after), k.freeze_after) * k.idle_rate
        return (full + slow) * k.speed

    def seconds_at(self, ts: float) -> float:
        i = bisect_right(self.starts, ts) - 1
        if i < 0:
            return 0.0  # 第一次对话之前，时间没有流动
        s, e = self.starts[i], self.ends[i]
        if ts <= e:
            return self.cum[i] + (ts - s) * self.k.speed
        return self.cum[i] + (e - s) * self.k.speed + self._gap(ts - e)

    def elapsed_days(self, t0: float, t1: float) -> float:
        """从 t0 到 t1 经过了多少天记忆时间。"""
        return max(0.0, self.seconds_at(t1) - self.seconds_at(t0)) / SECONDS_PER_DAY
