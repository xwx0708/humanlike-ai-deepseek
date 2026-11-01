"""可替换点（《模块化改进建议》改进三，D42）：只有能说出两个真实替代实现的地方才做成接口。

用户自定义：把一个 Python 文件放进数据文件夹的 plugins/ 里，文件里定义实现下面某个接口的类，
类上写 `plugin_kind = "decay"`（或 "surface"）和 `name = "……"`；然后在设置页（或 settings.json）
按名字选用。加载方式见 runtime.plugins。

不开放的：SQLite 存储、事件线和片段的数据结构、潜意识整理流程——它们是项目的本体。
"""

import random
from dataclasses import dataclass
from typing import Any, Protocol

from contracts.memory import Segment


class DecayModel(Protocol):
    """遗忘曲线：强度怎么随时间衰减、被想起后半衰期怎么变。时间以"天"计（记忆时钟量出来的，D21）。

    内置：指数衰减（默认，S = 0.5^(Δt/H)）、幂律衰减、不遗忘（调试用）。
    """

    name: str

    def strength(self, half_life: float, elapsed_days: float) -> float:
        """刚记下或刚被想起时为 1，之后随 elapsed_days 下降。half_life 是"降到一半"的天数。"""

    def reinforced_half_life(self, half_life: float, strength_now: float, reinforce: float) -> float:
        """被真正用到一次后的新半衰期（D20）。strength_now 是被想起时的强度，reinforce 是巩固强度旋钮。"""


@dataclass(frozen=True)
class Candidate:
    """扫描器这一格里看到的一个片段。"""

    segment: Segment
    strength: float  # 当前强度（已经不低于遗忘线）
    priming: float  # 启动加成：和当下话题有多相关（0～priming_max）
    minutes: float  # 距上次扫描到它经过了多少分钟


class SurfacePolicy(Protocol):
    """浮现策略：这一格扫描到的片段里，哪些冒头进浮现区。

    内置：概率浮现（默认，设想文档的伯努利采样）、纯相关度检索（只按和话题的相关程度取前几条）、关闭浮现。
    """

    name: str

    def surface(
        self, candidates: list[Candidate], temperature: float, knobs: Any, rng: random.Random
    ) -> list[Segment]:
        """knobs 是扫描器的参数（memory.scanner.ScannerKnobs：rate、threshold、buffer_size、negative_damp……）。"""
