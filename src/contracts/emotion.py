"""情绪打分与情绪基线：记忆和情绪衔接的地方（决策 D27，用户 2026-10-28 定）。

情绪模块实现这两个接口、产出分数和基线；记忆模块调用它们，拿分数和基线算半衰期（相对唤醒度 = 先减基线）。
情绪模块做出来之前，记忆模块用自己的暂行做法：正式情绪分由切段提示词一起输出，基线按 0 算
（偏离记录 #1、#2）。接口由总纲维护，改动通过 对外需求.md 提出。
"""

from typing import Protocol

from contracts.memory import EmotionScore


class EmotionScorer(Protocol):
    def score(self, text: str, title: str) -> EmotionScore:
        """对一个确认后的片段按峰终定律打分。text：片段的消息原文；title：一句话标题。"""
        ...


class EmotionBaseline(Protocol):
    def arousal_baseline(self, ts: float) -> float:
        """ts 时刻用户的长期唤醒度基线，0～1。"""
        ...


class ZeroBaseline:
    """暂行：情绪基线还没实现，按 0 计算（偏离记录 #2）。"""

    def arousal_baseline(self, ts: float) -> float:
        return 0.0
