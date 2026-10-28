"""情绪打分与基线：记忆和情绪衔接的地方（D27）。情绪模块提供时以它为准，没有时用暂行做法。"""

import pytest

from contracts.memory import EmotionScore
from llm import FakeModel
from memory.forgetting import initial_half_life
from memory.store import MemoryStore
from memory.subconscious import consolidate

SEG = {
    "segments": [
        {
            "messages": [1],
            "title": "他妈妈住院了",
            "people": [],
            "arousal": 0.1,
            "valence": 0,
            "importance": 0.1,
            "emotion_reason": "提示词打的分",
        }
    ]
}
NEW = {
    "thread_ids": [],
    "new_thread": True,
    "new_thread_title": "妈妈",
    "related": [],
    "new_thread_ongoing": True,
    "confidence": 1,
    "reason": "",
}


class Scorer:
    def __init__(self):
        self.seen = []

    def score(self, text, title):
        self.seen.append((text, title))
        return EmotionScore(0.9, -0.8, 0.6, "情绪模块打的分")


class Baseline:
    def arousal_baseline(self, ts):
        return 0.5


def test_emotion_module_scores_and_baseline_are_used():
    store = MemoryStore(":memory:")
    c = store.new_conversation(0)
    store.add_turn(c.id, "user", "我妈住院了", 0)
    scorer = Scorer()
    consolidate(store, FakeModel([SEG, NEW]), c.id, 10, scorer=scorer, baseline=Baseline())
    [seg] = store.segments()
    assert scorer.seen == [("我妈住院了", "他妈妈住院了")]
    assert seg.emotion.reason == "情绪模块打的分"
    knobs = store.load_global_knobs()
    # 相对唤醒度 = (0.9 - 0.5) / (1 - 0.5) = 0.8
    assert seg.half_life == pytest.approx(initial_half_life(0.8, 0.6, knobs, at_edge=True))
    store.close()
