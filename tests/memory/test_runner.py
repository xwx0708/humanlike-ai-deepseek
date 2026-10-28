"""后台调度 tick()，以及一场对话的摘要随遗忘重新概括（待办 #7）。"""

import time

import pytest

from contracts.memory import EmotionScore, Segment
from llm import FakeModel
from memory.lifecycle import RUNNING
from memory.runner import tick
from memory.store import MemoryStore
from memory.subconscious import summarize_conversation

MIN = 60.0


@pytest.fixture
def store():
    s = MemoryStore(":memory:")
    yield s
    s.close()


SEG = {
    "segments": [
        {
            "messages": [1],
            "title": "他说今天加班",
            "people": [],
            "arousal": 0.2,
            "valence": -0.1,
            "importance": 0.1,
            "emotion_reason": "",
        }
    ]
}
NEW = {
    "thread_ids": [],
    "new_thread": True,
    "new_thread_title": "加班",
    "related": [],
    "new_thread_ongoing": False,
    "confidence": 1,
    "reason": "",
}


def _noon(day=0):
    return time.mktime((2026, 10, 20 + day, 12, 0, 0, 0, 0, -1))


def test_familiar_phase_only_checks_whether_to_start(store):
    c = store.new_conversation(_noon())
    store.add_turn(c.id, "user", "今天加班", _noon())
    r = tick(store, FakeModel([]), _noon() + 60 * MIN, context_limit=100_000)
    assert r.started is None and r.consolidated == {}


def test_running_phase_consolidates_after_pause_and_merges_at_night(store):
    store.set_meta("phase", RUNNING)
    store.set_meta("last_merge_day", time.strftime("%Y-%m-%d", time.localtime(_noon())))
    c = store.new_conversation(_noon())
    store.add_turn(c.id, "user", "今天加班", _noon())
    assert tick(store, FakeModel([]), _noon() + 10 * MIN, 100_000).consolidated == {}  # 还在聊，不整理
    fake = FakeModel([SEG, NEW])
    r = tick(store, fake, _noon() + 30 * MIN, 100_000)
    assert list(r.consolidated) == [c.id] and r.merged is None  # 白天不合并
    assert store.get_conversation(c.id).summary == ""  # 停顿了，但对话还没结束（30 分钟）
    r = tick(store, FakeModel([{"summary": "那天他说在加班"}]), _noon() + 40 * MIN, 100_000)
    assert store.get_conversation(c.id).summary == "那天他说在加班"
    r = tick(store, FakeModel([]), _noon(1), 100_000)  # 第二天中午（已过凌晨 3 点）：夜间合并一次
    assert r.merged is not None
    assert tick(store, FakeModel([]), _noon(1) + 60, 100_000).merged is None  # 同一天只合并一次


def test_conversation_summary_gets_vaguer_as_segments_are_forgotten(store):
    c = store.new_conversation(_noon())
    turns = [store.add_turn(c.id, "user", t, _noon()) for t in ("妈妈住院", "想换工作")]
    a = store.add_segment(
        Segment(None, c.id, "他妈妈住院", 0, 0, [turns[0].id], [], EmotionScore(0.5, -0.5, 0.5))
    )
    b = store.add_segment(
        Segment(None, c.id, "他想换工作", 0, 0, [turns[1].id], [], EmotionScore(0.3, 0, 0.3))
    )
    a.state, a.trace = "abstract", "妈妈、住院"
    b.state = "archived"
    store.save_segment_state(a)
    store.save_segment_state(b)
    fake = FakeModel([{"summary": "那天聊到他妈妈，好像在医院"}])
    summarize_conversation(store, fake, c.id)
    prompt = fake.calls[0]["messages"][1].content
    assert "（模糊）妈妈、住院" in prompt and "换工作" not in prompt  # 移进档案的不再出现
    assert store.get_conversation(c.id).summary == "那天聊到他妈妈，好像在医院"
