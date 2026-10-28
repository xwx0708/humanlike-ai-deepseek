"""新鲜记忆（D30）：还没整理的消息按记忆时钟衰减，强度不低于 floor 的稳定在场。"""

import pytest

from contracts.memory import Segment
from memory.clock import ClockKnobs, clock_of
from memory.fresh import FreshKnobs, fresh_block, fresh_memories
from memory.store import MemoryStore

MIN = 60.0


@pytest.fixture
def store():
    s = MemoryStore(":memory:")
    yield s
    s.close()


def chat(store, cid, start, n, step=MIN):
    return [
        store.add_turn(cid, "user" if i % 2 == 0 else "assistant", f"第{i}句", start + i * step)
        for i in range(n)
    ]


def test_recent_turns_are_all_present(store):
    c = store.new_conversation(0)
    chat(store, c.id, 0, 6)
    items = fresh_memories(store, 6 * MIN)
    assert [m.turn.content for m in items] == [f"第{i}句" for i in range(6)]
    assert all(0.9 <= m.strength <= 1.0 for m in items)
    assert items[-1].strength > items[0].strength


def test_old_turns_fade_out_during_long_chat(store):
    """对话中流速 10 倍、半衰期 7 天：强度 0.9 约对应 2.5 小时前说的话。"""
    c = store.new_conversation(0)
    chat(store, c.id, 0, 4 * 60, step=MIN)  # 连续聊 4 小时，每分钟一句
    now = 4 * 60 * MIN
    items = fresh_memories(store, now)
    ages = [(now - m.turn.ts) / 3600 for m in items]
    assert 2.4 < max(ages) < 2.7
    assert all(m.strength >= 0.9 for m in items)


def test_pause_barely_ages_fresh_memory(store):
    """不对话就几乎不遗忘（D21）：停顿一天回来，刚才的话还在。"""
    c = store.new_conversation(0)
    chat(store, c.id, 0, 4)
    assert len(fresh_memories(store, 4 * MIN + 86400)) == 4


def test_segmented_turns_are_no_longer_fresh(store):
    """整理切段后变成普通片段，进入概率海；和浮现区不会重复。"""
    c = store.new_conversation(0)
    turns = chat(store, c.id, 0, 4)
    store.add_segment(Segment(None, c.id, "前两句", turns[0].ts, turns[1].ts, [t.id for t in turns[:2]]))
    assert [m.turn.content for m in fresh_memories(store, 4 * MIN)] == ["第2句", "第3句"]


def test_floor_and_clock_are_knobs(store):
    c = store.new_conversation(0)
    chat(store, c.id, 0, 3, step=10 * MIN)
    slow = clock_of(store, ClockKnobs(speed=1))
    assert len(fresh_memories(store, 20 * MIN, slow, FreshKnobs(floor=0.99999))) == 1


def test_block_text(store):
    c = store.new_conversation(0)
    chat(store, c.id, 0, 2)
    text = fresh_block(fresh_memories(store, 2 * MIN))
    assert text.splitlines() == ["刚才的对话（记得很清楚）：", "对方：第0句", "我：第1句"]
    assert fresh_block([]) == ""
