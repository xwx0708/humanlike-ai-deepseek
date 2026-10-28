"""切段第一层规则：分对话、语言信号、跨天、长度和时长上限。"""

import time

import pytest

from memory.boundaries import BoundaryRules, conversation_closed, ingest_turn, language_signal
from memory.store import MemoryStore

MIN = 60.0


@pytest.fixture
def store():
    s = MemoryStore(":memory:")
    yield s
    s.close()


@pytest.mark.parametrize(
    ("text", "hit"),
    [
        ("对了，周末爬山吗", "对了"),
        ("……话说你最近怎么样", "话说"),
        ("算了不聊这个了", "不聊这个了"),
        ("说回刚才那个工作的事", "说回"),
        ("我妈住院了", None),
        ("这样对了吗", None),  # "对了"不在句首，不算
    ],
)
def test_language_signal(text, hit):
    assert language_signal(text) == hit


def test_gap_starts_new_conversation(store):
    a = ingest_turn(store, "user", "早", 0)
    b = ingest_turn(store, "assistant", "早呀", 5 * MIN)
    c = ingest_turn(store, "user", "我回来了", 5 * MIN + 31 * MIN)
    assert a.conversation_id == b.conversation_id != c.conversation_id
    assert store.boundaries(c.conversation_id) == {}  # 新对话的第一条不用标


def test_language_signal_only_from_user(store):
    ingest_turn(store, "user", "我妈住院了", 0)
    ai = ingest_turn(store, "assistant", "对了，你吃饭了吗", MIN)
    user = ingest_turn(store, "user", "对了，周末爬山取消了", 2 * MIN)
    marks = store.boundaries(user.conversation_id)
    assert ai.id not in marks and marks[user.id] == ["语言信号：对了"]


def test_cross_midnight_is_candidate_not_new_conversation(store):
    late = time.mktime((2026, 10, 27, 23, 55, 0, 0, 0, -1))
    a = ingest_turn(store, "user", "还没睡", late)
    b = ingest_turn(store, "user", "过零点了", late + 10 * MIN)
    assert a.conversation_id == b.conversation_id
    assert store.boundaries(b.conversation_id) == {b.id: ["跨天"]}


def test_length_limit_counts_from_last_boundary(store):
    rules = BoundaryRules(max_turns=3, max_minutes=999)
    ts = [ingest_turn(store, "user", f"第{i}句", i * MIN, rules) for i in range(7)]
    marks = store.boundaries(ts[0].conversation_id)
    assert sorted(marks) == [ts[3].id, ts[6].id]  # 每满 3 条标一次，标过之后重新计数
    assert marks[ts[3].id] == ["长度上限"]


def test_duration_limit(store):
    rules = BoundaryRules(max_turns=999, max_minutes=20, conversation_gap_minutes=30)
    a = ingest_turn(store, "user", "a", 0, rules)
    ingest_turn(store, "user", "b", 10 * MIN, rules)
    c = ingest_turn(store, "user", "c", 25 * MIN, rules)
    assert store.boundaries(a.conversation_id) == {c.id: ["时长上限"]}


def test_conversation_closed(store):
    a = ingest_turn(store, "user", "a", 0)
    assert not conversation_closed(store, a.conversation_id, 10 * MIN)
    assert conversation_closed(store, a.conversation_id, 40 * MIN)  # 安静超过 30 分钟
    ingest_turn(store, "user", "b", 50 * MIN)  # 新对话开始
    assert conversation_closed(store, a.conversation_id, 50 * MIN)
