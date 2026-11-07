"""运行期上下文的总预算（项目总评 2026-11-02 第一步）：常驻栏、浮现区、新鲜记忆各有上限。"""

import pytest

from memory.budget import fit_lines, truncate
from memory.fresh import fit_fresh, fresh_memories
from memory.resident import refresh_residents, resident_block
from memory.store import MemoryStore
from memory.tokens import estimate_tokens

MIN = 60.0


@pytest.fixture
def store():
    s = MemoryStore(":memory:")
    yield s
    s.close()


def cost(lines):
    return sum(estimate_tokens(x) + 1 for x in lines)


def test_lines_within_budget_are_untouched():
    lines = ["- 一", "- 二"]
    assert fit_lines(lines, 100) == lines


def test_lines_over_budget_keep_order_and_say_how_many_left():
    lines = [f"- 第{i}条事实写得长一点" for i in range(50)]
    out = fit_lines(lines, 100)
    assert out[:-1] == lines[: len(out) - 1]  # 前面的先放
    assert out[-1] == f"（另有 {50 - len(out) + 1} 条因篇幅没列出）"
    assert cost(out) <= 100


def test_single_line_too_long_is_truncated():
    out = fit_lines(["长" * 500], 50)
    assert out[0].endswith("……") and cost(out) <= 50
    assert truncate("短", 50) == "短"


def test_resident_block_keeps_profile_first_under_budget(store):
    for i in range(40):
        store.add_profile_fact(None, "其他", f"第{i}个稳定事实，写得稍微长一些", 0)
    th = store.new_thread("一件常驻的事", "摘要", 0)
    store.mark_exempt(th.id, "要求记住")
    refresh_residents(store, 0)
    text = resident_block(store, budget=300)
    assert estimate_tokens(text) <= 300 + 5  # 换行按 1 个计，允许几个的误差
    assert "第0个稳定事实" in text and "条因篇幅没列出" in text
    assert resident_block(store, budget=100_000).count("稳定事实") == 40  # 预算够就全列


def test_fresh_keeps_newest_and_gives_way_from_oldest(store):
    c = store.new_conversation(0)
    for i in range(30):
        store.add_turn(c.id, "user", f"第{i}句" + "话" * 40, i * MIN)
    items = fresh_memories(store, 30 * MIN)
    kept = fit_fresh(items, budget=300)
    assert 0 < len(kept) < len(items)
    assert kept[-1].turn.content.startswith("第29句")  # 最近的一定在
    assert [m.turn.id for m in kept] == [m.turn.id for m in items[-len(kept) :]]  # 让出的是最早的


def test_fresh_single_huge_message_is_truncated(store):
    c = store.new_conversation(0)
    store.add_turn(c.id, "user", "长" * 5000, 0)
    [m] = fit_fresh(fresh_memories(store, MIN), budget=200)
    assert m.turn.content.endswith("……") and estimate_tokens(m.turn.content) < 200
    assert store.all_turns()[0].content == "长" * 5000  # 只截给模型看的，原文不动
