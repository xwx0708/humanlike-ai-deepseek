"""记忆时钟（D21）、事件完结后才允许缩短（D19）、批量遗忘（D22）。"""

import random

import pytest

from contracts.memory import EmotionScore, Segment
from llm import FakeModel
from memory.clock import ClockKnobs, MemoryClock
from memory.prompts import Adjustment, parse_keywords
from memory.store import MemoryStore
from memory.subconscious import apply_adjustments
from memory.sweep import SweepKnobs, forgotten, sweep

DAY = 86400.0
MIN = 60.0


@pytest.fixture
def store():
    s = MemoryStore(":memory:")
    yield s
    s.close()


# ---------- D21 记忆时钟 ----------


def test_clock_runs_fast_while_talking():
    c = MemoryClock([(0, 0), (10 * MIN, 10 * MIN)], ClockKnobs(speed=10))
    assert c.elapsed_days(0, 10 * MIN) == pytest.approx(100 * MIN / DAY)  # 聊 10 分钟 = 记忆里 100 分钟


def test_clock_slows_after_pause_and_freezes_later():
    k = ClockKnobs(speed=10, idle_after=25 * MIN, idle_rate=0.01, freeze_after=24 * 3600)
    c = MemoryClock([(0, 0)], k)
    full = 25 * MIN * 10
    assert c.elapsed_days(0, 25 * MIN) == pytest.approx(full / DAY)
    assert c.elapsed_days(0, 25 * MIN + 3600) == pytest.approx((full + 3600 * 0.01 * 10) / DAY)  # 减速到 0.01
    frozen = c.elapsed_days(0, 30 * DAY)
    assert frozen == pytest.approx((full + 24 * 3600 * 0.01 * 10) / DAY)
    assert c.elapsed_days(0, 300 * DAY) == pytest.approx(frozen)  # 不再对话，就不再遗忘


def test_clock_resumes_when_talk_resumes_and_tasks_count_as_activity():
    k = ClockKnobs(speed=10, idle_after=25 * MIN, idle_rate=0.01, freeze_after=3600)
    c = MemoryClock([(0, 0), (30 * DAY, 30 * DAY)], k)
    before = c.elapsed_days(0, 30 * DAY)
    assert c.elapsed_days(0, 30 * DAY + 5 * MIN) == pytest.approx(before + 50 * MIN / DAY)
    busy = MemoryClock([(0, 0), (0, 2 * 3600)], k)  # 智能体在处理用户布置的任务，没有消息也照常流动
    assert busy.elapsed_days(0, 2 * 3600) == pytest.approx(2 * 3600 * 10 / DAY)


def test_store_activity_includes_turns_and_busy_spans(store):
    c = store.new_conversation(0)
    store.add_turn(c.id, "user", "帮我查个东西", 0)
    store.add_busy_span(10, 500, "查资料")
    assert sorted(store.activity()) == [(0, 0), (10, 500)]


# ---------- D19 事件完结后才允许系数小于 1 ----------


def _seg_on_thread(store, ongoing):
    c = store.new_conversation(0)
    t = store.add_turn(c.id, "user", "取消周末爬山", 0)
    seg = store.add_segment(Segment(None, c.id, "取消爬山", 0, 0, [t.id], [], EmotionScore(0.2, 0, 0.2)))
    th = store.new_thread("妈妈的健康", "住院", 0)
    store.attach(th.id, seg.id, 1.0, 0)
    store.update_thread(th.id, ongoing=ongoing)
    return seg


def test_shrink_refused_while_event_ongoing(store):
    seg = _seg_on_thread(store, ongoing=True)
    [(_, _, ch)] = apply_adjustments(store, [Adjustment("memory", str(seg.id), 0.5, "已解决")], 1)
    assert not ch.applied and ch.note == "事件未完结，不允许系数小于 1"
    assert store.get_segment(seg.id).adjustment == 1.0
    [(_, _, ch)] = apply_adjustments(store, [Adjustment("memory", str(seg.id), 2, "变重要")], 1)
    assert ch.applied  # 拉长不受限
    [(_, _, ch)] = apply_adjustments(store, [Adjustment("memory", str(seg.id), 0.25, "又不重要了")], 1)
    assert ch.applied and store.get_segment(seg.id).adjustment == 1.0  # 可以往回调，但最多调回 1


def test_shrink_allowed_after_event_finished(store):
    seg = _seg_on_thread(store, ongoing=False)
    [(_, _, ch)] = apply_adjustments(store, [Adjustment("memory", str(seg.id), 0.5, "已解决")], 1)
    assert ch.applied and store.get_segment(seg.id).adjustment == 0.5


# ---------- D22 批量遗忘 ----------


def _many_old_segments(store, n):
    c = store.new_conversation(0)
    ids = []
    for i in range(n):
        t = store.add_turn(c.id, "user", f"闲聊第 {i} 件小事", i)
        imp = 0.9 if i == 7 else 0.1  # 第 7 条最重要；留下哪条是随机的，与重要性无关
        seg = store.add_segment(
            Segment(None, c.id, f"小事 {i}", i, i, [t.id], [], EmotionScore(0.1, 0, imp), half_life=0.001)
        )
        ids.append(seg.id)
    return ids


def test_nothing_happens_until_enough_are_forgotten(store):
    _many_old_segments(store, 5)
    assert len(forgotten(store, 1 * DAY)) == 5
    assert sweep(store, FakeModel([]), 1 * DAY, k=SweepKnobs(batch_size=20)) is None


def test_sweep_archives_95_percent_and_keeps_random_5_percent(store):
    ids = _many_old_segments(store, 20)
    rng = random.Random(1)
    pick = random.Random(1).sample(ids, 1)[0]  # 用同样的种子算出会被随机留下的那条
    fake = FakeModel([{"items": [{"id": pick, "keywords": ["小事", "第七件", "很重要的一件事情"]}]}])
    rep = sweep(store, fake, 1 * DAY, rng=rng)
    assert list(rep.kept) == [pick] and rep.kept[pick] == ["小事", "第七件", "很重要的一件"]
    assert sorted(rep.archived) == sorted(set(ids) - {pick})  # 19 / 20 = 95%
    kept = store.get_segment(pick)
    assert kept.state == "abstract" and kept.trace == "小事、第七件、很重要的一件"
    assert kept.last_recall_ts == 1 * DAY  # 重新开始一条遗忘曲线
    assert all(store.get_segment(i).state == "archived" for i in rep.archived)
    assert store.sweep_log()[0]["archived"] == rep.archived
    assert forgotten(store, 1 * DAY) == []  # 移进档案的不再出现在记忆系统里


def test_kept_5_percent_is_random_not_by_importance():
    picks = set()
    for seed in range(30):
        s = MemoryStore(":memory:")
        ids = _many_old_segments(s, 20)
        rep = sweep(s, FakeModel([{"items": []}]), 1 * DAY, rng=random.Random(seed))
        picks.add(ids.index(next(iter(rep.kept))))
        s.close()
    assert len(picks) > 5  # 不总是留重要性最高的第 7 条


def test_sweep_takes_fixed_size_batches_in_order_of_forgetting(store):
    """攒了 45 条：分两批、每批 20 条，按到达遗忘线的先后；余下 5 条继续攒。"""
    ids = _many_old_segments(store, 45)
    # 片段 i 在第 i 秒记下、半衰期相同，所以越早记下越早到达遗忘线
    fake = FakeModel([{"items": []}, {"items": []}])
    rep = sweep(store, fake, 1 * DAY)
    assert len(store.sweep_log()) == 2 and len(rep.kept) == 2 and len(rep.archived) == 38
    assert {i for log in store.sweep_log() for i in log["archived"]} | {int(i) for i in rep.kept} == set(
        ids[:40]
    )
    assert [s.id for s in forgotten(store, 1 * DAY)] == ids[40:]  # 余下的 5 条继续攒
    assert all(store.get_segment(i).trace for i in rep.kept)  # 模型没给关键词时用标题兜底


def test_parse_keywords_limits_and_drops_unknown_ids():
    data = {"items": [{"id": 1, "keywords": ["a", "a", "b", "c", "d", "e"]}, {"id": 9, "keywords": ["x"]}]}
    assert parse_keywords(data, [1, 2]) == {1: ["a", "b", "c", "d"]}
