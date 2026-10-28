"""夜间合并事件线（待办 #8）：用假模型离线测试；最后一个 live 测试真实调用本地 Qwen，默认跳过。"""

import pytest

from contracts.memory import EmotionScore, Segment
from llm import FakeModel
from memory.merge import (
    MERGE_SCHEMA,
    build_merge_prompt,
    find_duplicates,
    merge_log,
    merge_recent_threads,
    parse_merge,
    recent_threads,
)
from memory.store import MemoryStore

DAY = 86400.0
NOW = 100 * DAY
SUMMARY = {"summary": "他妈妈先住院，后来查出是肺炎", "new_title": "", "ongoing": True}


def make_thread(
    store: MemoryStore, title: str, text: str, people: list[tuple[str, list[str]]], created: float
):
    """建一场对话、一个片段、一条事件线，并把片段挂上去。"""
    c = store.new_conversation(created)
    turn = store.add_turn(c.id, "user", text, created)
    pids = [store.upsert_person(n, a) for n, a in people]
    seg = store.add_segment(
        Segment(
            None,
            c.id,
            text,
            created,
            created,
            [turn.id],
            pids,
            EmotionScore(0.5, -0.5, 0.6, ""),
            half_life=10.0,
            last_recall_ts=created,
        )
    )
    t = store.new_thread(title, text, created)
    store.attach(t.id, seg.id, 1.0, created)
    return t.id, seg.id


@pytest.fixture
def store():
    s = MemoryStore(":memory:")
    yield s
    s.close()


def two_mom_threads(store):
    old, old_seg = make_thread(store, "妈妈住院", "他妈妈住院了", [("妈妈", ["我妈"])], NOW - 20 * DAY)
    new, new_seg = make_thread(store, "母亲的肺炎", "他母亲查出肺炎", [("母亲", ["我妈"])], NOW - 1 * DAY)
    return old, old_seg, new, new_seg


def test_recent_threads_and_candidates(store):
    old, _, new, _ = two_mom_threads(store)
    other, _ = make_thread(store, "周末爬山", "他周末想去爬山", [("小王", [])], NOW - 2 * DAY)
    assert recent_threads(store, NOW) == [other, new]  # 只取 7 天内新建的，从早到晚
    hits = find_duplicates(store, new)
    assert old in hits and "人物重合" in hits[old]  # 别名表把"母亲""妈妈"归成同一个人
    assert other not in hits


def test_merge_moves_segments_and_inherits(store):
    old, old_seg, new, new_seg = two_mom_threads(store)
    side, _ = make_thread(store, "老板不批假", "老板不批他的假", [("老板", [])], NOW - 30 * DAY)
    store.relate_threads(new, side, "因为妈妈住院请假")
    store.mark_exempt(new, "用户要求记住")
    store.set_resident(new, NOW - DAY, True, "免竞争：用户要求记住")

    model = FakeModel([{"same": [{"id": old, "confidence": 0.9, "reason": "同一场病"}]}, SUMMARY])
    report = merge_recent_threads(store, model, NOW)

    assert report.merged == [(old, new, 0.9)]
    assert model.calls[0]["schema"] is MERGE_SCHEMA
    kept, dropped = store.get_thread(old), store.get_thread(new)
    assert dropped.merged_into == old and dropped.segment_ids == []
    assert kept.segment_ids == [old_seg, new_seg]
    assert kept.summary == SUMMARY["summary"]  # 合并后重新概括
    assert store.exempt_marks()[old] == "用户要求记住"  # 免竞争标记继承
    assert old in store.resident_threads() and new not in store.resident_threads()  # 常驻状态继承
    links = store.db.execute("SELECT a, b FROM thread_links").fetchall()
    assert [tuple(r) for r in links] == [(min(old, side), max(old, side))]  # 关联改指保留线
    [log] = merge_log(store)
    assert (log["keep_id"], log["drop_id"], log["merged"]) == (old, new, 1)
    assert log["reason"] == "同一场病" and log["drop_title"] == "母亲的肺炎"


def test_low_confidence_does_not_merge(store):
    old, _, new, _ = two_mom_threads(store)
    model = FakeModel([{"same": [{"id": old, "confidence": 0.7, "reason": "可能是同一件事"}]}])
    report = merge_recent_threads(store, model, NOW)
    assert report.merged == [] and report.rejected == [(old, new, 0.7)]
    assert store.get_thread(new).merged_into is None
    [log] = merge_log(store)
    assert log["merged"] == 0 and "把握低于 0.8" in log["note"]  # 没合并的也记，便于校准阈值


def test_unknown_ids_are_dropped(store):
    _, _, new, _ = two_mom_threads(store)
    reply = {
        "same": [
            {"id": 999, "confidence": 1.0, "reason": "编的"},
            {"id": new, "confidence": 1.0, "reason": "自己"},
            {"id": "x", "confidence": 1.0, "reason": "不是数"},
        ]
    }
    report = merge_recent_threads(store, FakeModel([reply]), NOW)
    assert report.merged == [] and merge_log(store) == []
    assert store.get_thread(new).merged_into is None


def test_parse_merge_validates():
    data = {
        "same": [
            {"id": 3, "confidence": 1.7, "reason": " 同一件事 "},
            {"id": 3, "confidence": 0.1, "reason": "重复"},
            {"id": 5, "confidence": "高", "reason": ""},
            {"id": 9, "confidence": 0.9, "reason": "不在候选"},
            {"id": True, "confidence": 0.9, "reason": "布尔"},
            "乱写",
        ]
    }
    out = parse_merge(data, [3, 5])
    assert [(j.other_id, j.confidence, j.reason) for j in out] == [(3, 1.0, "同一件事"), (5, 0.0, "")]
    assert parse_merge({}, [3]) == []


def test_merge_prompt_shows_cards():
    card = {"id": 1, "title": "妈妈住院", "summary": "s", "segments": ["他妈妈住院了"], "people": ["妈妈"]}
    body = build_merge_prompt(card, [{**card, "id": 2, "title": "母亲的肺炎"}])[1].content
    assert "[1] 妈妈住院" in body and "[2] 母亲的肺炎" in body and "人物：妈妈" in body


def test_no_candidates_no_model_call(store):
    make_thread(store, "周末爬山", "他周末想去爬山", [("小王", [])], NOW - DAY)
    model = FakeModel([])
    report = merge_recent_threads(store, model, NOW)
    assert model.calls == [] and report.merged == []


@pytest.mark.live
def test_merge_with_real_qwen():
    """真实调用本地 Qwen：两条重复的线被合并，无关的线不被合并。"""
    from llm import subconscious_model

    store, qwen = MemoryStore(":memory:"), subconscious_model()
    mom, _ = make_thread(store, "妈妈住院", "我妈住院了，我很担心", [("妈妈", ["我妈"])], NOW - 3 * DAY)
    pneu, _ = make_thread(
        store, "母亲的肺炎", "母亲查出来是肺炎，医生说要住一周", [("妈妈", ["母亲"])], NOW - DAY
    )
    # 和妈妈住院共用"住院"这个词（会成为关键词候选），但是另一个人的另一件事，不该合并
    dad, _ = make_thread(
        store, "爸爸做手术", "我爸也住院了，做了个阑尾手术", [("爸爸", ["我爸"])], NOW - 2 * DAY
    )
    hike, _ = make_thread(store, "周末爬山", "周末和同事小王去爬山", [("小王", [])], NOW - 2 * DAY)
    report = merge_recent_threads(store, qwen, NOW)
    print(report, merge_log(store))
    assert store.get_thread(pneu).merged_into == mom
    assert any(
        dad in hits or (t == dad and hits) for t, hits in report.candidate_sources.items()
    )  # 无关线确实被问过
    for t in (mom, dad, hike):
        assert store.get_thread(t).merged_into is None


def test_merged_exempt_keeps_higher_priority():
    """D26：名次保持优先——只要有一条免竞争，合并后仍免竞争。"""
    from memory.merge import _merged_exempt

    assert _merged_exempt("用户要求记住", "") == "用户要求记住"
    assert _merged_exempt("", "涉及用户真实信息") == "涉及用户真实信息"
    assert _merged_exempt("", None) == ""
    assert _merged_exempt(None, None) is None
