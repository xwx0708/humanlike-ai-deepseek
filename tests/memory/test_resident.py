"""常驻栏（决策 D17）：免竞争、加权竞争、进出、不遗忘、给模型的文字。"""

import pytest

from contracts.memory import EmotionScore, Segment
from memory.resident import (
    RESIDENT_HEADER,
    ResidentKnobs,
    effective_strength,
    refresh_residents,
    resident_block,
    score,
)
from memory.store import MemoryStore

DAY = 86400.0


@pytest.fixture
def store():
    s = MemoryStore(":memory:")
    yield s
    s.close()


def thread(store, title, ts, importance=0.2, arousal=0.2, conversations=1, remember=False, ongoing=True):
    """建一条事件线：在 conversations 场不同的对话里各有一个片段。"""
    th = store.new_thread(title, title, ts)
    segs = []
    for i in range(conversations):
        c = store.new_conversation(ts + i)
        t = store.add_turn(c.id, "user", title, ts + i)
        seg = store.add_segment(
            Segment(
                None,
                c.id,
                title,
                ts + i,
                ts + i,
                [t.id],
                [],
                EmotionScore(arousal, 0, importance),
                half_life=7.0,
                last_recall_ts=ts + i,
                remember_requested=remember,
            )
        )
        store.attach(th.id, seg.id, 1.0, ts + i)
        segs.append(seg)
    store.update_thread(th.id, ongoing=ongoing)
    return th.id, segs


def stats(**kw):
    base = {
        "importance": 0,
        "arousal": 0,
        "recalls": 0,
        "conversations": 1,
        "remember_requested": False,
        "updated_ts": 0,
        "ongoing": False,
    }
    return {**base, **kw}


def test_score_weights_and_ongoing_factor():
    assert score(stats(importance=1), 0) == pytest.approx(0.35 + 0.25)  # 刚被提及：活跃度 1
    assert score(stats(arousal=1), 0) == pytest.approx(0.25 + 0.25)
    assert score(stats(recalls=3), 0) == pytest.approx(0.25 + 0.15 * 0.5)
    assert score(stats(), 14 * DAY) == pytest.approx(0.25 * 0.5)  # 14 天没提，活跃度减半
    assert score(stats(importance=1, ongoing=True), 0) == pytest.approx((0.35 + 0.25) * 1.5)


def test_three_kinds_are_exempt_from_competition(store):
    profiled, [seg] = thread(store, "花生过敏", 0)
    store.add_profile_fact(None, "过敏与禁忌", "对花生过敏", 0, seg.id)
    asked, _ = thread(store, "妹妹的生日礼物", 0, remember=True)
    repeated, _ = thread(store, "考研", 0, conversations=3)
    plain, _ = thread(store, "午饭吃了面", 0)
    rep = refresh_residents(store, 0, ResidentKnobs(seats=0))  # 竞争名额为 0：只有免竞争的能常驻
    assert rep.exempt == {profiled: "涉及用户真实信息", asked: "用户要求记住", repeated: "多次反复提及"}
    assert (
        set(store.resident_threads()) == {profiled, asked, repeated} and plain not in store.resident_threads()
    )


def test_competition_takes_top_seats_and_demotes_losers(store):
    k = ResidentKnobs(seats=2)
    _low, _ = thread(store, "闲聊", 0, importance=0.1, arousal=0.1, ongoing=False)
    mid, _ = thread(store, "周末爬山", 0, importance=0.3, arousal=0.4, ongoing=False)
    high, [_hseg] = thread(store, "妈妈住院", 0, importance=0.8, arousal=0.8, ongoing=False)
    refresh_residents(store, 0, k)
    assert set(store.resident_threads()) == {mid, high}

    hot, _ = thread(store, "工作去留", 1, importance=0.5, arousal=0.6, ongoing=True)  # 进行中，乘系数后挤进来
    rep = refresh_residents(store, 1, k)
    assert rep.entered == [hot] and rep.left == [mid]
    assert set(store.resident_threads()) == {high, hot}
    log = store.resident_log()
    assert [(e["thread_id"], e["action"]) for e in log][-2:] == [(hot, "进入"), (mid, "退出")]
    assert "名额竞争落选" in log[-1]["reason"]


def test_resident_is_not_forgotten_and_demotion_does_not_reset(store):
    k = ResidentKnobs(seats=1)
    tid, [seg] = thread(store, "妈妈住院", 0, importance=0.9, arousal=0.9)
    refresh_residents(store, 0, k)
    assert effective_strength(store, seg, 60 * DAY) == 1.0  # 常驻期间不遗忘

    _newer, _ = thread(store, "更重要的事", 60 * DAY, importance=1, arousal=1)
    refresh_residents(store, 60 * DAY, k)
    assert tid not in store.resident_threads()
    # 退回普通记忆池：遗忘从最近一次被提及（第 0 天）开始算，60 天、半衰期 7 天，早已低于遗忘线
    assert effective_strength(store, store.get_segment(seg.id), 60 * DAY) == pytest.approx(0.5 ** (60 / 7))
    assert store.get_segment(seg.id).last_recall_ts == 0  # 退回时不重置


def test_resident_block_for_prompts(store):
    tid, [seg] = thread(store, "花生过敏", 0)
    mom = store.upsert_person("妈妈", ["我妈"])
    store.add_profile_fact(None, "过敏与禁忌", "对花生过敏", 0, seg.id)
    store.add_profile_fact(mom, "长期健康", "有哮喘", 0)
    refresh_residents(store, 0)
    text = resident_block(store)
    assert text.startswith(RESIDENT_HEADER) and "不必刻意提起" in text
    assert "- 他：对花生过敏" in text and "- 妈妈：有哮喘" in text and f"[{tid}] 花生过敏" in text
