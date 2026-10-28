"""画像矛盾处理（决策 D16）：牢固度、三选一的概率、唤起与覆盖。"""

import random

import pytest

from contracts.memory import ProfileFact
from llm import FakeModel
from memory.profile import EVOKE, NOTHING, OVERWRITE, firmness, outcome_probabilities, resolve_conflict
from memory.store import MemoryStore
from memory.subconscious import consolidate

DAY = 86400.0


def fact(mentions=1, last_ts=0.0):
    return ProfileFact(1, None, "职业与学业", "在杭州工作", 0.0, last_ts, mentions)


def test_firmness_grows_with_mentions_and_fades_with_time():
    assert firmness(fact(1), 0) == pytest.approx(0.5)
    assert firmness(fact(3), 0) == pytest.approx(0.875)
    assert firmness(fact(1), 180 * DAY) == pytest.approx(0.25)  # 180 天没提，减半


@pytest.mark.parametrize(("firm", "evoke", "overwrite"), [(0.5, 0.5, 0.4), (1.0, 0.8, 0.0), (0.0, 0.2, 0.8)])
def test_probabilities(firm, evoke, overwrite):
    p = outcome_probabilities(firm)
    assert p[EVOKE] == pytest.approx(evoke) and p[OVERWRITE] == pytest.approx(overwrite)
    assert sum(p.values()) == pytest.approx(1)


def test_firm_facts_are_evoked_more_and_overwritten_less():
    rng = random.Random(42)
    firm = [resolve_conflict(fact(5), 0, rng).outcome for _ in range(10000)]
    weak = [resolve_conflict(fact(1), 365 * DAY, rng).outcome for _ in range(10000)]
    assert firm.count(EVOKE) > weak.count(EVOKE) and firm.count(OVERWRITE) < weak.count(OVERWRITE)
    # 频率接近概率：牢固度 0.97 时唤起约 78%
    assert abs(firm.count(EVOKE) / 10000 - outcome_probabilities(firmness(fact(5), 0))[EVOKE]) < 0.02


class FixedRoll(random.Random):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def random(self):
        return self.value


def _segment_reply(fact_text):
    return {
        "segments": [
            {
                "messages": [1],
                "title": f"他说{fact_text}",
                "people": [],
                "arousal": 0.1,
                "valence": 0,
                "importance": 0.8,
                "emotion_reason": "",
                "facts": [{"about": "用户", "category": "职业与学业", "fact": fact_text}],
            }
        ]
    }


NEW_THREAD = {
    "thread_ids": [],
    "new_thread": True,
    "new_thread_title": "工作",
    "related": [],
    "new_thread_ongoing": True,
    "confidence": 1,
    "reason": "",
}


@pytest.mark.parametrize(
    ("roll", "outcome"),
    [(0.1, EVOKE), (0.7, OVERWRITE), (0.99, NOTHING)],  # 牢固度 0.5：唤起 50%、覆盖 40%、都不 10%
)
def test_conflict_flow(roll, outcome):
    store = MemoryStore(":memory:")
    c1 = store.new_conversation(0)
    store.add_turn(c1.id, "user", "我在杭州工作", 0)
    consolidate(store, FakeModel([_segment_reply("在杭州工作"), NEW_THREAD]), c1.id, 10)
    [old] = store.profile()

    c2 = store.new_conversation(100)
    store.add_turn(c2.id, "user", "我搬到上海工作了", 100)
    fake = FakeModel(
        [
            _segment_reply("在上海工作"),
            {"conflicts": [old.id, 999], "reason": "工作城市不同"},  # 999 不存在，会被丢掉
            {
                "thread_ids": [],
                "new_thread": True,
                "new_thread_title": "工作",
                "related": [],
                "new_thread_ongoing": True,
                "confidence": 1,
                "reason": "",
            },
        ]
    )
    report = consolidate(store, fake, c2.id, 100, rng=FixedRoll(roll))
    new = next(f for f in store.profile(everyone=True) if f.fact == "在上海工作")
    assert report.conflicts == [(old.id, new.id, outcome)]
    [log] = store.profile_conflicts()
    assert (log["outcome"], log["roll"], log["reason"]) == (outcome, roll, "工作城市不同")

    active = {f.fact for f in store.profile()}
    if outcome == OVERWRITE:
        assert active == {"在上海工作"}
        [archived] = store.profile(archived=True)
        assert archived.fact == "在杭州工作" and archived.superseded_by == new.id  # 进了档案，可以找回
        assert store.pending_evocations() == []
    elif outcome == EVOKE:
        assert active == {"在杭州工作", "在上海工作"}
        [ev] = store.pending_evocations()
        assert (ev["fact_id"], ev["trigger_fact_id"]) == (old.id, new.id)
    else:
        assert active == {"在杭州工作", "在上海工作"} and store.pending_evocations() == []
    store.close()


def test_no_conflict_check_when_no_old_fact_or_repeat():
    store = MemoryStore(":memory:")
    c = store.new_conversation(0)
    store.add_turn(c.id, "user", "我在杭州工作", 0)
    fake = FakeModel([_segment_reply("在杭州工作"), NEW_THREAD])
    consolidate(store, fake, c.id, 10)
    c2 = store.new_conversation(100)
    store.add_turn(c2.id, "user", "我在杭州工作哦", 100)
    fake2 = FakeModel(
        [
            _segment_reply("在杭州工作。"),
            {
                "thread_ids": [],
                "new_thread": True,
                "new_thread_title": "工作",
                "related": [],
                "new_thread_ongoing": True,
                "confidence": 1,
                "reason": "",
            },
        ]
    )
    consolidate(store, fake2, c2.id, 110)  # 重复的事实只合并，不检查矛盾（否则假模型的回答会不够用）
    assert store.profile()[0].mentions == 2 and store.profile_conflicts() == []
    store.close()
