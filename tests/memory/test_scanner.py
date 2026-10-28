"""扫描器与浮现区（待办 #9）、主动检索、记忆使用回执（待办 #10）。"""

import random

import pytest

from contracts.memory import EmotionScore, Segment
from memory.scanner import Scanner, ScannerKnobs, search_memory, use_memory
from memory.store import MemoryStore

DAY = 86400.0
MIN = 60.0


@pytest.fixture
def store():
    s = MemoryStore(":memory:")
    yield s
    s.close()


def add(store, title, ts, valence=0.0, people=(), half_life=30.0):
    c = store.new_conversation(ts)
    t = store.add_turn(c.id, "user", title, ts)
    return store.add_segment(
        Segment(
            None,
            c.id,
            title,
            ts,
            ts,
            [t.id],
            list(people),
            EmotionScore(0.3, valence, 0.3),
            half_life=half_life,
            last_recall_ts=ts,
        )
    )


def test_priming_charges_related_memories(store):
    mom = store.upsert_person("妈妈", ["我妈"])
    cough = add(store, "妈妈最近老咳嗽", 0, people=[mom])
    hike = add(store, "周末去爬山", 0)
    sc = Scanner(store)
    sc.set_topic("我妈今天去医院复查咳嗽", [mom])
    assert sc.priming[cough.id] > 0 and hike.id not in sc.priming
    assert sc.probability(cough, 1.0) > sc.probability(hike, 1.0)


def test_negative_memories_surface_less_unless_related(store):
    sad = add(store, "和朋友吵架了", 0, valence=-0.9)
    plain = add(store, "吃了碗面", 0, valence=0.0)
    sc = Scanner(store)
    assert sc.probability(sad, 1.0) < sc.probability(plain, 1.0)
    sc.set_topic("又和朋友吵架")
    assert sc.probability(sad, 1.0) > sc.probability(plain, 1.0)


def test_surfacing_is_per_minute_not_per_scan(store):
    seg = add(store, "妈妈最近老咳嗽", 0)
    sc = Scanner(store, ScannerKnobs(rate=1.0, threshold=0.0, temperature=0.01))  # 每分钟强度约 1
    # 两次扫描隔 1 秒，命中概率约 1.6%；隔 10 分钟，几乎必中。与扫描多快无关，只与经过多久有关
    hits_fast = sum(bool(Scanner(store, sc.k, random.Random(i)).tick(1)) for i in range(200))
    sc2 = Scanner(store, sc.k, random.Random(0))
    sc2._last_scan[seg.id] = 0
    assert hits_fast < 200 and bool(sc2.tick(10 * MIN))


def test_buffer_keeps_only_a_few(store):
    for i in range(10):
        add(store, f"小事 {i}", 0)
    sc = Scanner(
        store, ScannerKnobs(rate=100.0, threshold=0.0, temperature=0.01, buffer_size=4), random.Random(1)
    )
    sc.tick(5 * MIN)
    assert len(sc.buffer) == 4  # 旧的被挤掉


def test_search_memory_and_use_receipt(store):
    seg = add(store, "他对花生过敏", 0)
    add(store, "周末去爬山", 0)
    [hit] = search_memory(store, "花生过敏", 10)
    assert hit.segment_id == seg.id and 0 < hit.clarity <= 1
    use_memory(store, seg.id, 10)
    after = store.get_segment(seg.id)
    assert after.recall_count == 1 and after.last_recall_ts == 10 and after.half_life >= seg.half_life
