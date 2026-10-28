"""扫描器与浮现区（待办 #9）、主动检索、记忆使用回执（待办 #10）。"""

import random

import pytest

from contracts.memory import EmotionScore, Segment
from memory.scanner import Scanner, ScannerKnobs, focus_of, search_memory, use_memory
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
    sc = Scanner(
        store, ScannerKnobs(rate=1.0, threshold=0.0, temp_focus=0.01, temp_diffuse=0.01)
    )  # 每分钟强度约 1
    # 两次扫描隔 1 秒，命中概率约 1.6%；隔 10 分钟，几乎必中。与扫描多快无关，只与经过多久有关
    hits_fast = sum(bool(Scanner(store, sc.k, random.Random(i)).tick(1)) for i in range(200))
    sc2 = Scanner(store, sc.k, random.Random(0))
    sc2._last_scan[seg.id] = 0
    assert hits_fast < 200 and bool(sc2.tick(10 * MIN))


def test_buffer_keeps_only_a_few(store):
    for i in range(10):
        add(store, f"小事 {i}", 0)
    sc = Scanner(
        store,
        ScannerKnobs(rate=100.0, threshold=0.0, temp_focus=0.01, temp_diffuse=0.01, buffer_size=4),
        random.Random(1),
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


def test_temperature_follows_topic_focus(store):
    """D26：正专注于一个话题就是专注（低温），在闲聊（话题跳来跳去）就是发散（高温）。"""
    c = store.new_conversation(0)
    for i in range(6):
        store.add_turn(c.id, "user", f"考研数学第 {i} 题", i * 60)
    assert focus_of(store, c.id) == 1.0
    sc = Scanner(store)
    sc.set_focus(focus_of(store, c.id))
    assert sc.temperature == pytest.approx(0.1)
    t = store.add_turn(c.id, "user", "对了，晚饭吃啥", 400)
    store.add_boundary(t.id, "语言信号")
    assert focus_of(store, c.id) == pytest.approx(1 / 6)  # 刚换话题：发散
    sc.set_focus(focus_of(store, c.id))
    assert sc.temperature > 0.25
    strong = add(store, "考研", 0)
    hot, cold = Scanner(store), Scanner(store)
    hot.set_focus(0.0)
    cold.set_focus(1.0)
    weak = 0.3  # 弱记忆：放空时更容易冒头
    assert hot.probability(strong, weak) > cold.probability(strong, weak)
