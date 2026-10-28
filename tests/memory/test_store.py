"""记忆存储：对话、片段、事件线、人物别名表、参数与日志。"""

import pytest

from contracts.memory import EmotionScore, Segment
from memory.forgetting import ForgettingKnobs
from memory.store import MemoryStore
from memory.tuning import adjust_global


@pytest.fixture
def store():
    s = MemoryStore(":memory:")
    yield s
    s.close()


def make_segment(store, cid, turns, title, **kw):
    ts = [t.ts for t in turns]
    return store.add_segment(Segment(None, cid, title, min(ts), max(ts), [t.id for t in turns], **kw))


def test_conversation_and_turns(store):
    c = store.new_conversation(100.0)
    a = store.add_turn(c.id, "user", "我妈最近老咳嗽", 100.0)
    store.add_turn(c.id, "assistant", "多久了？", 105.0)
    assert store.get_conversation(c.id).end_ts == 105.0
    assert [t.content for t in store.turns(c.id)] == ["我妈最近老咳嗽", "多久了？"]
    assert [t.content for t in store.turns(c.id, after_id=a.id)] == ["多久了？"]
    assert store.last_conversation().id == c.id


def test_unsegmented_turns_and_overlap(store):
    c = store.new_conversation(0)
    t = [store.add_turn(c.id, "user", f"第{i}句", i) for i in range(4)]
    make_segment(store, c.id, t[:2], "A")
    make_segment(store, c.id, t[1:3], "B")  # 第 1 句同时属于两个片段
    assert [x.content for x in store.unsegmented_turns(c.id)] == ["第3句"]


def test_boundaries(store):
    c = store.new_conversation(0)
    t = store.add_turn(c.id, "user", "对了，周末爬山吗", 0)
    store.add_boundary(t.id, "语言信号")
    store.add_boundary(t.id, "语言信号")
    store.add_boundary(t.id, "长度上限")
    assert store.boundaries(c.id) == {t.id: ["语言信号", "长度上限"]}


def test_segment_roundtrip_and_state(store):
    c = store.new_conversation(0)
    t = [store.add_turn(c.id, "user", "x", i) for i in range(2)]
    mom = store.upsert_person("妈妈", ["我妈"])
    seg = make_segment(
        store,
        c.id,
        t,
        "妈妈住院了",
        person_ids=[mom],
        at_edge=True,
        emotion=EmotionScore(0.8, -0.7, 0.6, "担心"),
        half_life=40.0,
    )
    got = store.get_segment(seg.id)
    assert got.title == "妈妈住院了" and got.message_ids == [t[0].id, t[1].id] and got.person_ids == [mom]
    assert got.emotion == EmotionScore(0.8, -0.7, 0.6, "担心") and got.at_edge and got.half_life == 40.0
    got.state, got.trace = "forgotten", "十月聊过妈妈住院"
    store.save_segment_state(got)
    assert [s.trace for s in store.segments(state="forgotten")] == ["十月聊过妈妈住院"]
    assert store.segments(state="active") == []


def test_segment_links_form_a_graph(store):
    c = store.new_conversation(0)
    t = [store.add_turn(c.id, "user", "x", i) for i in range(3)]
    a, b, d = (make_segment(store, c.id, [x], n) for x, n in zip(t, "ABD", strict=True))
    store.link_segments(a.id, b.id)
    store.link_segments(a.id, d.id)
    assert sorted(store.segment_links(c.id)) == [(a.id, b.id, "next"), (a.id, d.id, "next")]


def test_threads_attach_card_and_merge(store):
    c = store.new_conversation(0)
    t = [store.add_turn(c.id, "user", "x", i) for i in range(3)]
    mom = store.upsert_person("妈妈", ["我妈", "老妈"])
    s1 = make_segment(store, c.id, [t[0]], "妈妈老咳嗽", person_ids=[mom])
    s2 = make_segment(store, c.id, [t[1]], "妈妈住院了", person_ids=[mom])
    s3 = make_segment(store, c.id, [t[2]], "周末爬山")
    th = store.new_thread("妈妈的健康", "先咳嗽，后住院", 0)
    dup = store.new_thread("妈妈住院", "住院", 1)
    store.attach(th.id, s1.id, 0.9, 2)
    store.attach(dup.id, s2.id, 0.6, 3)
    store.relate_threads(dup.id, th.id, "同一个人")
    assert store.threads_of_segment(s2.id) == [dup.id]

    store.merge_threads(th.id, dup.id, 4)
    merged = store.get_thread(th.id)
    assert merged.segment_ids == [s1.id, s2.id] and merged.person_ids == [mom] and merged.updated_ts == 4
    assert store.get_thread(dup.id).merged_into == th.id
    assert store.live_threads() == [th.id]
    assert store.thread_card(th.id) == {
        "id": th.id,
        "title": "妈妈的健康",
        "summary": "先咳嗽，后住院",
        "latest_segment": "妈妈住院了",
        "people": ["妈妈"],
    }
    assert s3.id not in merged.segment_ids


def test_alias_table_merges_people(store):
    a = store.upsert_person("妈妈", ["我妈"])
    b = store.upsert_person("母亲", ["老妈"])
    assert a != b
    c = store.upsert_person("老妈", ["我妈"])  # 发现"老妈"和"我妈"是同一个人：两条合并
    assert c == a
    assert store.resolve("母亲") == a and store.resolve("老妈") == a
    assert store.person(a).aliases == sorted(["妈妈", "我妈", "母亲", "老妈"])


def test_knobs_and_tuning_log(store):
    assert store.load_global_knobs() == ForgettingKnobs()
    log = [c for _, _, c in store.tuning_log("global")]
    knobs, change = adjust_global(store.load_global_knobs(), "base_half_life", 1.05, 10.0, "记性好", log)
    store.save_global_knobs(knobs)
    store.log_tuning("global", "base_half_life", change, {"segment": 3})
    assert store.load_global_knobs().base_half_life == pytest.approx(7.35)
    scope, target, logged = store.tuning_log()[0]
    assert (scope, target, logged) == ("global", "base_half_life", change)


def test_file_store_persists(tmp_path):
    path = tmp_path / "sub" / "m.db"
    s = MemoryStore(path)
    c = s.new_conversation(0)
    s.add_turn(c.id, "user", "记住这个", 0)
    s.close()
    s2 = MemoryStore(path)
    assert [t.content for t in s2.turns(c.id)] == ["记住这个"]
    s2.close()
