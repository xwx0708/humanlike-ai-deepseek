"""三个提示词的结果校验，以及整理流程（用假模型，不联网）。"""

import pytest

from contracts.memory import ChatTurn
from llm import FakeModel
from memory.forgetting import ForgettingKnobs
from memory.prompts import (
    SEGMENT_SCHEMA,
    THREAD_SCHEMA,
    TUNING_SCHEMA,
    Adjustment,
    MemoryCard,
    build_segment_prompt,
    build_thread_prompt,
    build_tuning_prompt,
    parse_segments,
    parse_thread,
    parse_tuning,
)
from memory.store import MemoryStore
from memory.subconscious import apply_adjustments, consolidate

DAY = 86400.0


def turns(*texts):
    return [
        ChatTurn(100 + i, 1, "user" if i % 2 == 0 else "assistant", t, 1000.0 + i * 60)
        for i, t in enumerate(texts)
    ]


def test_segment_prompt_numbers_messages_and_marks_boundaries():
    ts = turns("我妈住院了", "怎么了？", "对了，周末爬山吗")
    msgs = build_segment_prompt(ts, {102: ["语言信号"]}, previous_title="他在加班")
    body = msgs[1].content
    assert "1. [" in body and "用户：我妈住院了" in body and "AI：怎么了？" in body
    assert "3. [" in body and "[候选边界：语言信号]" in body
    assert "上一批最后一个片段的标题：他在加班" in body
    assert set(SEGMENT_SCHEMA["properties"]["segments"]["items"]["required"]) >= {
        "messages",
        "title",
        "people",
    }


def test_parse_segments_validates_and_fills_gaps():
    ts = turns("a", "b", "c", "d")
    data = {
        "segments": [
            {
                "messages": [1, 2, 99],
                "title": "妈妈住院",
                "people": [{"name": "妈妈", "aliases": ["我妈", "妈妈", ""]}],
                "arousal": 1.7,
                "valence": -3,
                "importance": "x",
                "emotion_reason": "担心",
            },
            {
                "messages": [],
                "title": "空片段",
                "people": [],
                "arousal": 0,
                "valence": 0,
                "importance": 0,
                "emotion_reason": "",
            },
            {
                "messages": [4],
                "title": "爬山",
                "people": [],
                "arousal": 0.3,
                "valence": 0.5,
                "importance": 0.1,
                "emotion_reason": "",
            },
        ]
    }
    drafts = parse_segments(data, ts)
    assert [d.title for d in drafts] == ["妈妈住院", "爬山"]
    assert drafts[0].turn_ids == [100, 101, 102]  # 第 3 条被漏掉，并到前面的片段
    assert drafts[0].people == [("妈妈", ["我妈"])]
    e = drafts[0].emotion
    assert (e.arousal, e.valence, e.importance) == (1.0, -1.0, 0.1)


def test_parse_segments_rejects_empty():
    with pytest.raises(ValueError):
        parse_segments({"segments": []}, turns("a"))


def test_thread_prompt_and_parse():
    msgs = build_thread_prompt(
        {"title": "妈妈出院了", "people": ["妈妈"]},
        [
            {
                "id": 7,
                "title": "妈妈的健康",
                "summary": "咳嗽后住院",
                "latest_segment": "住院",
                "people": ["妈妈"],
            }
        ],
    )
    assert "[7] 妈妈的健康" in msgs[1].content and "人物：妈妈" in msgs[1].content
    base = {"new_thread_title": "", "related": [], "reason": "同一件事"}
    d = parse_thread({**base, "thread_ids": [7, 8], "new_thread": False, "confidence": 0.9}, [7])
    assert d.attach_to == [7] and not d.create
    low = parse_thread({**base, "thread_ids": [7], "new_thread": False, "confidence": 0.3}, [7])
    assert low.create and low.low_confidence and low.related == [7]
    new = parse_thread(
        {**base, "thread_ids": [], "new_thread": True, "new_thread_title": "爬山计划", "confidence": 0.8}, [7]
    )
    assert new.create and new.new_title == "爬山计划"
    assert set(THREAD_SCHEMA["required"]) >= {"thread_ids", "confidence"}


def test_tuning_target_copied_with_card_text():
    data = {
        "adjustments": [
            {"scope": "memory", "target": "[1] 他妈妈老咳嗽｜180 天前", "factor": 2, "reason": "r"}
        ]
    }
    assert [(a.target, a.factor) for a in parse_tuning(data, [1])] == [("1", 2.0)]


def test_tuning_prompt_and_parse():
    msgs = build_tuning_prompt(
        {"title": "妈妈住院了"}, [MemoryCard(3, "妈妈老咳嗽", 180, 0.12, 7)], ForgettingKnobs()
    )
    assert "[3] 妈妈老咳嗽｜180 天前" in msgs[1].content and "base_half_life（当前 7）" in msgs[0].content
    data = {
        "adjustments": [
            {"scope": "memory", "target": "[3]", "factor": 2.5, "reason": "住院让咳嗽变重要"},
            {"scope": "memory", "target": "99", "factor": 2, "reason": "不存在的记忆"},
            {"scope": "global", "target": "adjust_total_max", "factor": 2, "reason": "想给自己松绑"},
            {"scope": "global", "target": "forget_line", "factor": 1, "reason": "没变化"},
            {"scope": "global", "target": "base_half_life", "factor": 1.05, "reason": "长期记性好"},
        ]
    }
    adj = parse_tuning(data, [3])
    assert [(a.scope, a.target, a.factor) for a in adj] == [
        ("memory", "3", 2.5),
        ("global", "base_half_life", 1.05),
    ]
    assert set(TUNING_SCHEMA["required"]) == {"adjustments"}


@pytest.fixture
def store():
    s = MemoryStore(":memory:")
    yield s
    s.close()


def _add_old_cough_memory(store):
    c = store.new_conversation(0)
    t = store.add_turn(c.id, "user", "我妈最近老咳嗽", 0)
    fake = FakeModel(
        [
            {
                "segments": [
                    {
                        "messages": [1],
                        "title": "他妈妈最近老咳嗽",
                        "people": [{"name": "妈妈", "aliases": ["我妈"]}],
                        "arousal": 0.2,
                        "valence": -0.2,
                        "importance": 0.2,
                        "emotion_reason": "随口一提",
                    }
                ]
            },
            {
                "thread_ids": [],
                "new_thread": True,
                "new_thread_title": "妈妈的健康",
                "related": [],
                "confidence": 1,
                "reason": "没有候选",
            },
        ]
    )
    consolidate(store, fake, c.id, 10)
    assert store.get_thread(store.live_threads()[0]).title == "妈妈的健康"
    return c, t


def test_consolidate_end_to_end_with_fake_model(store):
    _add_old_cough_memory(store)
    [seg0] = store.segments()
    [thread] = store.live_threads()
    assert store.get_thread(thread).segment_ids == [seg0.id] and seg0.at_edge

    c = store.new_conversation(180 * DAY)
    store.add_turn(c.id, "user", "我妈住院了", 180 * DAY)
    store.add_turn(c.id, "assistant", "怎么回事？", 180 * DAY + 30)
    store.add_turn(c.id, "user", "对了周末还爬山吗", 180 * DAY + 90)
    fake = FakeModel(
        [
            {
                "segments": [
                    {
                        "messages": [1, 2],
                        "title": "他妈妈住院了",
                        "people": [{"name": "老妈", "aliases": ["我妈"]}],
                        "arousal": 0.8,
                        "valence": -0.8,
                        "importance": 0.7,
                        "emotion_reason": "担心",
                    },
                    {
                        "messages": [3],
                        "title": "约周末爬山",
                        "people": [],
                        "arousal": 0.2,
                        "valence": 0.3,
                        "importance": 0.1,
                        "emotion_reason": "",
                    },
                ]
            },
            {
                "thread_ids": [thread],
                "new_thread": False,
                "new_thread_title": "",
                "related": [],
                "confidence": 0.9,
                "reason": "同一个人的健康",
            },
            {
                "thread_ids": [],
                "new_thread": True,
                "new_thread_title": "周末爬山",
                "related": [],
                "confidence": 0.8,
                "reason": "新的事",
            },
            {
                "adjustments": [
                    {"scope": "memory", "target": str(seg0.id), "factor": 2.5, "reason": "住院让咳嗽变重要"}
                ]
            },
        ]
    )
    report = consolidate(store, fake, c.id, 180 * DAY + 3600)
    hospital, hike = (store.get_segment(s) for s in report.segment_ids)
    assert store.resolve("老妈") == store.resolve("我妈") == seg0.person_ids[0]  # 别名表认出是同一个人
    assert store.get_thread(thread).segment_ids == [seg0.id, hospital.id]
    assert store.get_thread(store.threads_of_segment(hike.id)[0]).title == "周末爬山"
    assert hospital.half_life > hike.half_life  # 情绪强、重要，忘得慢
    assert store.get_segment(seg0.id).half_life == pytest.approx(seg0.half_life * 2.5)  # 旧记忆被拉长
    assert [(s, t) for s, t, _ in store.tuning_log()] == [("memory", str(seg0.id))]
    assert store.segment_links(c.id) == [(hospital.id, hike.id, "next")]
    assert len(fake.calls) == 4  # 切段 1 次、归属 2 次、调参 1 次（爬山没有相关旧记忆，不调）


def test_apply_adjustments_caps_and_logs(store):
    _add_old_cough_memory(store)
    [seg] = store.segments()
    out = apply_adjustments(store, [Adjustment("memory", str(seg.id), 10, "太夸张")], 20 * DAY)
    assert out[0][2].new == 3.0 and out[0][2].note == "已限幅"
    assert store.get_segment(seg.id).half_life == pytest.approx(seg.half_life * 3)
    apply_adjustments(store, [Adjustment("global", "base_half_life", 1.5, "a")], 20 * DAY)
    apply_adjustments(store, [Adjustment("global", "base_half_life", 1.05, "b")], 21 * DAY)  # 冷却中
    logs = store.tuning_log("global")
    assert [c.applied for _, _, c in logs] == [True, False]
    assert store.load_global_knobs().base_half_life == pytest.approx(7.7)
