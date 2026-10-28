"""事件线候选：五种来源各自生效，并集上限 20。"""

import pytest

from contracts.memory import EmotionScore, Segment
from memory.candidates import CandidateRules, bm25, select_candidates, tokenize
from memory.store import MemoryStore


@pytest.fixture
def store():
    s = MemoryStore(":memory:")
    yield s
    s.close()


def thread_with(store, title, text, ts, people=(), emotion=None):
    """建一场对话、一个片段、一条事件线。返回 (事件线 id, 片段)。"""
    c = store.new_conversation(ts)
    t = store.add_turn(c.id, "user", text, ts)
    seg = store.add_segment(Segment(None, c.id, title, ts, ts, [t.id], list(people), emotion))
    th = store.new_thread(title, title, ts)
    store.attach(th.id, seg.id, 1.0, ts)
    return th.id, seg


def new_segment(store, title, text, ts, people=()):
    c = store.new_conversation(ts)
    t = store.add_turn(c.id, "user", text, ts)
    return store.add_segment(Segment(None, c.id, title, ts, ts, [t.id], list(people)))


def test_tokenize_and_bm25():
    assert "住院" in tokenize("我妈住院了，查出来是肺炎") and "了" not in tokenize("我妈住院了")
    docs = {1: tokenize("妈妈 咳嗽 肺炎 住院"), 2: tokenize("周末 爬山 计划"), 3: tokenize("老板 请假 辞职")}
    ranked = bm25(tokenize("肺炎住院"), docs)
    assert ranked[0][0] == 1 and all(d != 2 for d, _ in ranked)


def test_each_source_contributes(store):
    rules = CandidateRules(recent=1)
    mom = store.upsert_person("妈妈", ["我妈", "母亲"])
    by_people, _ = thread_with(store, "妈妈的健康", "我妈最近老咳嗽", 100, people=[mom])
    # 关键词检索的是事件线的标题、摘要和片段标题（不是原文），片段标题是潜意识模型写的一句话概括
    by_keyword, _ = thread_with(store, "老板嫌他请假太多，他想辞职", "老板说我请假太多，想辞职", 200)
    pinned, _ = thread_with(store, "花生过敏", "记住我对花生过敏", 300, emotion=EmotionScore(0.1, 0, 0.95))
    store.set_resident(pinned, 300, True, "测试")  # 常驻栏单独给，不在候选里（D17）
    by_archive, _ = thread_with(
        store, "老家的事", "母亲在老家种了很多菜", 400
    )  # 原文提到"母亲"，但片段没登记人物
    for i in range(3):  # 更晚的几条无关事件线，占掉"最近活跃"
        thread_with(store, f"无关{i}", f"无关的闲聊{i}", 500 + i)
    recent = store.live_threads()[0]

    seg = new_segment(
        store, "他妈妈住院了，他请假陪护，老板不满", "我妈住院了，我请假陪护，老板不太高兴", 1000, [mom]
    )
    hits = select_candidates(store, seg, rules)
    assert hits[recent] == ["最近活跃"]
    assert "人物重合" in hits[by_people]
    assert "关键词重合" in hits[by_keyword]
    assert pinned not in hits
    assert "档案人名" in hits[by_archive]  # 别名表认出"母亲"就是"我妈"


def test_archive_match_ignores_short_aliases_and_own_turns(store):
    bro = store.upsert_person("哥哥", ["哥"])
    thread_with(store, "看球", "哥你看比赛了吗", 100)  # 只含单字"哥"
    seg = new_segment(store, "他哥哥换了工作", "我哥哥换了工作", 200, [bro])
    hits = select_candidates(store, seg, CandidateRules(recent=0, keyword=0))
    assert hits == {}  # "哥"太短不匹配；片段自己的消息不算


def test_capped_at_max_total_preferring_more_sources(store):
    mom = store.upsert_person("妈妈", ["我妈"])
    best, _ = thread_with(store, "妈妈的健康", "我妈咳嗽", 0, people=[mom])  # 最旧，但命中人物和关键词
    for i in range(30):
        thread_with(store, f"闲聊{i}", f"今天天气{i}", 10 + i)
    seg = new_segment(store, "他妈妈咳嗽加重", "我妈咳嗽更厉害了", 100, [mom])
    hits = select_candidates(store, seg, CandidateRules(max_total=5, recent=10))
    assert len(hits) == 5 and best in hits
    assert next(iter(hits)) == best  # 命中来源最多，排在最前
