"""长对话分批（待办 #4）、熟悉期 → 启动（D24）、忘掉与翻档案（D25）。"""

import pytest

from contracts.memory import ChatTurn, EmotionScore, Segment
from llm import FakeModel
from memory.archive import ERASE, TO_ARCHIVE, forget_thread, restore_segment, search_archive
from memory.lifecycle import FAMILIAR, RUNNING, context_used, maybe_start, phase, should_start
from memory.store import MemoryStore
from memory.subconscious import split_turns
from memory.tokens import estimate_tokens

DAY = 86400.0


@pytest.fixture
def store():
    s = MemoryStore(":memory:")
    yield s
    s.close()


def seg_reply(n, title="闲聊", facts=()):
    return {
        "segments": [
            {
                "messages": list(range(1, n + 1)),
                "title": title,
                "people": [],
                "arousal": 0.2,
                "valence": 0,
                "importance": 0.2,
                "emotion_reason": "",
                "facts": list(facts),
            }
        ]
    }


NEW = {
    "thread_ids": [],
    "new_thread": True,
    "new_thread_title": "闲聊",
    "related": [],
    "new_thread_ongoing": False,
    "confidence": 1,
    "reason": "",
}


# ---------- 长对话分批 ----------


def test_estimate_tokens_is_conservative_for_chinese():
    assert estimate_tokens("我妈住院了") == 5
    assert estimate_tokens("hello world!") == 4


def test_split_turns_respects_budget_and_prefers_boundaries():
    turns = [ChatTurn(i, 1, "user", "字" * 100, i) for i in range(1, 11)]  # 每条约 112
    batches = split_turns(turns, {}, budget=500)
    assert [len(b) for b in batches] == [4, 4, 2]
    batches = split_turns(turns, {4: ["语言信号"]}, budget=500)  # 第 4 条是新话题的开头：在它前面切
    assert [b[0].id for b in batches] == [1, 4, 8]
    assert [t.id for b in batches for t in b] == list(range(1, 11))  # 一条不漏、不重复


# ---------- 熟悉期 → 启动 ----------


def test_familiar_phase_starts_at_70_percent_of_context(store):
    c = store.new_conversation(0)
    for i in range(3):
        store.add_turn(c.id, "user", "字" * 96, i * 60)  # 每条 100
    assert phase(store) == FAMILIAR and context_used(store) == 300
    assert not should_start(store, 500)  # 300 < 350
    assert should_start(store, 400)  # 300 ≥ 280


def test_start_up_consolidates_everything_and_fills_strength(store):
    c1 = store.new_conversation(0)
    store.add_turn(c1.id, "user", "字" * 96, 0)
    c2 = store.new_conversation(10 * DAY)
    store.add_turn(c2.id, "user", "字" * 96, 10 * DAY)
    assert maybe_start(store, FakeModel([]), 10 * DAY, context_limit=10_000) is None  # 还没到 70%
    fake = FakeModel([seg_reply(1, "第一场"), NEW, {"summary": "聊了第一场"}, seg_reply(1, "第二场"), NEW])
    rep = maybe_start(store, fake, 10 * DAY + 60, context_limit=250)
    assert rep is not None and len(rep.reports) == 2 and phase(store) == RUNNING
    segs = store.segments()
    assert [s.title for s in segs] == ["第一场", "第二场"]
    assert all(s.last_recall_ts == 10 * DAY + 60 for s in segs)  # 初始强度满格
    assert maybe_start(store, FakeModel([]), 20 * DAY, context_limit=250) is None  # 只启动一次


# ---------- 忘掉：用户选择存进档案或彻底删除 ----------


def _thread_with_fact(store, text="我对花生过敏", title="花生过敏"):
    c = store.new_conversation(0)
    t = store.add_turn(c.id, "user", text, 0)
    seg = store.add_segment(Segment(None, c.id, title, 0, 0, [t.id], [], EmotionScore(0.3, 0, 0.9)))
    th = store.new_thread(title, title, 0)
    store.attach(th.id, seg.id, 1.0, 0)
    store.add_profile_fact(None, "过敏与禁忌", "对花生过敏", 0, seg.id)
    store.set_resident(th.id, 0, True, "免竞争：涉及用户真实信息")
    return th.id, seg, t


def test_forget_to_archive_keeps_text_and_can_be_found_again(store):
    tid, seg, _ = _thread_with_fact(store)
    msg = forget_thread(store, tid, TO_ARCHIVE, 1)
    assert "存进档案" in msg
    assert tid not in store.live_threads() and tid not in store.resident_threads()
    assert store.get_segment(seg.id).state == "archived" and store.profile() == []  # 不再出现在记忆系统里
    assert store.profile(archived=True)[0].fact == "对花生过敏"  # 但没有被删掉

    [hit] = search_archive(store, "花生")  # 用户要求时可以搜索
    assert hit.segment_id == seg.id and "花生" in hit.excerpt
    restore_segment(store, seg.id, 2 * DAY)
    back = store.get_segment(seg.id)
    assert back.state == "active" and back.recall_count == 1 and back.last_recall_ts == 2 * DAY
    assert tid in store.live_threads()


def test_erase_deletes_everything_and_leaves_no_content(store):
    tid, seg, turn = _thread_with_fact(store)
    other, oseg, _ = _thread_with_fact(store, "我妈住院了", "妈妈住院")
    msg = forget_thread(store, tid, ERASE, 1)
    assert "彻底删除" in msg
    assert tid not in store.live_threads() and other in store.live_threads()
    with pytest.raises(KeyError):
        store.get_segment(seg.id)
    assert store.db.execute("SELECT COUNT(*) FROM turns WHERE id = ?", (turn.id,)).fetchone()[0] == 0
    assert search_archive(store, "花生") == []
    assert store.get_segment(oseg.id).title == "妈妈住院"  # 别的记忆不受影响
    [log] = store.erase_log()
    assert (log["segments"], log["turns"]) == (1, 1) and "花生" not in log["reason"]


def test_forget_requires_users_choice(store):
    tid, _, _ = _thread_with_fact(store)
    with pytest.raises(ValueError):
        forget_thread(store, tid, "随便", 1)
