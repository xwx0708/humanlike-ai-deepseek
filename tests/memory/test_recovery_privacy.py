"""归档、彻底删除和后台整理失败后的回归场景；只用合成数据与假模型。"""

import json

import pytest

from contracts.llm import LLMError
from contracts.memory import Segment
from llm import FakeModel
from memory.archive import ERASE, TO_ARCHIVE, forget_thread, restore_segment, search_archive
from memory.fresh import fresh_memories
from memory.merge import find_duplicates, merge_log, merge_recent_threads, recent_threads
from memory.runner import tick
from memory.scanner import Scanner, Surfaced
from memory.store import MemoryStore
from memory.subconscious import consolidate
from response.responder import PENDING_KEY, Responder
from runtime.lifecycle import FAMILIAR, RUNNING, Lifecycle

SECRET = "AURORA-7319"
SEGMENT = {"segments": [{"messages": [1], "title": "项目进展"}]}
NEW = {
    "thread_ids": [],
    "new_thread": True,
    "new_thread_title": "项目",
    "new_thread_ongoing": True,
    "related": [],
    "confidence": 1,
    "reason": "",
}
SUMMARY = {"summary": "项目进展", "new_title": "", "ongoing": True}


@pytest.fixture
def store():
    s = MemoryStore(":memory:")
    yield s
    s.close()


def make_thread(store, text=SECRET, title="秘密项目", ts=100):
    cid = store.new_conversation(ts).id
    turn = store.add_turn(cid, "user", text, ts)
    seg = store.add_segment(Segment(None, cid, title, ts, ts, [turn.id], last_recall_ts=ts))
    tid = store.new_thread(title, text, ts).id
    store.attach(tid, seg.id, 1, ts)
    return cid, seg, tid


def responder(store, phase=RUNNING):
    return Responder(store, Scanner(store), lambda: "你叫小满。", phase=lambda: phase)


@pytest.mark.parametrize("phase", [FAMILIAR, RUNNING])
def test_archive_excludes_raw_history_but_keeps_searchable_archive(store, phase):
    _, seg, tid = make_thread(store)
    r = responder(store, phase)
    r.on_user_message("秘密项目忘掉吧", 110)
    assert json.loads(store.get_meta(PENDING_KEY))["thread"] == tid
    r.on_user_message("存进档案", 120)
    prompt = r.on_user_message("今天吃什么", 130)
    assert SECRET not in prompt.system + json.dumps(prompt.contexts)
    assert store.get_segment(seg.id).state == "archived"
    assert search_archive(store, SECRET)[0].segment_id == seg.id


def test_natural_archival_does_not_hide_fresh_history(store):
    _, seg, _ = make_thread(store)
    seg.state = "archived"  # 批量遗忘，不是用户主动要求忘掉
    store.save_segment_state(seg)
    assert SECRET in " ".join(m.turn.content for m in fresh_memories(store, 120))


@pytest.mark.parametrize("mode", [ERASE, TO_ARCHIVE])
def test_forgetting_a_surfaced_memory_does_not_break_following_messages(store, mode):
    _, seg, _ = make_thread(store)
    r = responder(store)
    r.scanner.buffer = [Surfaced(seg.id, SECRET, 1, 0, 100)]
    r.scanner.priming[seg.id] = 0.5
    r.on_user_message("秘密项目忘掉吧", 110)
    r.on_user_message(mode, 120)
    prompt = r.on_user_message("在吗", 130)
    assert SECRET not in prompt.system
    assert r.scanner.buffer == [] and seg.id not in r.scanner.priming


def test_scanner_revalidates_archived_and_abstract_cached_text(store):
    _, seg, tid = make_thread(store)
    r = responder(store)
    r.scanner.buffer = [Surfaced(seg.id, SECRET, 1, 0, 100)]
    seg.state, seg.trace = "abstract", "项目"
    store.save_segment_state(seg)
    store.add_busy_span(100, 20000)  # 原消息不再新鲜
    assert SECRET not in r.on_user_message("在吗", 20000).system
    forget_thread(store, tid, TO_ARCHIVE, 20010)  # 模拟别的连接更新状态
    r.scanner.tick(20020)
    assert r.scanner.buffer == []


@pytest.mark.parametrize("confidence", [0.99, 0.5])
def test_erase_after_successful_or_rejected_nightly_merge(store, confidence):
    _, _, first = make_thread(store, "妈妈住院治疗", "妈妈住院", 100)
    _, _, second = make_thread(store, "妈妈住院复查", "妈妈住院", 200)
    replies = [{"same": [{"id": second, "confidence": confidence, "reason": "同一件事"}]}]
    if confidence >= 0.8:
        replies.append(SUMMARY)
    else:
        replies.append({"same": []})
    merge_recent_threads(store, FakeModel(replies), 300)
    assert merge_log(store)
    forget_thread(store, first, ERASE, 310)
    assert first not in store.live_threads()
    assert merge_log(store) == []
    assert store.db.execute("PRAGMA foreign_key_check").fetchall() == []


def test_erase_invalidates_conversation_and_shared_thread_summaries(store):
    cid, seg, tid = make_thread(store)
    store.set_conversation_summary(cid, SECRET)
    other_turn = store.add_turn(cid, "user", "今天吃面", 105)
    other_seg = store.add_segment(Segment(None, cid, "吃面", 105, 105, [other_turn.id]))
    other_tid = store.new_thread("生活", f"{SECRET}，今天吃面", 105).id
    store.attach(other_tid, seg.id, 1, 105)
    store.attach(other_tid, other_seg.id, 1, 105)
    forget_thread(store, tid, ERASE, 110)
    assert SECRET not in store.get_conversation(cid).summary
    assert SECRET not in store.get_thread(other_tid).summary
    assert store.get_segment(other_seg.id).title == "吃面"


def test_archived_thread_is_excluded_from_all_nightly_merge_entries(store):
    _, _, forgotten = make_thread(store)
    forget_thread(store, forgotten, TO_ARCHIVE, 150)
    _, _, current = make_thread(store, "秘密项目正常进展", ts=200)
    store.relate_threads(forgotten, current, "以前有关")
    assert forgotten not in recent_threads(store, 300)
    assert forgotten not in find_duplicates(store, current)
    model = FakeModel([])
    report = merge_recent_threads(store, model, 300)
    assert model.calls == [] and report.merged == []
    assert current in store.live_threads()


def test_explicit_restore_makes_raw_history_visible_again(store):
    _, seg, tid = make_thread(store)
    forget_thread(store, tid, TO_ARCHIVE, 110)
    restore_segment(store, seg.id, 120)
    assert SECRET in " ".join(m.turn.content for m in fresh_memories(store, 130))


def test_assignment_failure_is_resumed_after_reopening_database(tmp_path):
    path = tmp_path / "memory.db"
    store = MemoryStore(path)
    cid = store.new_conversation(100).id
    store.add_turn(cid, "user", "今天有项目进展", 100)
    with pytest.raises(LLMError):
        consolidate(store, FakeModel([SEGMENT]), cid, 110)
    store.close()
    store = MemoryStore(path)
    try:
        model = FakeModel([NEW])
        # 即使没有新消息，后台下一次 tick 也要继续未完成的整理
        tick(store, model, 120, Lifecycle(path=None, phase=RUNNING), 100000)
        [seg] = store.segments(cid)
        assert len(store.threads_of_segment(seg.id)) == 1
        assert len(store.segments()) == 1
        consolidate(store, FakeModel([]), cid, 130)  # 完成后不重复执行
    finally:
        store.close()


def test_profile_conflict_failure_does_not_duplicate_fact_mentions(store):
    store.add_profile_fact(None, "居住地", "住在杭州", 50)
    cid = store.new_conversation(100).id
    store.add_turn(cid, "user", "住在上海", 100)
    reply = {
        "segments": [
            {
                "messages": [1],
                "title": "搬家",
                "facts": [{"about": "用户", "category": "居住地", "fact": "住在上海"}],
            }
        ]
    }
    with pytest.raises(LLMError):
        consolidate(store, FakeModel([reply]), cid, 110)
    model = FakeModel([{"conflicts": [], "reason": "未冲突"}, NEW])
    consolidate(store, model, cid, 120)
    facts = [f for f in store.profile() if f.fact == "住在上海"]
    assert len(facts) == 1 and facts[0].mentions == 1
    assert len(model.calls) == 2


def test_existing_thread_summary_failure_does_not_assign_twice(store):
    _, _, tid = make_thread(store, "项目已经开始", "项目", 50)
    cid = store.new_conversation(100).id
    store.add_turn(cid, "user", "项目有进展", 100)
    attach = {**NEW, "new_thread": False, "thread_ids": [tid]}
    with pytest.raises(LLMError):
        consolidate(store, FakeModel([SEGMENT, attach]), cid, 110)
    model = FakeModel([SUMMARY, {"adjustments": []}])
    consolidate(store, model, cid, 120)
    assert len(store.live_threads()) == 1
    assert len(store.get_thread(tid).segment_ids) == 2
    assert len(model.calls) == 2


def test_failed_progress_commit_rolls_back_adjustments_before_retry(store, monkeypatch):
    _, old, tid = make_thread(store, "项目已经开始", "项目", 50)
    cid = store.new_conversation(100).id
    store.add_turn(cid, "user", "项目有进展", 100)
    attach = {**NEW, "new_thread": False, "thread_ids": [tid]}
    tuning = {"adjustments": [{"scope": "memory", "target": str(old.id), "factor": 2, "reason": "后续"}]}
    save_job = store.save_consolidation_job

    def fail_at_done(sid, batch, stage, payload):
        if stage == "done":
            raise RuntimeError("模拟进度提交失败")
        return save_job(sid, batch, stage, payload)

    monkeypatch.setattr(store, "save_consolidation_job", fail_at_done)
    with pytest.raises(RuntimeError, match="模拟进度提交失败"):
        consolidate(store, FakeModel([SEGMENT, attach, SUMMARY, tuning]), cid, 110)
    assert store.get_segment(old.id).adjustment == 1
    assert store.tuning_log() == []
    monkeypatch.setattr(store, "save_consolidation_job", save_job)
    consolidate(store, FakeModel([tuning]), cid, 120)
    assert store.get_segment(old.id).adjustment == 2
    consolidate(store, FakeModel([]), cid, 130)
    assert store.get_segment(old.id).adjustment == 2


def test_batch_retry_does_not_adjust_same_old_memory_twice(store):
    _, old, tid = make_thread(store, "项目已经开始", "项目", 50)
    cid = store.new_conversation(100).id
    store.add_turn(cid, "user", "第一件事", 100)
    store.add_turn(cid, "user", "第二件事", 105)
    segments = {"segments": [{"messages": [1], "title": "第一件事"}, {"messages": [2], "title": "第二件事"}]}
    attach = {**NEW, "new_thread": False, "thread_ids": [tid]}
    tuning = {"adjustments": [{"scope": "memory", "target": str(old.id), "factor": 2, "reason": "后续"}]}
    with pytest.raises(LLMError):
        consolidate(store, FakeModel([segments, attach, SUMMARY, attach, SUMMARY, tuning]), cid, 110)
    assert store.get_segment(old.id).adjustment == 2
    consolidate(store, FakeModel([tuning]), cid, 120)
    assert store.get_segment(old.id).adjustment == 2
    assert store.tuning_log()[-1][2].note == "本轮已调整过"


def test_legacy_orphan_segment_is_recovered_on_open(tmp_path):
    path = tmp_path / "legacy.db"
    s = MemoryStore(path)
    cid = s.new_conversation(100).id
    turn = s.add_turn(cid, "user", "项目开始", 100)
    seg = s.add_segment(Segment(None, cid, "项目开始", 100, 100, [turn.id]))
    s.close()  # 模拟旧版本留下的片段，没有整理任务记录
    s = MemoryStore(path)
    try:
        consolidate(s, FakeModel([NEW]), cid, 110)
        assert len(s.threads_of_segment(seg.id)) == 1
    finally:
        s.close()


def test_model_calls_do_not_hold_a_write_lock(tmp_path):
    path = tmp_path / "memory.db"
    s = MemoryStore(path)
    cid = s.new_conversation(100).id
    s.add_turn(cid, "user", "项目开始", 100)
    calls = []

    def answer(messages):
        other = MemoryStore(path)
        try:
            other.add_turn(cid, "assistant", "并发收发消息", 115 + len(calls))
        finally:
            other.close()
        calls.append(messages)
        return SEGMENT if len(calls) == 1 else NEW

    try:
        consolidate(s, FakeModel(answer), cid, 110)
        assert len(calls) == 2
        assert len(s.all_turns()) == 3
    finally:
        s.close()


def test_erasure_cleans_all_generations_of_merged_threads(store):
    _, _, first = make_thread(store, title="项目", ts=100)
    _, _, second = make_thread(store, title="项目", ts=200)
    _, _, third = make_thread(store, title="项目", ts=300)
    store.merge_threads(second, third, 310)
    store.merge_threads(first, second, 320)
    forget_thread(store, first, ERASE, 330)
    assert store.live_threads() == [] and store.segments() == []
    assert store.db.execute("PRAGMA foreign_key_check").fetchall() == []


def test_background_summary_cannot_restore_content_forgotten_during_model_call(store):
    _, secret_seg, secret_tid = make_thread(store, title="旧项目", ts=50)
    shared = store.new_thread("综合项目", SECRET, 60).id
    store.attach(shared, secret_seg.id, 1, 60)
    cid = store.new_conversation(100).id
    store.add_turn(cid, "user", "新的普通进展", 100)
    attach = {**NEW, "new_thread": False, "thread_ids": [shared]}
    answers = [SEGMENT, attach]

    def answer(messages):
        if answers:
            return answers.pop(0)
        forget_thread(store, secret_tid, TO_ARCHIVE, 115)
        return {**SUMMARY, "summary": f"旧内容：{SECRET}"}

    consolidate(store, FakeModel(answer), cid, 110)
    assert SECRET not in store.get_thread(shared).summary
    assert store.pending_consolidations(cid)
    consolidate(store, FakeModel([SUMMARY]), cid, 120)
    assert store.pending_consolidations(cid) == []
