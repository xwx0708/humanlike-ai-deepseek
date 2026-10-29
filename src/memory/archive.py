"""档案：忘掉的东西去哪、怎么找回（设想文档"档案层：遗忘不等于删除"一节；决策 D25，用户 2026-10-28 定）。

- 没有信息是彻底忘掉的：遗忘（包括批量遗忘的 95%）只是移出记忆系统，原文一直留在档案（储存库）里。
- 用户要求"忘掉这件事"时，由用户选择：彻底删除（连档案一起删，不可恢复），或者存进档案。
- 档案里的东西，用户要求时，AI 可以自己去搜索；找回的记忆重新回到记忆系统，相当于回想一次（D20）。
"""

from dataclasses import dataclass

from memory.candidates import bm25, tokenize
from memory.clock import clock_of
from memory.forgetting import Trace, recall
from memory.store import MemoryStore
from runtime import config

TO_ARCHIVE = "存进档案"
ERASE = "彻底删除"


def forget_thread(store: MemoryStore, tid: int, mode: str, ts: float) -> str:
    """用户要求忘掉一条事件线。mode 必须由用户选择，不替用户默认。返回给用户看的一句说明。"""
    if mode == TO_ARCHIVE:
        n = len(store.forget_thread_to_archive(tid, ts))
        return f"已从记忆里移出 {n} 个片段，原文存进档案；你要我找的时候还能找回来。"
    if mode == ERASE:
        segs, turns = store.erase_thread(tid, ts, "用户要求彻底删除")  # 删除日志不写内容
        return f"已彻底删除 {segs} 个片段和 {turns} 条消息原文，不能再找回。"
    raise ValueError(f"不认识的方式：{mode}；只能是「{TO_ARCHIVE}」或「{ERASE}」")


@dataclass
class ArchiveHit:
    segment_id: int
    title: str
    start_ts: float
    score: float
    excerpt: str


def search_archive(
    store: MemoryStore,
    query: str,
    limit: int = config.value("archive", "limit", 5),
    excerpt_chars: int = config.value("archive", "excerpt_chars", 200),
) -> list[ArchiveHit]:
    """在档案里搜索（只搜已经移出记忆系统的片段）：jieba 分词 + BM25，对象是片段标题和消息原文。
    只在用户要求时调用（D25）。"""
    docs, segs = {}, {}
    for s in store.segments(state="archived"):
        text = store.turn_text(s.message_ids)
        docs[s.id] = tokenize(s.title + " " + text)
        segs[s.id] = (s, text)
    hits = []
    for sid, score in bm25(tokenize(query), docs)[:limit]:
        s, text = segs[sid]
        hits.append(ArchiveHit(sid, s.title, s.start_ts, score, text[:excerpt_chars]))
    return hits


def restore_segment(store: MemoryStore, sid: int, ts: float) -> None:
    """把档案里找回的片段放回记忆系统：算作回想一次，强度回满、半衰期按"忘了多少"加长（D20）。
    如果它所在的事件线是被用户忘掉的，那条线也一起回来。"""
    seg = store.get_segment(sid)
    clock = clock_of(store)
    trace = Trace(seg.half_life, 0.0, seg.recall_count)
    after = recall(trace, clock.elapsed_days(seg.last_recall_ts, ts), store.load_global_knobs())
    seg.half_life, seg.recall_count, seg.last_recall_ts = after.half_life, after.recall_count, ts
    seg.state = "active"
    store.save_segment_state(seg)
    for tid in store.threads_of_segment(sid):
        store.restore_thread(tid)
