"""潜意识的整理流程：对话停顿时，把一段新对话切成片段、挂到事件线、按需修改参数。

见 01-记忆/潜意识.md。目前是最小可用版本，几处暂时做法在函数里用"暂行"标出，对应 00-总纲/待办.md。
"""

from dataclasses import dataclass, field

from contracts.llm import ChatModel
from contracts.memory import Segment, days
from memory.boundaries import conversation_closed
from memory.forgetting import Trace, clamp_adjustment, initial_half_life, relative_arousal
from memory.prompts import (
    SEGMENT_SCHEMA,
    SUMMARY_SCHEMA,
    THREAD_SCHEMA,
    TUNING_SCHEMA,
    Adjustment,
    MemoryCard,
    ThreadDecision,
    build_segment_prompt,
    build_summary_prompt,
    build_thread_prompt,
    build_tuning_prompt,
    parse_segments,
    parse_summary,
    parse_thread,
    parse_tuning,
)
from memory.store import MemoryStore
from memory.tuning import GlobalChange, adjust_global

MAX_CANDIDATES = 20


@dataclass
class ConsolidationReport:
    segment_ids: list[int] = field(default_factory=list)
    threads: dict[int, ThreadDecision] = field(default_factory=dict)  # 片段 id → 归属决定
    changes: list[tuple[str, str, GlobalChange]] = field(default_factory=list)


def consolidate(
    store: MemoryStore, model: ChatModel, conversation_id: int, now_ts: float
) -> ConsolidationReport:
    report = ConsolidationReport()
    turns = store.unsegmented_turns(conversation_id)
    if not turns:
        _mark_last_segment(store, conversation_id, now_ts)  # 已经切完的对话，结束后也要补标末个片段
        return report
    knobs = store.load_global_knobs()
    earlier = store.segments(conversation_id)
    previous_title = earlier[-1].title if earlier else None

    # 1. 切段第二层
    reply = model.chat(
        build_segment_prompt(turns, store.boundaries(conversation_id), previous_title),
        schema=SEGMENT_SCHEMA,
        temperature=0.2,
    )
    drafts = parse_segments(reply.data, turns)
    ts_of = {t.id: t.ts for t in turns}
    prev = earlier[-1] if earlier else None
    for i, d in enumerate(drafts):
        person_ids = [store.upsert_person(name, aliases) for name, aliases in d.people]
        # 暂行：情绪基线还没实现，先按基线 0 计算相对唤醒度
        rel = relative_arousal(d.emotion.arousal, baseline=0.0)
        at_edge = not earlier and i == 0  # 首个片段；末个片段要等对话结束时才知道，见 _mark_last_segment
        seg = store.add_segment(
            Segment(
                None,
                conversation_id,
                d.title,
                min(ts_of[t] for t in d.turn_ids),
                max(ts_of[t] for t in d.turn_ids),
                d.turn_ids,
                person_ids,
                d.emotion,
                half_life=initial_half_life(rel, d.emotion.importance, knobs, at_edge=at_edge),
                last_recall_ts=max(ts_of[t] for t in d.turn_ids),
                at_edge=at_edge,
            )
        )
        if prev is not None:
            store.link_segments(prev.id, seg.id, "next")
        prev = seg
        report.segment_ids.append(seg.id)

    # 2. 挂到事件线
    for sid in report.segment_ids:
        seg = store.get_segment(sid)
        people = [store.person(p).name for p in seg.person_ids]
        # 暂行：候选取最近活跃的 20 条；正式的五种来源并集见 01-记忆/事件切分.md
        candidates = [store.thread_card(t) for t in store.live_threads()[:MAX_CANDIDATES]]
        # 没有候选时也问一次：让潜意识模型给新事件线起一个概括这件事的标题
        r = model.chat(
            build_thread_prompt({"title": seg.title, "people": people}, candidates),
            schema=THREAD_SCHEMA,
            temperature=0,  # 判断类任务，尽量稳定
        )
        decision = parse_thread(r.data, [c["id"] for c in candidates])
        targets = decision.attach_to or [
            store.new_thread(decision.new_title or seg.title, seg.title, now_ts).id
        ]
        for tid in targets:
            store.attach(tid, sid, decision.confidence if decision.attach_to else 1.0, now_ts)
        for rid in decision.related:
            for tid in targets:
                store.relate_threads(tid, rid, decision.reason)
        for tid in decision.attach_to:  # 挂到已有的线上：更新这条线的摘要（新建的线摘要就是这个片段的标题）
            _update_summary(store, model, tid)
        report.threads[sid] = decision

    # 3. 修改参数：拿新片段所在事件线上的旧片段给潜意识模型看
    adjusted: set[str] = set()  # 同一轮整理里，每条旧记忆最多调一次，避免几个新片段为同一个理由重复加码
    for sid in report.segment_ids:
        seg = store.get_segment(sid)
        related_ids = {
            s
            for t in store.threads_of_segment(sid)
            for s in store.get_thread(t).segment_ids
            if s not in report.segment_ids
        }
        related = [_card(store, store.get_segment(s), now_ts) for s in sorted(related_ids)]
        if not related:
            continue  # 暂行：没有相关旧记忆时不调参（全局参数的调整需要长期证据，以后另设流程）
        r = model.chat(
            build_tuning_prompt(_segment_brief(store, seg), related, knobs),
            schema=TUNING_SCHEMA,
            temperature=0,  # 判断类任务，尽量稳定
        )
        adjustments = parse_tuning(r.data, [m.id for m in related])
        for a in adjustments:
            if a.scope == "memory" and a.target in adjusted:
                cur = store.get_segment(int(a.target)).adjustment
                skipped = GlobalChange(
                    "adjustment", days(now_ts), cur, cur * a.factor, cur, False, a.reason, "本轮已调整过"
                )
                store.log_tuning(a.scope, a.target, skipped, {"segment": sid})
                report.changes.append((a.scope, a.target, skipped))
        fresh = [a for a in adjustments if not (a.scope == "memory" and a.target in adjusted)]
        adjusted |= {a.target for a in fresh if a.scope == "memory"}
        report.changes += apply_adjustments(store, fresh, now_ts, evidence={"segment": sid})
        knobs = store.load_global_knobs()

    _mark_last_segment(store, conversation_id, now_ts)
    return report


def _mark_last_segment(store: MemoryStore, conversation_id: int, now_ts: float) -> None:
    """对话已经结束时，给最后一个片段加上首尾加成（近因效应）。可以重复调用，只生效一次。"""
    if not conversation_closed(store, conversation_id, now_ts):
        return
    segs = store.segments(conversation_id)
    if not segs or segs[-1].at_edge:  # 只有一个片段时，它已经作为首个片段加过了
        return
    last = segs[-1]
    last.at_edge = True
    last.half_life *= store.load_global_knobs().position_bonus
    store.save_segment_state(last)


def _update_summary(store: MemoryStore, model: ChatModel, thread_id: int) -> None:
    t = store.get_thread(thread_id)
    history = [store.get_segment(s) for s in t.segment_ids[-10:]]  # 最近 10 个片段就够概括来龙去脉
    r = model.chat(build_summary_prompt(t.title, t.summary, history), schema=SUMMARY_SCHEMA, temperature=0)
    summary, title = parse_summary(r.data, t.title)
    store.update_thread(thread_id, title=title, summary=summary)


def apply_adjustments(
    store: MemoryStore, adjustments: list[Adjustment], now_ts: float, evidence: dict | None = None
) -> list[tuple[str, str, GlobalChange]]:
    """把潜意识模型的调整请求经护栏后写入存储；每条请求都记日志，包括被限幅或拒绝的。"""
    now = days(now_ts)
    knobs = store.load_global_knobs()
    done = []
    for a in adjustments:
        if a.scope == "memory":
            seg = store.get_segment(int(a.target))
            new_total = clamp_adjustment(seg.adjustment, a.factor, knobs)
            requested = seg.adjustment * a.factor
            applied = new_total != seg.adjustment
            note = "" if abs(new_total - requested) < 1e-9 else "已限幅"
            change = GlobalChange(
                "adjustment", now, seg.adjustment, requested, new_total, applied, a.reason, note
            )
            if applied:
                seg.half_life *= new_total / seg.adjustment
                seg.adjustment = new_total
                store.save_segment_state(seg)
        else:
            log = [c for _, _, c in store.tuning_log("global")]
            knobs, change = adjust_global(knobs, a.target, a.factor, now, a.reason, log)
            if change.applied:
                store.save_global_knobs(knobs)
        store.log_tuning(a.scope, a.target, change, evidence)
        done.append((a.scope, a.target, change))
    return done


def _segment_brief(store: MemoryStore, seg: Segment) -> dict:
    e = seg.emotion
    return {
        "title": seg.title,
        "people": [store.person(p).name for p in seg.person_ids],
        "arousal": e and e.arousal,
        "valence": e and e.valence,
        "importance": e and e.importance,
    }


def _card(store: MemoryStore, seg: Segment, now_ts: float) -> MemoryCard:
    trace = Trace(seg.half_life, days(seg.last_recall_ts), seg.recall_count)
    return MemoryCard(
        seg.id,
        seg.title,
        days(now_ts - seg.start_ts),
        trace.strength_at(days(now_ts)),
        seg.half_life,
        [store.person(p).name for p in seg.person_ids],
    )
