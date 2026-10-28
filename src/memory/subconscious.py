"""潜意识的整理流程：对话停顿时，把一段新对话切成片段、挂到事件线、按需修改参数。

见 01-记忆/潜意识.md。目前是最小可用版本，几处暂时做法在函数里用"暂行"标出，对应 00-总纲/待办.md。
"""

import random
import time
from dataclasses import dataclass, field, replace

from contracts.llm import ChatModel
from contracts.memory import ChatTurn, Segment, days
from memory.boundaries import conversation_closed
from memory.candidates import select_candidates
from memory.clock import MemoryClock, clock_of
from memory.forgetting import clamp_adjustment, initial_half_life, relative_arousal
from memory.profile import EVOKE, OVERWRITE, resolve_conflict
from memory.prompts import (
    CONFLICT_SCHEMA,
    CONVERSATION_SCHEMA,
    SEGMENT_SCHEMA,
    SUMMARY_SCHEMA,
    THREAD_SCHEMA,
    TUNING_SCHEMA,
    Adjustment,
    MemoryCard,
    ThreadDecision,
    build_conflict_prompt,
    build_conversation_prompt,
    build_segment_prompt,
    build_summary_prompt,
    build_thread_prompt,
    build_tuning_prompt,
    parse_conflicts,
    parse_conversation,
    parse_segments,
    parse_summary,
    parse_thread,
    parse_tuning,
)
from memory.resident import ResidentReport, effective_strength, refresh_residents
from memory.store import MemoryStore
from memory.sweep import SweepReport, sweep
from memory.tokens import estimate_tokens
from memory.tuning import GlobalChange, adjust_global


@dataclass
class ConsolidationReport:
    segment_ids: list[int] = field(default_factory=list)
    threads: dict[int, ThreadDecision] = field(default_factory=dict)  # 片段 id → 归属决定
    candidate_sources: dict[int, dict[int, list[str]]] = field(default_factory=dict)  # 片段 id → 候选及来源
    changes: list[tuple[str, str, GlobalChange]] = field(default_factory=list)
    conflicts: list[tuple[int, int, str]] = field(default_factory=list)  # (旧事实, 新事实, 结果)
    residents: ResidentReport | None = None  # 整理后的常驻栏变化（D17）
    sweep: SweepReport | None = None  # 攒够已遗忘片段时的批量清理（D22）


def consolidate(
    store: MemoryStore,
    model: ChatModel,
    conversation_id: int,
    now_ts: float,
    rng: random.Random | None = None,
    sweep_forgotten: bool = True,
) -> ConsolidationReport:
    """rng：画像矛盾时掷骰子、批量遗忘时随机挑留下的 5% 用；测试时传入固定种子的 random.Random 以便复现。
    sweep_forgotten：最后是否批量遗忘；启动时补整理熟悉期的对话，不该在中途遗忘（D24）。"""
    rng = rng or random.Random()
    report = ConsolidationReport()
    turns = store.unsegmented_turns(conversation_id)
    if not turns:
        _mark_last_segment(store, conversation_id, now_ts)  # 已经切完的对话，结束后也要补标末个片段
        _summarize_if_closed(store, model, conversation_id, now_ts)
        return report
    knobs = store.load_global_knobs()
    earlier = store.segments(conversation_id)
    prev = earlier[-1] if earlier else None
    boundaries = store.boundaries(conversation_id)

    # 1. 切段第二层。长对话分批，每批不超过潜意识模型能看的长度（待办 #4）
    for batch in split_turns(turns, boundaries):
        reply = model.chat(
            build_segment_prompt(batch, boundaries, prev.title if prev else None),
            schema=SEGMENT_SCHEMA,
            temperature=0.2,
        )
        drafts = parse_segments(reply.data, batch)
        ts_of = {t.id: t.ts for t in batch}
        for i, d in enumerate(drafts):
            person_ids = [store.upsert_person(name, aliases) for name, aliases in d.people]
            # 暂行：情绪基线还没实现，先按基线 0 计算相对唤醒度
            rel = relative_arousal(d.emotion.arousal, baseline=0.0)
            # 整场对话的首个片段；末个片段要等对话结束时才知道，见 _mark_last_segment
            at_edge = prev is None and i == 0
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
                    remember_requested=d.remember_requested,
                )
            )
            if prev is not None:
                store.link_segments(prev.id, seg.id, "next")
            prev = seg
            report.segment_ids.append(seg.id)
            for about, category, fact in d.facts:  # 稳定事实进用户画像：原文不改写、不遗忘
                person = None if about in ("用户", "我", "他", "她") else store.upsert_person(about)
                before = {f.id: f for f in store.profile(person) if f.category == category}
                fid = store.add_profile_fact(person, category, fact, seg.end_ts, seg.id)
                if fid not in before and before:  # 是新事实，且同一个人同一类已有旧事实：检查矛盾
                    _handle_conflicts(store, model, rng, fid, fact, about, before, now_ts, report)

    # 2. 挂到事件线
    for sid in report.segment_ids:
        seg = store.get_segment(sid)
        people = [store.person(p).name for p in seg.person_ids]
        hits = select_candidates(store, seg)  # 四种来源取并集，最多 20 条，不含常驻栏
        report.candidate_sources[sid] = hits
        candidates = [store.thread_card(t) for t in hits]
        residents = [store.thread_card(t) for t in store.resident_threads()]  # 常驻栏整栏给，不占名额（D17）
        # 没有候选时也问一次：让潜意识模型给新事件线起一个概括这件事的标题
        r = model.chat(
            build_thread_prompt({"title": seg.title, "people": people}, candidates, residents),
            schema=THREAD_SCHEMA,
            temperature=0,  # 判断类任务，尽量稳定
        )
        decision = parse_thread(r.data, [c["id"] for c in candidates + residents])
        if decision.attach_to:
            targets = decision.attach_to
        else:
            new_tid = store.new_thread(decision.new_title or seg.title, seg.title, now_ts).id
            store.update_thread(new_tid, ongoing=decision.new_ongoing)
            targets = [new_tid]
        for tid in targets:
            store.attach(tid, sid, decision.confidence if decision.attach_to else 1.0, now_ts)
        for rid in decision.related:
            for tid in targets:
                store.relate_threads(tid, rid, decision.reason)
        for tid in decision.attach_to:  # 挂到已有的线上：更新这条线的摘要（新建的线摘要就是这个片段的标题）
            update_summary(store, model, tid)
        report.threads[sid] = decision

    # 3. 修改参数：拿新片段所在事件线上的旧片段给潜意识模型看
    clock = clock_of(store)
    adjusted: set[str] = set()  # 同一轮整理里，每条旧记忆最多调一次，避免几个新片段为同一个理由重复加码
    for sid in report.segment_ids:
        seg = store.get_segment(sid)
        related_ids = {
            s
            for t in store.threads_of_segment(sid)
            for s in store.get_thread(t).segment_ids
            if s not in report.segment_ids
            and store.get_segment(s).state != "archived"  # 移进档案的不再出现（D22）
        }
        related = [_card(store, store.get_segment(s), now_ts, clock) for s in sorted(related_ids)]
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
    _summarize_if_closed(store, model, conversation_id, now_ts)
    report.residents = refresh_residents(store, now_ts)  # 新片段、新事实可能改变常驻栏（D17）
    if sweep_forgotten:
        report.sweep = sweep(store, model, now_ts, rng=rng)
        if report.sweep:  # 有片段被遗忘：它们所在的对话，摘要重新概括、更抽象一层
            gone = set(report.sweep.archived) | {int(i) for i in report.sweep.kept}
            for cid in sorted({store.get_segment(i).conversation_id for i in gone}):
                summarize_conversation(store, model, cid)  # 已遗忘的攒够了就批量清理（D22）
    return report


SEGMENT_BUDGET = 6000  # 每批消息最多占多少 token（暂定）。Qwen 上下文 16384，要给提示词和输出留余量


def split_turns(
    turns: list[ChatTurn], boundaries: dict[int, list[str]], budget: int = SEGMENT_BUDGET
) -> list[list[ChatTurn]]:
    """把消息分成几批，每批不超过 budget。超出时尽量在这一批后半段最后一个候选边界前切开，
    让一个话题不被拦腰切断；后半段没有候选边界就直接在超出处切。"""
    batches, cur, used = [], [], 0
    for t in turns:
        cost = estimate_tokens(t.content) + 12  # 12：编号、时间、角色等
        if cur and used + cost > budget:
            cut = next(
                (i for i in range(len(cur) - 1, len(cur) // 2, -1) if cur[i].id in boundaries), len(cur)
            )
            batches.append(cur[:cut])
            cur = cur[cut:]
            used = sum(estimate_tokens(x.content) + 12 for x in cur)
        cur.append(t)
        used += cost
    if cur:
        batches.append(cur)
    return batches


def _handle_conflicts(store, model, rng, new_id, new_fact, about, olds, now_ts, report) -> None:
    """新事实和旧事实矛盾时，按旧事实的牢固度三选一：唤起 / 覆盖 / 都不（决策 D16）。"""
    r = model.chat(
        build_conflict_prompt(new_fact, [(i, f.fact) for i, f in olds.items()], about),
        schema=CONFLICT_SCHEMA,
        temperature=0,
    )
    ids, reason = parse_conflicts(r.data, list(olds))
    for old_id in ids:
        old = olds[old_id]
        result = resolve_conflict(old, clock_of(store).elapsed_days(old.last_ts, now_ts), rng)
        if result.outcome == EVOKE:
            store.add_evocation(old_id, new_id, now_ts)
        elif result.outcome == OVERWRITE:
            store.archive_profile_fact(old_id, new_id, now_ts)
        store.log_profile_conflict(
            old_id, new_id, now_ts, result.firmness, result.probabilities, result.roll, result.outcome, reason
        )
        report.conflicts.append((old_id, new_id, result.outcome))


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


def _summarize_if_closed(store: MemoryStore, model: ChatModel, cid: int, now_ts: float) -> None:
    """对话结束后概括一次：一场对话本身也是一条记忆（待办 #7）。"""
    if conversation_closed(store, cid, now_ts) and not store.get_conversation(cid).summary:
        summarize_conversation(store, model, cid)


def summarize_conversation(store: MemoryStore, model: ChatModel, cid: int) -> None:
    """按这场对话里现在还记得的片段（完整标题或只剩关键词）重新概括。全部被遗忘时不再改动：
    这时的摘要本身就是这场对话留下的残影（事件切分.md"遗忘效果"）。"""
    parts = [
        s.title if s.state == "active" else f"（模糊）{s.trace}"
        for s in store.segments(cid)
        if s.state != "archived"
    ]
    if not parts:
        return
    date = time.strftime("%Y-%m-%d", time.localtime(store.get_conversation(cid).start_ts))
    r = model.chat(build_conversation_prompt(date, parts), schema=CONVERSATION_SCHEMA, temperature=0)
    store.set_conversation_summary(cid, parse_conversation(r.data))


def update_summary(store: MemoryStore, model: ChatModel, thread_id: int) -> None:
    t = store.get_thread(thread_id)
    history = [  # 最近 10 个片段就够概括来龙去脉；移进档案的不算，抽象化的只给关键词（D22）
        s if s.state == "active" else replace(s, title=f"（只剩关键词）{s.trace}")
        for s in (store.get_segment(i) for i in t.segment_ids)
        if s.state != "archived"
    ][-10:]
    r = model.chat(build_summary_prompt(t.title, t.summary, history), schema=SUMMARY_SCHEMA, temperature=0)
    summary, title, ongoing = parse_summary(r.data, t.title)
    store.update_thread(thread_id, title=title, summary=summary, ongoing=ongoing)


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
            note = "" if abs(new_total - requested) < 1e-9 else "已限幅"
            floor = min(
                1.0, seg.adjustment
            )  # D19：事件完结前，系数不能调到 1 以下（已经低于 1 的不再往下调）
            if new_total < floor and not is_finished(store, seg.id):
                new_total, note = floor, "事件未完结，不允许系数小于 1"
            applied = new_total != seg.adjustment
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


def _card(store: MemoryStore, seg: Segment, now_ts: float, clock: MemoryClock) -> MemoryCard:
    return MemoryCard(
        seg.id,
        seg.title if seg.state == "active" else f"（只剩关键词）{seg.trace}",
        days(now_ts - seg.start_ts),
        effective_strength(store, seg, now_ts, clock),  # 常驻的线不遗忘（D17）
        seg.half_life,
        [store.person(p).name for p in seg.person_ids],
        is_finished(store, seg.id),
    )


def is_finished(store: MemoryStore, segment_id: int) -> bool:
    """片段所在的事件线是否都已完结（不在进行中）。不属于任何事件线的，不算完结。"""
    tids = store.threads_of_segment(segment_id)
    return bool(tids) and not any(store.get_thread(t).ongoing for t in tids)
