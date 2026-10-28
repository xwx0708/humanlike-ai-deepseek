"""常驻栏（决策 D17，用户 2026-10-28 定）。单位是事件线（故事线）。

- 常驻栏像提示词一样一直在场：作为背景知道就好，不必刻意提起，也不刻意遗忘（常驻期间不参与遗忘）。
- 三种事件线几乎一直常驻、不参与竞争：涉及用户真实信息的（是用户画像的出处）、用户特地要求记住的、被多次反复提及的。
- 其余的事件线竞争 20 个名额：得分 = 重要性 35% + 情绪强度 25% + 活跃度 25% + 回忆频率 15%，
  当下正在进行的事再乘一个系数。排名靠后的退回普通记忆池，照常参与遗忘和浮现；
  遗忘从最近一次被提及开始算，退回时不重置。
具体数值除四个权重外都是暂定的旋钮。
"""

from dataclasses import dataclass

from contracts.memory import Segment, days
from memory.forgetting import Trace
from memory.store import MemoryStore


@dataclass(frozen=True)
class ResidentKnobs:
    seats: int = 20  # 竞争名额（用户定）
    w_importance: float = 0.35  # 四项权重（用户定）
    w_emotion: float = 0.25
    w_activity: float = 0.25
    w_recall: float = 0.15
    ongoing_factor: float = 1.5  # 当下正在进行的事，得分乘这个系数（用户要求有系数，数值暂定）
    activity_half_life: float = 14.0  # 天：多久没被提及，活跃度减半（暂定）
    recall_saturation: float = 3.0  # 回忆频率 = 次数 / (次数 + 此值)，回忆 3 次得 0.5（暂定）
    repeated_conversations: int = 3  # 在至少几场不同的对话里被提及，算"多次反复提及"（暂定）


DEFAULT_KNOBS = ResidentKnobs()


def score(stats: dict, now_ts: float, k: ResidentKnobs = DEFAULT_KNOBS) -> float:
    """竞争名额用的得分，0～1（乘进行中系数后可能超过 1）。stats 来自 MemoryStore.thread_stats。"""
    activity = 0.5 ** (max(0.0, days(now_ts - stats["updated_ts"])) / k.activity_half_life)
    recall = stats["recalls"] / (stats["recalls"] + k.recall_saturation)
    s = (
        k.w_importance * stats["importance"]
        + k.w_emotion * stats["arousal"]
        + k.w_activity * activity
        + k.w_recall * recall
    )
    return s * (k.ongoing_factor if stats["ongoing"] else 1.0)


def exempt_reason(store: MemoryStore, tid: int, stats: dict, k: ResidentKnobs = DEFAULT_KNOBS) -> str | None:
    """免竞争常驻的原因；不属于三种免竞争情况时返回 None。"""
    if store.thread_has_profile_fact(tid):
        return "涉及用户真实信息"
    if stats["remember_requested"]:
        return "用户要求记住"
    if stats["conversations"] >= k.repeated_conversations:
        return "多次反复提及"
    return None


@dataclass
class ResidentReport:
    exempt: dict[int, str]  # 事件线 → 免竞争原因
    ranked: list[tuple[int, float]]  # 参与竞争的事件线及得分，从高到低
    entered: list[int]
    left: list[int]


def refresh_residents(store: MemoryStore, now_ts: float, k: ResidentKnobs = DEFAULT_KNOBS) -> ResidentReport:
    """重新计算常驻栏：免竞争的全部常驻，其余按得分取前 k.seats 名；进出都记日志。"""
    exempt, scores = {}, {}
    for tid in store.live_threads():
        stats = store.thread_stats(tid)
        reason = exempt_reason(store, tid, stats, k)
        if reason:
            exempt[tid] = reason
        else:
            scores[tid] = score(stats, now_ts, k)
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    winners = {tid: (i + 1, s) for i, (tid, s) in enumerate(ranked[: k.seats])}

    current = set(store.resident_threads())
    desired = set(exempt) | set(winners)
    for tid in sorted(desired - current):
        if tid in exempt:
            store.set_resident(tid, now_ts, True, f"免竞争：{exempt[tid]}")
        else:
            rank, s = winners[tid]
            store.set_resident(tid, now_ts, True, f"竞争第 {rank} 名", s)
    for tid in sorted(current - desired):
        store.set_resident(tid, now_ts, False, "名额竞争落选，退回普通记忆池", scores.get(tid))
    return ResidentReport(exempt, ranked, sorted(desired - current), sorted(current - desired))


def is_resident_segment(store: MemoryStore, seg: Segment) -> bool:
    resident = set(store.resident_threads())
    return any(t in resident for t in store.threads_of_segment(seg.id))


def effective_strength(store: MemoryStore, seg: Segment, now_ts: float) -> float:
    """片段当前的强度。所在事件线常驻时不遗忘（强度视为 1）；否则按遗忘曲线，从最近一次被提及开始算。"""
    if is_resident_segment(store, seg):
        return 1.0
    return Trace(seg.half_life, days(seg.last_recall_ts), seg.recall_count).strength_at(days(now_ts))


RESIDENT_HEADER = "以下是你一直记得的事。作为背景知道就好，不必刻意提起；只有话题相关时才自然地用上。"


def resident_block(store: MemoryStore) -> str:
    """给模型的常驻栏文字：用户画像 + 常驻事件线（标题和一句话摘要）。"""
    lines = [RESIDENT_HEADER]
    facts = store.profile(everyone=True)
    if facts:
        lines.append("【关于他的事实】")
        for f in facts:
            who = "他" if f.person_id is None else store.person(f.person_id).name
            lines.append(f"- {who}：{f.fact}")
    tids = store.resident_threads()
    if tids:
        lines.append("【一直记得的事】")
        for tid in tids:
            c = store.thread_card(tid)
            lines.append(f"- [{tid}] {c['title']}：{c['summary']}")
    return "\n".join(lines)
