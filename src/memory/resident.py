"""常驻栏（决策 D17，用户 2026-10-28 定）。单位是事件线（故事线）。

- 常驻栏像提示词一样一直在场：作为背景知道就好，不必刻意提起，也不刻意遗忘（常驻期间不参与遗忘）。
- 三种事件线几乎一直常驻、不参与竞争：涉及用户真实信息的（是用户画像的出处）、用户特地要求记住的、被多次反复提及的。
- 其余的事件线竞争 20 个名额：得分 = 重要性 35% + 情绪强度 25% + 活跃度 25% + 回忆频率 15%，
  当下正在进行的事再乘一个系数。排名靠后的退回普通记忆池，照常参与遗忘和浮现；
  遗忘从最近一次被提及开始算，退回时不重置。
具体数值除四个权重外都是暂定的旋钮。

免竞争一旦成立就一直保持，除非用户要求修改（决策 D18）：之后即使不再满足条件
（例如画像事实被覆盖进档案），也不退出。用户要求时由 MemoryStore.release_exempt 解除。
"""

from dataclasses import dataclass

from contracts.knobs import knob
from contracts.memory import Segment
from memory.budget import DEFAULT_KNOBS as BUDGET
from memory.budget import fit_lines
from memory.clock import MemoryClock, clock_of
from memory.forgetting import strength
from memory.store import MemoryStore
from memory.tokens import estimate_tokens
from runtime import config


@dataclass(frozen=True)
class ResidentKnobs:
    seats: int = knob(20, "竞争席位", "参与竞争的常驻席位数。", lo=0, hi=100)  # 竞争名额（用户定）
    w_importance: float = knob(
        0.35, "权重：重要性", "得分 = Σ 权重 × 分项。四项权重之和建议为 1。", lo=0, hi=1, step=0.05
    )  # 四项权重（用户定）
    w_emotion: float = knob(0.25, "权重：情绪强度", "同上。", lo=0, hi=1, step=0.05)
    w_activity: float = knob(
        0.25, "权重：活跃度", "同上。活跃度按最近一次提及的时间衰减。", lo=0, hi=1, step=0.05
    )
    w_recall: float = knob(
        0.15, "权重：回忆频率", "同上。回忆频率 = 次数 / (次数 + 饱和常数)。", lo=0, hi=1, step=0.05
    )
    ongoing_factor: float = knob(
        1.5, "进行中系数", "尚未完结的事件线得分乘以此系数。", lo=1, hi=5, step=0.1
    )  # 当下正在进行的事，得分乘这个系数（用户要求有系数，数值暂定）
    activity_half_life: float = knob(
        14.0, "活跃度半衰期", "多久未被提及，活跃度减半。", lo=1, hi=180, step=1, unit="天"
    )  # 天：多久没被提及，活跃度减半（暂定）
    recall_saturation: float = knob(
        3.0, "回忆饱和常数", "回忆频率 = 次数 / (次数 + 此值)。", lo=0.5, hi=20, step=0.5
    )  # 回忆频率 = 次数 / (次数 + 此值)，回忆 3 次得 0.5（暂定）
    repeated_conversations: int = knob(
        3, "多次提及判定", '在至少多少场不同对话中被提及，算免竞争的"多次提及"。', lo=2, hi=20
    )  # 在至少几场不同的对话里被提及，算"多次反复提及"（暂定）


DEFAULT_KNOBS = config.knobs(
    ResidentKnobs(),
    "resident",
    "常驻栏",
    "免竞争项（画像出处、用户要求记住、多次提及）之外，其余事件线按加权得分竞争席位。常驻期间不衰减、不参与扫描。",
)


def score(stats: dict, idle_days: float, k: ResidentKnobs = DEFAULT_KNOBS) -> float:
    """竞争名额用的得分，0～1（乘进行中系数后可能超过 1）。stats 来自 MemoryStore.thread_stats；
    idle_days：最近一次被提及后经过的记忆时间（D21）。"""
    activity = 0.5 ** (max(0.0, idle_days) / k.activity_half_life)
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


def refresh_residents(
    store: MemoryStore, now_ts: float, k: ResidentKnobs = DEFAULT_KNOBS, clock: MemoryClock | None = None
) -> ResidentReport:
    """重新计算常驻栏：免竞争的全部常驻，其余按得分取前 k.seats 名；进出都记日志。"""
    clock = clock or clock_of(store)
    exempt, scores = {}, {}
    marks = store.exempt_marks()
    for tid in store.live_threads():
        stats = store.thread_stats(tid)
        reason = marks[tid] if tid in marks else exempt_reason(store, tid, stats, k)
        if reason and tid not in marks:
            store.mark_exempt(tid, reason)
        if reason:
            exempt[tid] = reason
        else:
            scores[tid] = score(stats, clock.elapsed_days(stats["updated_ts"], now_ts), k)
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


def effective_strength(
    store: MemoryStore, seg: Segment, now_ts: float, clock: MemoryClock | None = None
) -> float:
    """片段当前的强度。所在事件线常驻时不遗忘（强度视为 1）；否则按遗忘曲线，从最近一次被提及开始算，
    经过的时间用记忆时钟量（D21）：不对话就不遗忘。"""
    if is_resident_segment(store, seg):
        return 1.0
    return strength(seg.half_life, (clock or clock_of(store)).elapsed_days(seg.last_recall_ts, now_ts))


RESIDENT_HEADER = "以下是你一直记得的事。作为背景知道就好，不必刻意提起；只有话题相关时才自然地用上。"
FACTS_TITLE = "【关于他的事实】"
THREADS_TITLE = "【一直记得的事】"


def resident_block(store: MemoryStore, budget: int | None = None) -> str:
    """给模型的常驻栏文字：用户画像 + 常驻事件线（标题和一句话摘要）。

    budget：最多占多少 token（memory.budget，默认取设置里的常驻栏预算）。放不下时画像优先，
    其次免竞争的事件线，最后竞争得来的；没列出的只注明条数，它们仍然常驻、不遗忘。"""
    budget = BUDGET.resident if budget is None else budget
    lines = [RESIDENT_HEADER]
    facts = [
        f"- {'他' if f.person_id is None else store.person(f.person_id).name}：{f.fact}"
        for f in store.profile(everyone=True)
    ]
    exempt = store.exempt_marks()
    tids = sorted(store.resident_threads(), key=lambda t: t not in exempt)  # 稳定排序：免竞争的在前
    threads = []
    for tid in tids:
        c = store.thread_card(tid)
        threads.append(f"- [{tid}] {c['title']}：{c['summary']}")
    left = budget - estimate_tokens(RESIDENT_HEADER) - 1
    if facts:
        lines.append(FACTS_TITLE)
        left -= estimate_tokens(FACTS_TITLE) + 1
        facts = fit_lines(facts, left)
        lines += facts
        left -= sum(estimate_tokens(x) + 1 for x in facts)
    if threads:
        lines.append(THREADS_TITLE)
        left -= estimate_tokens(THREADS_TITLE) + 1
        lines += fit_lines(threads, left)
    return "\n".join(lines)
