"""夜间合并事件线（待办 #8；用户方案见 01-记忆/事件切分.md 第三层第 5 条）。

**我的设想**：夜间整理时检查近期新建的事件线，重复的合并。
**补充**：错误合并比错误新建更难发现，所以挂事件线时阈值偏保守（宁可先新建），合并时更保守。

流程：
1. 取近期（created_ts 在 RECENT_DAYS 天内）新建、尚未被合并的事件线，按新建时间从早到晚逐条检查。
2. 每条用硬规则找可能重复的其他线（复用 memory.candidates 的分词、BM25 和 store 的人物查询）：
   人物重合、关键词重合、档案人名、已关联（挂线时把握不够被记成"有关"的，最可能是重复）。
3. 把这条线和它的候选（一小批）交给潜意识模型，判断哪些是同一件事，每条给把握值。
4. 把握值不低于 MERGE_CONFIDENCE_MIN 才合并；保留较早新建的那条，另一条 merged_into 指向它（不删除）。
5. 合并后：线间关联改指保留线，常驻状态和免竞争标记合理继承，重新生成保留线的摘要，最后重算一次常驻栏。
6. 每一次判断（合并的和没合并的）都记进 merge_log，便于事后追查和校准阈值。

提示词、schema、校验函数写在本文件（build_merge_prompt / MERGE_SCHEMA / parse_merge），写法同 memory.prompts：
模型的输出一律不直接信任，不存在的 id 丢掉，把握值收回 0～1。
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any

from contracts.llm import ChatModel, Message, system, user
from contracts.memory import SECONDS_PER_DAY
from memory.candidates import bm25, tokenize
from memory.resident import ResidentReport, refresh_residents
from memory.store import MemoryStore
from memory.subconscious import update_summary

RECENT_DAYS = 7.0  # 暂定：多少天内新建的线算"近期新建"。按日历时间（created_ts），不按记忆时钟
MERGE_CONFIDENCE_MIN = 0.8  # 暂定：比挂事件线的 0.6 更保守，错误合并比错误新建更难发现
MAX_CANDIDATES = 8  # 暂定：每条线一次最多给模型看几条候选（一小批，Qwen 判断更稳）
KEYWORD_TOP = 5  # 暂定：关键词重合取前几条
MIN_ALIAS_LEN = 2  # 档案人名匹配时叫法至少几个字，同 candidates.CandidateRules.min_alias_len
SEGMENT_TITLES = 5  # 暂定：卡片里列出最近几个片段的标题

MERGE_LOG_SCHEMA = """
CREATE TABLE IF NOT EXISTS merge_log (
    id INTEGER PRIMARY KEY, ts REAL NOT NULL, keep_id INTEGER NOT NULL REFERENCES threads(id),
    drop_id INTEGER NOT NULL REFERENCES threads(id), confidence REAL NOT NULL, merged INTEGER NOT NULL,
    reason TEXT NOT NULL, keep_title TEXT NOT NULL, drop_title TEXT NOT NULL, note TEXT NOT NULL
);
"""


def _clamp(x: Any, lo: float, hi: float, default: float) -> float:
    try:
        return min(hi, max(lo, float(x)))
    except (TypeError, ValueError):
        return default


# ====================================================================================
# 提示词
# ====================================================================================

MERGE_SYSTEM = """你是一个聊天 AI 的"潜意识"，在夜里整理记忆。用户的经历按"事件线"组织：一条事件线是跨越多天的同一件事，
例如"妈妈的健康：咳嗽 → 住院 → 出院"。白天挂线时拿不准的片段会先新建一条线，所以可能有重复的线。
现在给你一条近期新建的事件线和几条候选，判断哪些候选和它**是同一件事**，是的话会把它们合并成一条。

## 规则
- 同一件事：同一个人的同一件事、同一个计划或同一个问题的不同阶段，只是叫法或描述不同。
  例如"妈妈住院"和"母亲的肺炎"（同一个人的同一场病）；"考虑辞职"和"工作去留"。
- 不是同一件事：只是沾边或有因果关系。例如"妈妈住院"和"爸爸的体检"（不同的人）；
  "妈妈住院"和"老板不批假"（后者由前者引起，但是另一件事）；都和工作有关但说的是两回事。
- 人物不同（别名表已经把"我妈""母亲"归成同一个人，人物栏写的是常用叫法）时，几乎不可能是同一件事。
- 错误地合并比漏掉更糟：合并后很难发现、很难拆开。拿不准就不要列出，或者给低把握值。
- 只列出是同一件事的候选；都不是就给空列表。每条给 confidence（0～1 的把握）和一句话 reason。

只输出 JSON。"""

MERGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "same": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "confidence": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["id", "confidence", "reason"],
            },
        }
    },
    "required": ["same"],
}


def _card_line(c: dict[str, Any]) -> str:
    segs = "；".join(c.get("segments", [])) or "无"
    return (
        f"[{c['id']}] {c['title']}｜摘要：{c.get('summary', '')}｜片段：{segs}"
        f"｜人物：{'、'.join(c.get('people', [])) or '无'}｜新建于：{c.get('created', '')}"
    )


def build_merge_prompt(target: dict[str, Any], candidates: list[dict[str, Any]]) -> list[Message]:
    """target / candidates 都是卡片：id、标题、摘要、最近几个片段的标题、人物、新建日期。"""
    body = f"近期新建的事件线：\n{_card_line(target)}\n\n候选：\n" + (
        "\n".join(_card_line(c) for c in candidates) or "（没有）"
    )
    return [system(MERGE_SYSTEM), user(body)]


@dataclass
class MergeJudgement:
    other_id: int
    confidence: float
    reason: str


def parse_merge(data: dict[str, Any], candidate_ids: list[int]) -> list[MergeJudgement]:
    """校验模型的回答：不在候选里的 id 丢掉，重复的只留第一次，把握值收回 0～1（缺失或不是数时按 0）。
    阈值不在这里判断，由 merge_recent_threads 判断并把没过阈值的也记进日志。"""
    valid, seen, out = set(candidate_ids), set(), []
    for raw in data.get("same", []) or []:
        if not isinstance(raw, dict):
            continue
        i = raw.get("id")
        if isinstance(i, bool) or not isinstance(i, int) or i not in valid or i in seen:
            continue
        seen.add(i)
        out.append(
            MergeJudgement(i, _clamp(raw.get("confidence"), 0, 1, 0.0), str(raw.get("reason", "")).strip())
        )
    return out


# ====================================================================================
# 候选与卡片
# ====================================================================================


def _alive(store: MemoryStore, tid: int) -> bool:
    r = store.db.execute("SELECT merged_into FROM threads WHERE id = ?", (tid,)).fetchone()
    return r is not None and r[0] is None


def recent_threads(store: MemoryStore, now_ts: float, days: float = RECENT_DAYS) -> list[int]:
    """近期新建、尚未被合并的事件线，按新建时间从早到晚。"""
    since = now_ts - days * SECONDS_PER_DAY
    return [
        r[0]
        for r in store.db.execute(
            "SELECT id FROM threads WHERE merged_into IS NULL AND created_ts >= ? ORDER BY created_ts, id",
            (since,),
        )
    ]


def _thread_turn_ids(store: MemoryStore, tid: int) -> list[int]:
    return [
        r[0]
        for r in store.db.execute(
            "SELECT st.turn_id FROM segment_turns st JOIN thread_segments ts ON ts.segment_id = st.segment_id "
            "WHERE ts.thread_id = ?",
            (tid,),
        )
    ]


def _linked(store: MemoryStore, tid: int) -> list[int]:
    return [
        r[0]
        for r in store.db.execute(
            "SELECT CASE WHEN a = ? THEN b ELSE a END FROM thread_links WHERE a = ? OR b = ?", (tid, tid, tid)
        )
    ]


def find_duplicates(
    store: MemoryStore, tid: int, max_candidates: int = MAX_CANDIDATES
) -> dict[int, list[str]]:
    """可能和 tid 重复的线：{事件线 id: 命中的来源}。复用挂线候选的几种硬规则，不含"最近活跃"
    （夜间合并只找重复，最近活跃的线和这条无关时只是噪声）。常驻的线也算（重复的线可能已经常驻）。"""
    hits: dict[int, list[str]] = {}

    def add(tids, source):
        for t in tids:
            if t == tid or not _alive(store, t):
                continue
            hits.setdefault(t, [])
            if source not in hits[t]:
                hits[t].append(source)

    t = store.get_thread(tid)
    add(_linked(store, tid), "已关联")
    add(store.threads_with_people(t.person_ids), "人物重合")
    live = store.live_threads()
    docs = {x: tokenize(store.thread_text(x)) for x in live if x != tid}
    order = [x for x, _ in bm25(tokenize(store.thread_text(tid)), docs)]
    add(order[:KEYWORD_TOP], "关键词重合")
    own_turns = _thread_turn_ids(store, tid)
    for pid in t.person_ids:
        for alias in store.person(pid).aliases:
            if len(alias) >= MIN_ALIAS_LEN:
                add(store.threads_mentioning(alias, exclude_turns=own_turns), "档案人名")
    if len(hits) > max_candidates:
        # 暂定：命中来源多的优先，其次关键词得分高的优先
        rank = {x: i for i, x in enumerate(order)}
        keep = sorted(hits, key=lambda x: (-len(hits[x]), rank.get(x, len(order))))[:max_candidates]
        hits = {x: hits[x] for x in keep}
    return hits


def merge_card(store: MemoryStore, tid: int) -> dict[str, Any]:
    """在 store.thread_card 的基础上多给几个片段标题和新建日期，便于判断是不是同一件事。"""
    card = store.thread_card(tid)
    t = store.get_thread(tid)
    titles = [
        s.title if s.state == "active" else f"（只剩关键词）{s.trace}"
        for s in map(store.get_segment, t.segment_ids)
        if s.state != "archived"
    ][-SEGMENT_TITLES:]
    return {**card, "segments": titles, "created": time.strftime("%Y-%m-%d", time.localtime(t.created_ts))}


# ====================================================================================
# 合并
# ====================================================================================


@dataclass
class MergeReport:
    checked: list[int] = field(default_factory=list)  # 检查过的近期新建线
    merged: list[tuple[int, int, float]] = field(default_factory=list)  # (保留, 被合并, 把握)
    rejected: list[tuple[int, int, float]] = field(default_factory=list)  # 模型认为相同但把握不够的
    candidate_sources: dict[int, dict[int, list[str]]] = field(default_factory=dict)
    residents: ResidentReport | None = None


def _ensure_log(store: MemoryStore) -> None:
    store.db.executescript(MERGE_LOG_SCHEMA)


def _log(
    store: MemoryStore, ts: float, keep: int, drop: int, conf: float, merged: bool, reason: str, note: str
):
    titles = {
        r[0]: r[1] for r in store.db.execute("SELECT id, title FROM threads WHERE id IN (?, ?)", (keep, drop))
    }
    with store.db:
        store.db.execute(
            "INSERT INTO merge_log (ts, keep_id, drop_id, confidence, merged, reason, keep_title, drop_title, note) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (ts, keep, drop, conf, int(merged), reason, titles.get(keep, ""), titles.get(drop, ""), note),
        )


def merge_log(store: MemoryStore) -> list[dict[str, Any]]:
    _ensure_log(store)
    return [dict(r) for r in store.db.execute("SELECT * FROM merge_log ORDER BY id")]


def _choose_keep(store: MemoryStore, a: int, b: int) -> tuple[int, int]:
    """保留较早新建的那条（它通常是白天挂线时的"正主"，后建的是没把握挂上而新建的）。返回 (保留, 被合并)。"""
    ca, cb = (store.get_thread(x).created_ts for x in (a, b))
    return (a, b) if (ca, a) <= (cb, b) else (b, a)


def _merged_exempt(keep: str | None, drop: str | None) -> str | None:
    """合并后保留线的免竞争标记（D18）。
    暂定：用户解除过（空字符串）的优先，因为 D18 说免竞争只在用户要求时解除，而两条线是同一件事；
    其次保留已有的原因；都没有时为 None（之后 refresh_residents 会按条件重新判断）。"""
    if keep == "" or drop == "":
        return ""
    return keep or drop or None


def merge_pair(store: MemoryStore, model: ChatModel | None, keep: int, drop: int, now_ts: float) -> None:
    """把 drop 合并进 keep。片段与 merged_into 用 store.merge_threads；其余在这里用 SQL 补齐。"""
    k, d = store.get_thread(keep), store.get_thread(drop)
    ex = {
        r[0]: r[1]
        for r in store.db.execute("SELECT id, exempt FROM threads WHERE id IN (?, ?)", (keep, drop))
    }
    # updated_ts 取两条中较晚的，不用 now_ts：合并不是"被提及"，不该让常驻打分的活跃度变高
    store.merge_threads(keep, drop, d.updated_ts)
    with store.db:
        # 线间关联改指保留线；两条之间原有的关联去掉
        for other, link_note in store.db.execute(
            "SELECT CASE WHEN a = ? THEN b ELSE a END, note FROM thread_links WHERE a = ? OR b = ?",
            (drop, drop, drop),
        ).fetchall():
            if other != keep:
                a, b = min(keep, other), max(keep, other)
                store.db.execute("INSERT OR IGNORE INTO thread_links VALUES (?, ?, ?)", (a, b, link_note))
        store.db.execute("DELETE FROM thread_links WHERE a = ? OR b = ?", (drop, drop))
        # 新建时间取较早的；任一条还在进行就算进行中（摘要更新时模型会重新判断）
        store.db.execute(
            "UPDATE threads SET created_ts = ?, ongoing = ?, exempt = ? WHERE id = ?",
            (
                min(k.created_ts, d.created_ts),
                int(k.ongoing or d.ongoing),
                _merged_exempt(ex[keep], ex[drop]),
                keep,
            ),
        )
    # 常驻状态：被合并的线退出常驻栏；它原来常驻而保留线不在栏里时，保留线继承进入
    if d.resident_since is not None:
        store.set_resident(drop, now_ts, False, f"已合并到事件线 {keep}")
        if k.resident_since is None:
            store.set_resident(keep, now_ts, True, f"合并继承：事件线 {drop} 原在常驻栏")
    if model is not None:
        update_summary(store, model, keep)  # 两条线的片段合到一起，重新概括来龙去脉


def merge_recent_threads(
    store: MemoryStore,
    model: ChatModel,
    now_ts: float,
    days: float = RECENT_DAYS,
    threshold: float = MERGE_CONFIDENCE_MIN,
    refresh: bool = True,
) -> MergeReport:
    """夜间整理：检查近期新建的事件线，重复的合并。refresh：有合并时最后重算一次常驻栏。"""
    _ensure_log(store)
    report = MergeReport()
    for tid in recent_threads(store, now_ts, days):
        if not _alive(store, tid):  # 本轮前面已被合并掉
            continue
        report.checked.append(tid)
        hits = find_duplicates(store, tid)
        report.candidate_sources[tid] = hits
        if not hits:
            continue
        cands = [merge_card(store, x) for x in hits]
        r = model.chat(build_merge_prompt(merge_card(store, tid), cands), schema=MERGE_SCHEMA, temperature=0)
        # 把握高的先处理：同一条线和几条候选都像时，先并最像的那条
        for j in sorted(parse_merge(r.data or {}, list(hits)), key=lambda j: -j.confidence):
            if not (_alive(store, tid) and _alive(store, j.other_id)):
                continue
            keep, drop = _choose_keep(store, tid, j.other_id)
            src = json.dumps(hits[j.other_id], ensure_ascii=False)
            if j.confidence < threshold:
                _log(
                    store,
                    now_ts,
                    keep,
                    drop,
                    j.confidence,
                    False,
                    j.reason,
                    f"把握低于 {threshold:g}；来源 {src}",
                )
                report.rejected.append((keep, drop, j.confidence))
                continue
            _log(store, now_ts, keep, drop, j.confidence, True, j.reason, f"来源 {src}")
            merge_pair(store, model, keep, drop, now_ts)
            report.merged.append((keep, drop, j.confidence))
            if drop == tid:  # 这条线本身被并进了更早的线，它的其余候选留给保留线下次检查
                break
    if refresh and report.merged:
        report.residents = refresh_residents(store, now_ts)
    return report
