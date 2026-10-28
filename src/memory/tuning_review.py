"""调参日志的查看与回滚（待办 #14）。

潜意识.md："每次调整都写日志……日志可以对照档案原文复查，也可以回滚。"
- 查看：list_tuning 返回结构化条目（附可读说明），format_entries 把它排成中文文本。
- 回滚：rollback 撤销某一条日志的那一次调整，回滚本身也记一条日志（理由"回滚第 N 条"）。

命令行（需要 src 在 sys.path 里，例如在项目根目录执行 `$env:PYTHONPATH="src"` 后）：
    python -m memory.tuning_review [--scope memory|global] [--rollback ID]

本模块只读写 store 已有的表，store 没有的查询直接用 store.db 写 SQL。
"""

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from contracts.memory import SECONDS_PER_DAY, days
from memory.store import DEFAULT_PATH, MemoryStore
from memory.subconscious import is_finished
from memory.tuning import DEFAULT_KNOBS, DEFAULT_LIMITS, TUNABLE, GlobalChange, GlobalLimits

KNOB_LABELS = {
    "adjustment": "调整系数",
    "base_half_life": "基础半衰期",
    "emotion_max": "情绪放大",
    "importance_max": "重要性放大",
    "reinforce": "回忆加固",
    "position_bonus": "首尾加成",
    "forget_line": "遗忘线",
}

ROLLBACK_REASON = "回滚第 {} 条"
_ROLLBACK_RE = re.compile(r"^回滚第 (\d+) 条$")


@dataclass
class TuningEntry:
    """一条调参日志，外加便于人读的信息。"""

    id: int
    scope: str  # "memory" 或 "global"
    target: str  # 单条记忆：片段 id；全局：参数名
    knob: str
    at_day: float
    old: float
    requested: float
    new: float
    applied: bool
    reason: str
    note: str
    evidence: dict[str, Any]
    target_title: str = ""  # 单条记忆：片段标题；全局：参数中文名
    evidence_title: str = ""  # 依据的片段标题
    rollback_of: int | None = None  # 本条是回滚哪一条
    rolled_back_by: list[int] = field(default_factory=list)  # 本条被哪些条回滚过
    superseded_by: list[int] = field(default_factory=list)  # 之后对同一目标生效的调整（含回滚）

    @property
    def when(self) -> datetime:
        return datetime.fromtimestamp(self.at_day * SECONDS_PER_DAY)

    @property
    def description(self) -> str:
        """一句话说明：哪条记忆 / 哪个参数，从多少到多少，理由，依据。"""
        if self.scope == "memory":
            what = f"记忆 #{self.target}「{self.target_title or '（片段不存在）'}」的调整系数"
        else:
            what = f"全局参数「{self.target_title}」({self.knob})"
        if self.applied:
            change = f"从 {self.old:.4g} 调到 {self.new:.4g}（×{self.new / self.old:.3g}）"
        else:
            change = f"请求从 {self.old:.4g} 调到 {self.requested:.4g}，未生效"
        parts = [f"{what}{change}", f"理由：{self.reason or '（无）'}"]
        if self.note:
            parts.append(f"护栏：{self.note}")
        if self.evidence_title:
            parts.append(f"依据片段 #{self.evidence.get('segment')}「{self.evidence_title}」")
        if self.rolled_back_by:
            parts.append("已被第 " + "、".join(map(str, self.rolled_back_by)) + " 条回滚")
        elif self.superseded_by:
            parts.append("之后又被第 " + "、".join(map(str, self.superseded_by)) + " 条调整过")
        return "；".join(parts)


def _segment_title(store: MemoryStore, sid: Any) -> str:
    try:
        r = store.db.execute("SELECT title FROM segments WHERE id = ?", (int(sid),)).fetchone()
    except (TypeError, ValueError):
        return ""
    return r["title"] if r else ""


def _all_entries(store: MemoryStore) -> list[TuningEntry]:
    """读出全部日志并补上标题、回滚关系、覆盖关系。"""
    rows = store.db.execute("SELECT * FROM tuning_log ORDER BY id").fetchall()
    entries = []
    for r in rows:
        evidence = json.loads(r["evidence"] or "{}")
        m = _ROLLBACK_RE.match(r["reason"])
        e = TuningEntry(
            r["id"],
            r["scope"],
            r["target"],
            r["knob"],
            r["at_day"],
            r["old"],
            r["requested"],
            r["new"],
            bool(r["applied"]),
            r["reason"],
            r["note"],
            evidence,
            rollback_of=int(m.group(1)) if m else evidence.get("rollback_of"),
        )
        if e.scope == "memory":
            e.target_title = _segment_title(store, e.target)
        else:
            e.target_title = KNOB_LABELS.get(e.knob, e.knob)
        if "segment" in evidence:
            e.evidence_title = _segment_title(store, evidence["segment"])
        entries.append(e)
    by_id = {e.id: e for e in entries}
    for e in entries:
        if e.rollback_of is not None and e.applied and e.rollback_of in by_id:
            by_id[e.rollback_of].rolled_back_by.append(e.id)
        if e.applied:
            for earlier in entries:
                if (
                    earlier.id < e.id
                    and earlier.applied
                    and (earlier.scope, earlier.target, earlier.knob) == (e.scope, e.target, e.knob)
                ):
                    earlier.superseded_by.append(e.id)
    return entries


def list_tuning(
    store: MemoryStore,
    scope: str | None = None,
    since_ts: float | None = None,
    until_ts: float | None = None,
    applied: bool | None = None,
    target: str | None = None,
) -> list[TuningEntry]:
    """列出调参日志，按 id 排序。时间用 Unix 时间戳（秒），与 store 其他地方一致；日志本身存的是"天"。"""
    out = []
    for e in _all_entries(store):
        if scope is not None and e.scope != scope:
            continue
        if since_ts is not None and e.at_day < days(since_ts):
            continue
        if until_ts is not None and e.at_day > days(until_ts):
            continue
        if applied is not None and e.applied != applied:
            continue
        if target is not None and e.target != str(target):
            continue
        out.append(e)
    return out


def format_entries(entries: list[TuningEntry]) -> str:
    """排成中文文本，一条一段。"""
    if not entries:
        return "（没有调参日志）"
    scope_label = {"memory": "单条记忆", "global": "全局"}
    lines = []
    for e in entries:
        mark = "生效" if e.applied else "未生效"
        head = f"第 {e.id} 条 [{scope_label.get(e.scope, e.scope)}·{mark}] {e.when:%Y-%m-%d %H:%M}"
        lines.append(head)
        lines.append("  " + e.description)
    return "\n".join(lines)


@dataclass
class RollbackResult:
    entry: TuningEntry  # 被回滚的那条
    change: GlobalChange  # 回滚本身的日志内容
    log_id: int  # 回滚日志的 id
    message: str  # 给人看的说明


def _limit_note(notes: list[str]) -> str:
    return "；".join(n for n in notes if n)


def rollback(
    store: MemoryStore,
    entry_id: int,
    now_ts: float,
    limits: GlobalLimits = DEFAULT_LIMITS,
    defaults=DEFAULT_KNOBS,
) -> RollbackResult:
    """撤销第 entry_id 条日志的那一次调整，并记一条日志（理由"回滚第 N 条"）。

    - 没有被后续调整覆盖：直接改回原值（old）。
    - 已经被后续调整覆盖：按"撤销这一次的倍数"处理，即当前值 × (old / new)，并在说明里写明。
    - 单条记忆：调整系数和半衰期按同一比例改回；全局：改参数值。
    全局回滚不受"同一参数间隔 7 天"的冷却限制，但仍限制在默认值的 1/1.5～1.5 倍之间。
    """
    by_id = {e.id: e for e in _all_entries(store)}
    if entry_id not in by_id:
        raise ValueError(f"没有第 {entry_id} 条调参日志")
    entry = by_id[entry_id]
    if not entry.applied:
        raise ValueError(f"第 {entry_id} 条没有生效过，无需回滚")
    # 暂定：已经回滚过的不再重复回滚，避免误操作撤销两次；要恢复可以回滚那条回滚日志。
    if entry.rolled_back_by:
        raise ValueError(f"第 {entry_id} 条已被第 {entry.rolled_back_by[0]} 条回滚过")

    now = days(now_ts)
    undo = entry.old / entry.new
    notes = []
    if entry.superseded_by:
        notes.append(
            "已被第 " + "、".join(map(str, entry.superseded_by)) + " 条后续调整覆盖，按撤销这一次的倍数"
            f"（×{undo:.3g}）处理"
        )

    if entry.scope == "memory":
        seg = store.get_segment(int(entry.target))
        knobs = store.load_global_knobs()
        current = seg.adjustment
        target_value = current * undo if entry.superseded_by else entry.old
        # 暂定：回滚后的系数仍要落在单条记忆累积调整的范围内（1/adjust_total_max～adjust_total_max）。
        lo, hi = 1.0 / knobs.adjust_total_max, knobs.adjust_total_max
        new = min(hi, max(lo, target_value))
        if abs(new - target_value) > 1e-12:
            notes.append("已到单条记忆调整范围的边界")
        # 暂定：回滚同样遵守 D19——事件未完结时系数不能调到 1 以下（原本已低于 1 的不再往下调）。
        floor = min(1.0, current)
        if new < floor and not is_finished(store, seg.id):
            new = floor
            notes.append("事件未完结，不允许系数小于 1")
        applied = abs(new - current) > 1e-12
        if applied:
            seg.half_life *= new / current
            seg.adjustment = new
            store.save_segment_state(seg)
        else:
            new = current
            notes.append("没有变化")
        change = GlobalChange(
            "adjustment",
            now,
            current,
            target_value,
            new,
            applied,
            ROLLBACK_REASON.format(entry_id),
            _limit_note(notes),
        )
    else:
        if entry.knob not in TUNABLE:
            raise ValueError(f"{entry.knob} 不是可调的全局参数")
        knobs = store.load_global_knobs()
        current = getattr(knobs, entry.knob)
        target_value = current * undo if entry.superseded_by else entry.old
        default = getattr(defaults, entry.knob)
        # 不受冷却限制，但范围限制照旧
        new = min(default * limits.range_max, max(default / limits.range_max, target_value))
        if abs(new - target_value) > 1e-12:
            notes.append("已到允许范围的边界")
        applied = abs(new - current) > 1e-12
        if applied:
            store.save_global_knobs(replace(knobs, **{entry.knob: new}))
        else:
            new = current
            notes.append("没有变化")
        change = GlobalChange(
            entry.knob,
            now,
            current,
            target_value,
            new,
            applied,
            ROLLBACK_REASON.format(entry_id),
            _limit_note(notes),
        )

    store.log_tuning(entry.scope, entry.target, change, {"rollback_of": entry_id})
    log_id = store.db.execute("SELECT MAX(id) FROM tuning_log").fetchone()[0]
    what = entry.target_title if entry.scope == "global" else f"记忆 #{entry.target}「{entry.target_title}」"
    if applied:
        msg = f"已回滚第 {entry_id} 条：{what} 从 {change.old:.4g} 改为 {change.new:.4g}"
    else:
        msg = f"第 {entry_id} 条回滚后没有变化：{what} 仍为 {change.old:.4g}"
    if change.note:
        msg += f"（{change.note}）"
    return RollbackResult(entry, change, log_id, msg + f"。已记为第 {log_id} 条日志。")


def _parse_date(s: str) -> float:
    return datetime.strptime(s, "%Y-%m-%d").timestamp()


def main(argv: list[str] | None = None) -> int:
    # 输出被管道或重定向时，Windows 默认编码可能不是 UTF-8，中文会报错
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(prog="python -m memory.tuning_review", description="查看与回滚调参日志")
    p.add_argument("--db", default=str(DEFAULT_PATH), help="记忆库路径，默认 data/memory.db")
    p.add_argument("--scope", choices=["memory", "global"], help="只看单条记忆或全局")
    p.add_argument("--since", help="起始日期 YYYY-MM-DD")
    p.add_argument("--until", help="截止日期 YYYY-MM-DD（含当天）")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--applied", action="store_true", help="只看生效过的")
    g.add_argument("--rejected", action="store_true", help="只看未生效的")
    p.add_argument("--rollback", type=int, metavar="ID", help="回滚第 ID 条日志")
    args = p.parse_args(argv)

    if not Path(args.db).exists():
        print(f"找不到记忆库：{args.db}")
        return 1
    store = MemoryStore(args.db)
    try:
        if args.rollback is not None:
            try:
                result = rollback(store, args.rollback, time.time())
            except (ValueError, KeyError) as exc:
                print(f"回滚失败：{exc}")
                return 1
            print(result.message)
            return 0
        applied = True if args.applied else False if args.rejected else None
        entries = list_tuning(
            store,
            scope=args.scope,
            since_ts=_parse_date(args.since) if args.since else None,
            until_ts=_parse_date(args.until) + SECONDS_PER_DAY if args.until else None,
            applied=applied,
        )
        print(format_entries(entries))
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
