"""可视化页面用的只读 JSON 接口，由 serve.py 调用。只用标准库，不写任何文件。

- /api/memory：以只读方式（sqlite URI 的 mode=ro）打开项目根目录 data/memory.db，返回片段、事件线、
  常驻栏、用户画像等的统计和列表。数据库不存在时返回 exists=false。
  环境变量 MEMORY_DB 可以指向别的数据库文件（例如测试用的副本）。
- /api/docs：总纲里几份文档的原文（决策日志、偏离记录、待办、待你确认、接口约定），
  加上测试数量（pytest --collect-only，缓存 60 秒）和最近的提交记录，给「全部工作」页实时解析。
"""

import os
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
DB_PATH = Path(os.environ.get("MEMORY_DB") or PROJECT / "data" / "memory.db")
DOCS = ["决策日志", "偏离记录", "待办", "待你确认", "接口约定"]
PYTHON = PROJECT / ".venv" / "Scripts" / "python.exe"
FORGET_LINE = 0.1  # 与 src/memory/forgetting.py 的默认遗忘线一致；数据库里有 global_knobs 时以它为准

if str(PROJECT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT / "src"))


# ────────────────────────────── /api/memory ──────────────────────────────


def _connect() -> sqlite3.Connection:
    uri = "file:" + quote(DB_PATH.as_posix(), safe="/:") + "?mode=ro"
    db = sqlite3.connect(uri, uri=True, timeout=2)
    db.row_factory = sqlite3.Row
    return db


def _tables(db) -> set[str]:
    return {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def _cols(db, table: str) -> set[str]:
    return {r[1] for r in db.execute(f"PRAGMA table_info({table})")}


def _rows(db, sql: str, args=()) -> list[dict]:
    return [dict(r) for r in db.execute(sql, args)]


def memory_snapshot() -> dict:
    now = time.time()
    out: dict = {"db": str(DB_PATH), "now": now, "exists": DB_PATH.is_file()}
    if not out["exists"]:
        out["note"] = "还没有记忆数据库：记忆模块第一次存消息时才会创建 data/memory.db。"
        return out
    try:
        db = _connect()
    except sqlite3.Error as e:
        out["error"] = f"打不开数据库：{e}"
        return out
    try:
        t = _tables(db)
        count = lambda table, where="": (  # noqa: E731
            db.execute(f"SELECT COUNT(*) FROM {table} {where}").fetchone()[0] if table in t else None
        )
        th_cols = _cols(db, "threads") if "threads" in t else set()
        live_where = "WHERE merged_into IS NULL" + (" AND forgotten_ts IS NULL" if "forgotten_ts" in th_cols else "")

        knobs = {r["name"]: r["value"] for r in _rows(db, "SELECT name, value FROM global_knobs")} if "global_knobs" in t else {}
        forget_line = knobs.get("forget_line", FORGET_LINE)

        # 记忆时钟：用全部消息时间和任务时段算"记忆时间"（D21）
        activity = [(r[0], r[0]) for r in db.execute("SELECT ts FROM turns")] if "turns" in t else []
        if "busy_spans" in t:
            activity += [(r[0], r[1]) for r in db.execute("SELECT start_ts, end_ts FROM busy_spans")]
        try:
            from memory.clock import MemoryClock

            clock = MemoryClock(activity)
            elapsed = lambda a, b: clock.elapsed_days(a, b)  # noqa: E731
            out["clock"] = "记忆时钟（D21）"
        except Exception as e:  # 记忆模块的代码正在被别的对话修改时，退回日历时间
            elapsed = lambda a, b: max(0.0, b - a) / 86400  # noqa: E731
            out["clock"] = f"日历时间（记忆时钟导入失败：{e}）"

        resident_tids = (
            [r[0] for r in db.execute(f"SELECT id FROM threads {live_where} AND resident_since IS NOT NULL")]
            if "resident_since" in th_cols else []
        )
        resident_sids = set()
        if resident_tids and "thread_segments" in t:
            marks = ",".join("?" * len(resident_tids))
            resident_sids = {r[0] for r in db.execute(
                f"SELECT segment_id FROM thread_segments WHERE thread_id IN ({marks})", resident_tids)}

        seg_thread = {}
        if "thread_segments" in t:
            for r in db.execute("SELECT segment_id, thread_id FROM thread_segments"):
                seg_thread.setdefault(r[0], []).append(r[1])

        segments = []
        if "segments" in t:
            for r in _rows(db, "SELECT * FROM segments ORDER BY start_ts"):
                st = r.get("state", "active")
                if st == "archived":
                    s = None
                elif r["id"] in resident_sids:
                    s = 1.0
                else:
                    s = 0.5 ** (elapsed(r["last_recall_ts"], now) / r["half_life"]) if r["half_life"] else 0.0
                segments.append({
                    "id": r["id"], "conversation": r["conversation_id"], "title": r["title"],
                    "start_ts": r["start_ts"], "state": st, "trace": r.get("trace", ""),
                    "arousal": r.get("arousal"), "valence": r.get("valence"), "importance": r.get("importance"),
                    "half_life": r["half_life"], "adjustment": r.get("adjustment"), "recall_count": r.get("recall_count"),
                    "at_edge": bool(r.get("at_edge")), "remember": bool(r.get("remember_requested")),
                    "strength": s, "resident": r["id"] in resident_sids, "threads": seg_thread.get(r["id"], []),
                    "below_line": s is not None and s < forget_line,
                })

        threads = []
        if "threads" in t:
            exempt = "exempt" if "exempt" in th_cols else "NULL AS exempt"
            ongoing = "ongoing" if "ongoing" in th_cols else "1 AS ongoing"
            res = "resident_since" if "resident_since" in th_cols else "NULL AS resident_since"
            for r in _rows(db, f"SELECT id, title, summary, created_ts, updated_ts, {ongoing}, {res}, {exempt} "
                               f"FROM threads {live_where} ORDER BY updated_ts DESC"):
                r["segments"] = [s["id"] for s in segments if r["id"] in s["threads"]]
                threads.append(r)

        profile = []
        if "profile_facts" in t:
            people = {r[0]: r[1] for r in db.execute("SELECT id, name FROM people")} if "people" in t else {}
            for r in _rows(db, "SELECT id, person_id, category, fact, mentions, archived_ts FROM profile_facts ORDER BY id"):
                r["person"] = "用户本人" if r["person_id"] is None else people.get(r["person_id"], f"#{r['person_id']}")
                profile.append(r)

        states = {}
        for s in segments:
            states[s["state"]] = states.get(s["state"], 0) + 1

        out.update({
            "counts": {
                "conversations": count("conversations"), "turns": count("turns"),
                "segments": len(segments), "segment_states": states,
                "below_line": sum(1 for s in segments if s["below_line"] and s["state"] == "active"),
                "threads": len(threads), "threads_merged": count("threads", "WHERE merged_into IS NOT NULL"),
                "threads_forgotten": count("threads", "WHERE forgotten_ts IS NOT NULL") if "forgotten_ts" in th_cols else None,
                "resident": len(resident_tids), "exempt": sum(1 for x in threads if x.get("exempt")),
                "profile": sum(1 for f in profile if not f["archived_ts"]),
                "profile_archived": sum(1 for f in profile if f["archived_ts"]),
                "people": count("people"), "tuning_log": count("tuning_log"),
                "tuning_applied": count("tuning_log", "WHERE applied = 1"),
                "sweeps": count("sweep_log"), "erased": count("erase_log"),
                "merges": count("merge_log"), "evocations_pending": count("evocations", "WHERE consumed = 0"),
            },
            "forget_line": forget_line, "knobs": knobs,
            "meta": {r["key"]: r["value"] for r in _rows(db, "SELECT key, value FROM meta")} if "meta" in t else {},
            "segments": segments, "threads": threads, "profile": profile,
            "resident_log": _rows(db, "SELECT * FROM resident_log ORDER BY id DESC LIMIT 30") if "resident_log" in t else [],
            "tuning_log": _rows(db, "SELECT id, scope, target, knob, old, requested, new, applied, reason, note "
                                    "FROM tuning_log ORDER BY id DESC LIMIT 30") if "tuning_log" in t else [],
        })
    except sqlite3.Error as e:
        out["error"] = f"读取失败：{e}"
    finally:
        db.close()
    return out


# ────────────────────────────── /api/docs ──────────────────────────────

_tests_cache: dict = {"at": 0.0, "data": None}


def _collect(marker: str) -> dict:
    """pytest --collect-only，返回每个文件收集到的测试数。只收集，不运行。"""
    try:
        p = subprocess.run(
            [str(PYTHON if PYTHON.is_file() else sys.executable), "-m", "pytest", "--collect-only", "-q",
             "-p", "no:cacheprovider", "-m", marker],
            cwd=PROJECT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"error": str(e)}
    files: dict[str, int] = {}
    for line in p.stdout.splitlines():
        if "::" in line:
            f = line.split("::", 1)[0]
            files[f] = files.get(f, 0) + 1
    m = re.search(r"(\d+)(?:/\d+)? tests? collected", p.stdout)
    return {"files": files, "total": sum(files.values()), "summary": (p.stdout.strip().splitlines() or [""])[-1],
            "ok": p.returncode in (0, 5), "collected": int(m.group(1)) if m else None}


def tests_snapshot() -> dict:
    if _tests_cache["data"] is None or time.time() - _tests_cache["at"] > 60:
        _tests_cache["data"] = {"offline": _collect("not live"), "live": _collect("live")}
        _tests_cache["at"] = time.time()
    return _tests_cache["data"]


def git_log(n: int = 80) -> list[dict]:
    try:
        p = subprocess.run(
            ["git", "log", f"-n{n}", "--date=format:%Y-%m-%d %H:%M", "--format=%h%x1f%ad%x1f%s"],
            cwd=PROJECT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [dict(zip(("hash", "date", "subject"), line.split("\x1f"))) for line in p.stdout.splitlines() if line]


def docs_snapshot() -> dict:
    docs = {}
    for name in DOCS:
        f = PROJECT / "00-总纲" / f"{name}.md"
        docs[name] = {"text": f.read_text(encoding="utf-8"), "mtime": f.stat().st_mtime} if f.is_file() else None
    return {"docs": docs, "tests": tests_snapshot(), "git": git_log(), "now": time.time()}
