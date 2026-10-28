"""记忆存储：SQLite 单文件。默认放在 data/memory.db（不进版本库）。

只负责存取，不做判断：什么时候切段、挂到哪条事件线、怎么调参数，由潜意识的流程决定。
时间存 Unix 时间戳（秒）；调参日志沿用遗忘曲线的单位"天"。
"""

import json
import sqlite3
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from contracts.memory import ChatTurn, Conversation, EmotionScore, Person, Segment, Thread
from memory.forgetting import ForgettingKnobs
from memory.tuning import GlobalChange

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "memory.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY, start_ts REAL NOT NULL, end_ts REAL NOT NULL, summary TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY, conversation_id INTEGER NOT NULL REFERENCES conversations(id),
    role TEXT NOT NULL, content TEXT NOT NULL, ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS boundaries (
    turn_id INTEGER NOT NULL REFERENCES turns(id), reason TEXT NOT NULL, PRIMARY KEY (turn_id, reason)
);
CREATE TABLE IF NOT EXISTS segments (
    id INTEGER PRIMARY KEY, conversation_id INTEGER NOT NULL REFERENCES conversations(id),
    title TEXT NOT NULL, start_ts REAL NOT NULL, end_ts REAL NOT NULL,
    arousal REAL, valence REAL, importance REAL, emotion_reason TEXT,
    half_life REAL NOT NULL, last_recall_ts REAL NOT NULL, recall_count INTEGER NOT NULL,
    adjustment REAL NOT NULL, at_edge INTEGER NOT NULL, state TEXT NOT NULL, trace TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS segment_turns (
    segment_id INTEGER NOT NULL REFERENCES segments(id), turn_id INTEGER NOT NULL REFERENCES turns(id),
    PRIMARY KEY (segment_id, turn_id)
);
CREATE TABLE IF NOT EXISTS segment_people (
    segment_id INTEGER NOT NULL REFERENCES segments(id), person_id INTEGER NOT NULL REFERENCES people(id),
    PRIMARY KEY (segment_id, person_id)
);
CREATE TABLE IF NOT EXISTS segment_links (
    from_id INTEGER NOT NULL REFERENCES segments(id), to_id INTEGER NOT NULL REFERENCES segments(id),
    kind TEXT NOT NULL, PRIMARY KEY (from_id, to_id, kind)
);
CREATE TABLE IF NOT EXISTS threads (
    id INTEGER PRIMARY KEY, title TEXT NOT NULL, summary TEXT NOT NULL,
    created_ts REAL NOT NULL, updated_ts REAL NOT NULL, merged_into INTEGER REFERENCES threads(id)
);
CREATE TABLE IF NOT EXISTS thread_segments (
    thread_id INTEGER NOT NULL REFERENCES threads(id), segment_id INTEGER NOT NULL REFERENCES segments(id),
    confidence REAL NOT NULL, PRIMARY KEY (thread_id, segment_id)
);
CREATE TABLE IF NOT EXISTS thread_links (
    a INTEGER NOT NULL REFERENCES threads(id), b INTEGER NOT NULL REFERENCES threads(id),
    note TEXT NOT NULL, PRIMARY KEY (a, b)
);
CREATE TABLE IF NOT EXISTS people (id INTEGER PRIMARY KEY, name TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS aliases (
    alias TEXT PRIMARY KEY, person_id INTEGER NOT NULL REFERENCES people(id)
);
CREATE TABLE IF NOT EXISTS global_knobs (name TEXT PRIMARY KEY, value REAL NOT NULL);
CREATE TABLE IF NOT EXISTS tuning_log (
    id INTEGER PRIMARY KEY, scope TEXT NOT NULL, target TEXT NOT NULL, knob TEXT NOT NULL, at_day REAL NOT NULL,
    old REAL NOT NULL, requested REAL NOT NULL, new REAL NOT NULL, applied INTEGER NOT NULL,
    reason TEXT NOT NULL, note TEXT NOT NULL, evidence TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS turns_by_conversation ON turns(conversation_id, id);
CREATE INDEX IF NOT EXISTS segments_by_conversation ON segments(conversation_id);
CREATE INDEX IF NOT EXISTS thread_segments_by_segment ON thread_segments(segment_id);
"""


class MemoryStore:
    def __init__(self, path: str | Path = DEFAULT_PATH):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    def _insert(self, sql: str, args: tuple) -> int:
        with self.db:
            return self.db.execute(sql, args).lastrowid

    # ---------- 一场对话与消息（档案原文） ----------

    def new_conversation(self, start_ts: float) -> Conversation:
        cid = self._insert("INSERT INTO conversations (start_ts, end_ts) VALUES (?, ?)", (start_ts, start_ts))
        return Conversation(cid, start_ts, start_ts)

    def add_turn(self, conversation_id: int, role: str, content: str, ts: float) -> ChatTurn:
        with self.db:
            tid = self.db.execute(
                "INSERT INTO turns (conversation_id, role, content, ts) VALUES (?, ?, ?, ?)",
                (conversation_id, role, content, ts),
            ).lastrowid
            self.db.execute(
                "UPDATE conversations SET end_ts = MAX(end_ts, ?) WHERE id = ?", (ts, conversation_id)
            )
        return ChatTurn(tid, conversation_id, role, content, ts)

    def get_conversation(self, cid: int) -> Conversation:
        r = self.db.execute("SELECT * FROM conversations WHERE id = ?", (cid,)).fetchone()
        return Conversation(r["id"], r["start_ts"], r["end_ts"], r["summary"])

    def last_conversation(self) -> Conversation | None:
        r = self.db.execute("SELECT id FROM conversations ORDER BY end_ts DESC LIMIT 1").fetchone()
        return self.get_conversation(r["id"]) if r else None

    def set_conversation_summary(self, cid: int, summary: str) -> None:
        with self.db:
            self.db.execute("UPDATE conversations SET summary = ? WHERE id = ?", (summary, cid))

    def turns(self, conversation_id: int, after_id: int = 0) -> list[ChatTurn]:
        rows = self.db.execute(
            "SELECT * FROM turns WHERE conversation_id = ? AND id > ? ORDER BY id",
            (conversation_id, after_id),
        )
        return [ChatTurn(r["id"], r["conversation_id"], r["role"], r["content"], r["ts"]) for r in rows]

    def unsegmented_turns(self, conversation_id: int) -> list[ChatTurn]:
        """还没被切进任何片段的消息：第二层要处理的部分。"""
        rows = self.db.execute(
            "SELECT * FROM turns WHERE conversation_id = ? "
            "AND id NOT IN (SELECT turn_id FROM segment_turns) ORDER BY id",
            (conversation_id,),
        )
        return [ChatTurn(r["id"], r["conversation_id"], r["role"], r["content"], r["ts"]) for r in rows]

    # ---------- 候选边界（切段第一层的输出） ----------

    def add_boundary(self, turn_id: int, reason: str) -> None:
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO boundaries (turn_id, reason) VALUES (?, ?)", (turn_id, reason)
            )

    def boundaries(self, conversation_id: int) -> dict[int, list[str]]:
        rows = self.db.execute(
            "SELECT b.turn_id, b.reason FROM boundaries b JOIN turns t ON t.id = b.turn_id "
            "WHERE t.conversation_id = ? ORDER BY b.turn_id",
            (conversation_id,),
        )
        out: dict[int, list[str]] = {}
        for r in rows:
            out.setdefault(r["turn_id"], []).append(r["reason"])
        return out

    # ---------- 片段 ----------

    def add_segment(self, seg: Segment) -> Segment:
        e = seg.emotion
        with self.db:
            sid = self.db.execute(
                "INSERT INTO segments (conversation_id, title, start_ts, end_ts, arousal, valence, importance, "
                "emotion_reason, half_life, last_recall_ts, recall_count, adjustment, at_edge, state, trace) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    seg.conversation_id,
                    seg.title,
                    seg.start_ts,
                    seg.end_ts,
                    e and e.arousal,
                    e and e.valence,
                    e and e.importance,
                    e and e.reason,
                    seg.half_life,
                    seg.last_recall_ts,
                    seg.recall_count,
                    seg.adjustment,
                    int(seg.at_edge),
                    seg.state,
                    seg.trace,
                ),
            ).lastrowid
            self.db.executemany(
                "INSERT INTO segment_turns VALUES (?, ?)", [(sid, t) for t in seg.message_ids]
            )
            self.db.executemany(
                "INSERT INTO segment_people VALUES (?, ?)", [(sid, p) for p in seg.person_ids]
            )
        seg.id = sid
        return seg

    def get_segment(self, sid: int) -> Segment:
        r = self.db.execute("SELECT * FROM segments WHERE id = ?", (sid,)).fetchone()
        if r is None:
            raise KeyError(f"没有片段 {sid}")
        emotion = None
        if r["arousal"] is not None:
            emotion = EmotionScore(r["arousal"], r["valence"], r["importance"], r["emotion_reason"] or "")
        turns = [
            x[0]
            for x in self.db.execute(
                "SELECT turn_id FROM segment_turns WHERE segment_id = ? ORDER BY turn_id", (sid,)
            )
        ]
        people = [
            x[0] for x in self.db.execute("SELECT person_id FROM segment_people WHERE segment_id = ?", (sid,))
        ]
        return Segment(
            r["id"],
            r["conversation_id"],
            r["title"],
            r["start_ts"],
            r["end_ts"],
            turns,
            people,
            emotion,
            r["half_life"],
            r["last_recall_ts"],
            r["recall_count"],
            r["adjustment"],
            bool(r["at_edge"]),
            r["state"],
            r["trace"],
        )

    def save_segment_state(self, seg: Segment) -> None:
        """保存遗忘曲线相关的字段（半衰期、回忆、调整、状态、残影）。"""
        with self.db:
            self.db.execute(
                "UPDATE segments SET half_life = ?, last_recall_ts = ?, recall_count = ?, adjustment = ?, "
                "at_edge = ?, state = ?, trace = ? WHERE id = ?",
                (
                    seg.half_life,
                    seg.last_recall_ts,
                    seg.recall_count,
                    seg.adjustment,
                    int(seg.at_edge),
                    seg.state,
                    seg.trace,
                    seg.id,
                ),
            )

    def segments(self, conversation_id: int | None = None, state: str | None = None) -> list[Segment]:
        sql, args = "SELECT id FROM segments WHERE 1 = 1", []
        if conversation_id is not None:
            sql, args = sql + " AND conversation_id = ?", [*args, conversation_id]
        if state is not None:
            sql, args = sql + " AND state = ?", [*args, state]
        return [self.get_segment(r["id"]) for r in self.db.execute(sql + " ORDER BY start_ts, id", args)]

    def link_segments(self, from_id: int, to_id: int, kind: str = "next") -> None:
        """片段之间的关系。kind="next" 表示"接着聊到"；话题交错时可以有多条，是一张图。"""
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO segment_links VALUES (?, ?, ?)", (from_id, to_id, kind))

    def segment_links(self, conversation_id: int) -> list[tuple[int, int, str]]:
        rows = self.db.execute(
            "SELECT l.* FROM segment_links l JOIN segments s ON s.id = l.from_id WHERE s.conversation_id = ?",
            (conversation_id,),
        )
        return [(r["from_id"], r["to_id"], r["kind"]) for r in rows]

    # ---------- 事件线 ----------

    def new_thread(self, title: str, summary: str, ts: float) -> Thread:
        tid = self._insert(
            "INSERT INTO threads (title, summary, created_ts, updated_ts) VALUES (?, ?, ?, ?)",
            (title, summary, ts, ts),
        )
        return Thread(tid, title, summary, ts, ts)

    def attach(self, thread_id: int, segment_id: int, confidence: float, ts: float) -> None:
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO thread_segments VALUES (?, ?, ?)", (thread_id, segment_id, confidence)
            )
            self.db.execute(
                "UPDATE threads SET updated_ts = MAX(updated_ts, ?) WHERE id = ?", (ts, thread_id)
            )

    def get_thread(self, tid: int) -> Thread:
        r = self.db.execute("SELECT * FROM threads WHERE id = ?", (tid,)).fetchone()
        if r is None:
            raise KeyError(f"没有事件线 {tid}")
        segs = [
            x[0]
            for x in self.db.execute(
                "SELECT ts.segment_id FROM thread_segments ts JOIN segments s ON s.id = ts.segment_id "
                "WHERE ts.thread_id = ? ORDER BY s.start_ts",
                (tid,),
            )
        ]
        people = [
            x[0]
            for x in self.db.execute(
                "SELECT DISTINCT sp.person_id FROM segment_people sp JOIN thread_segments ts "
                "ON ts.segment_id = sp.segment_id WHERE ts.thread_id = ? ORDER BY sp.person_id",
                (tid,),
            )
        ]
        return Thread(
            r["id"],
            r["title"],
            r["summary"],
            r["created_ts"],
            r["updated_ts"],
            segs,
            people,
            r["merged_into"],
        )

    def update_thread(self, tid: int, *, title: str | None = None, summary: str | None = None) -> None:
        with self.db:
            if title is not None:
                self.db.execute("UPDATE threads SET title = ? WHERE id = ?", (title, tid))
            if summary is not None:
                self.db.execute("UPDATE threads SET summary = ? WHERE id = ?", (summary, tid))

    def threads_of_segment(self, segment_id: int) -> list[int]:
        return [
            r[0]
            for r in self.db.execute(
                "SELECT thread_id FROM thread_segments WHERE segment_id = ?", (segment_id,)
            )
        ]

    def relate_threads(self, a: int, b: int, note: str) -> None:
        """两条事件线"有关但不是同一件事"，例如漂移路径把它们连在一起。"""
        a, b = min(a, b), max(a, b)
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO thread_links VALUES (?, ?, ?)", (a, b, note))

    def merge_threads(self, keep: int, drop: int, ts: float) -> None:
        """夜间合并：把 drop 的片段移到 keep，drop 标记为已合并（不删除，便于追溯）。"""
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO thread_segments SELECT ?, segment_id, confidence "
                "FROM thread_segments WHERE thread_id = ?",
                (keep, drop),
            )
            self.db.execute("DELETE FROM thread_segments WHERE thread_id = ?", (drop,))
            self.db.execute("UPDATE threads SET merged_into = ? WHERE id = ?", (keep, drop))
            self.db.execute("UPDATE threads SET updated_ts = MAX(updated_ts, ?) WHERE id = ?", (ts, keep))

    def live_threads(self) -> list[int]:
        return [
            r[0]
            for r in self.db.execute(
                "SELECT id FROM threads WHERE merged_into IS NULL ORDER BY updated_ts DESC"
            )
        ]

    def thread_card(self, tid: int) -> dict[str, Any]:
        """给潜意识模型看的事件线卡片：只含标题、一句话摘要、最近片段标题、涉及人物。"""
        t = self.get_thread(tid)
        latest = self.get_segment(t.segment_ids[-1]).title if t.segment_ids else ""
        return {
            "id": t.id,
            "title": t.title,
            "summary": t.summary,
            "latest_segment": latest,
            "people": [self.person(p).name for p in t.person_ids],
        }

    # ---------- 人物别名表 ----------

    def resolve(self, name: str) -> int | None:
        r = self.db.execute("SELECT person_id FROM aliases WHERE alias = ?", (name.strip(),)).fetchone()
        return r[0] if r else None

    def upsert_person(self, name: str, aliases: list[str] = ()) -> int:
        """登记一个人和他的叫法。任何一个叫法已经认识，就并到那个人；否则新建。"""
        names = [n.strip() for n in [name, *aliases] if n and n.strip()]
        known = {pid for n in names if (pid := self.resolve(n)) is not None}
        with self.db:
            if known:
                pid = min(known)
                for other in known - {pid}:  # 两个人其实是同一个：合并
                    self.db.execute("UPDATE aliases SET person_id = ? WHERE person_id = ?", (pid, other))
                    self.db.execute(
                        "UPDATE segment_people SET person_id = ? WHERE person_id = ? "
                        "AND segment_id NOT IN (SELECT segment_id FROM segment_people WHERE person_id = ?)",
                        (pid, other, pid),
                    )
                    self.db.execute("DELETE FROM segment_people WHERE person_id = ?", (other,))
            else:
                pid = self.db.execute("INSERT INTO people (name) VALUES (?)", (names[0],)).lastrowid
            self.db.executemany("INSERT OR IGNORE INTO aliases VALUES (?, ?)", [(n, pid) for n in names])
        return pid

    def person(self, pid: int) -> Person:
        r = self.db.execute("SELECT * FROM people WHERE id = ?", (pid,)).fetchone()
        aliases = [
            x[0]
            for x in self.db.execute("SELECT alias FROM aliases WHERE person_id = ? ORDER BY alias", (pid,))
        ]
        return Person(r["id"], r["name"], aliases)

    # ---------- 全局参数与调参日志 ----------

    def load_global_knobs(self) -> ForgettingKnobs:
        saved = {r["name"]: r["value"] for r in self.db.execute("SELECT * FROM global_knobs")}
        valid = {f.name for f in fields(ForgettingKnobs)}
        return ForgettingKnobs(**{k: v for k, v in saved.items() if k in valid})

    def save_global_knobs(self, knobs: ForgettingKnobs) -> None:
        with self.db:
            self.db.executemany(
                "INSERT OR REPLACE INTO global_knobs VALUES (?, ?)", list(asdict(knobs).items())
            )

    def log_tuning(
        self, scope: str, target: str, change: GlobalChange, evidence: dict[str, Any] | None = None
    ) -> None:
        """记一条调参日志。scope："memory"（单条记忆）或 "global"；被拒绝的请求也记。"""
        with self.db:
            self.db.execute(
                "INSERT INTO tuning_log (scope, target, knob, at_day, old, requested, new, applied, reason, note, "
                "evidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    scope,
                    target,
                    change.knob,
                    change.at,
                    change.old,
                    change.requested,
                    change.new,
                    int(change.applied),
                    change.reason,
                    change.note,
                    json.dumps(evidence or {}, ensure_ascii=False),
                ),
            )

    def tuning_log(self, scope: str | None = None) -> list[tuple[str, str, GlobalChange]]:
        sql, args = "SELECT * FROM tuning_log", ()
        if scope is not None:
            sql, args = sql + " WHERE scope = ?", (scope,)
        return [
            (
                r["scope"],
                r["target"],
                GlobalChange(
                    r["knob"],
                    r["at_day"],
                    r["old"],
                    r["requested"],
                    r["new"],
                    bool(r["applied"]),
                    r["reason"],
                    r["note"],
                ),
            )
            for r in self.db.execute(sql + " ORDER BY id", args)
        ]
