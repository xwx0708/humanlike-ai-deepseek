"""记忆存储：SQLite 单文件。默认放在 data/memory.db（不进版本库）。

只负责存取，不做判断：什么时候切段、挂到哪条事件线、怎么调参数，由潜意识的流程决定。
时间存 Unix 时间戳（秒）；调参日志沿用遗忘曲线的单位"天"。
"""

import json
import re
import sqlite3
from collections.abc import Iterator
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from contracts.memory import ChatTurn, Conversation, EmotionScore, Person, ProfileFact, Segment, Thread
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
    adjustment REAL NOT NULL, at_edge INTEGER NOT NULL, state TEXT NOT NULL, trace TEXT NOT NULL,
    remember_requested INTEGER NOT NULL DEFAULT 0
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
    created_ts REAL NOT NULL, updated_ts REAL NOT NULL, merged_into INTEGER REFERENCES threads(id),
    ongoing INTEGER NOT NULL DEFAULT 1, resident_since REAL, exempt TEXT, forgotten_ts REAL
);
CREATE TABLE IF NOT EXISTS sweep_log (
    id INTEGER PRIMARY KEY, ts REAL NOT NULL, archived TEXT NOT NULL, kept TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS busy_spans (
    id INTEGER PRIMARY KEY, start_ts REAL NOT NULL, end_ts REAL NOT NULL, note TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS resident_log (
    id INTEGER PRIMARY KEY, thread_id INTEGER NOT NULL REFERENCES threads(id), ts REAL NOT NULL,
    action TEXT NOT NULL, reason TEXT NOT NULL, score REAL
);
CREATE TABLE IF NOT EXISTS thread_segments (
    thread_id INTEGER NOT NULL REFERENCES threads(id), segment_id INTEGER NOT NULL REFERENCES segments(id),
    confidence REAL NOT NULL, PRIMARY KEY (thread_id, segment_id)
);
CREATE TABLE IF NOT EXISTS thread_links (
    a INTEGER NOT NULL REFERENCES threads(id), b INTEGER NOT NULL REFERENCES threads(id),
    note TEXT NOT NULL, PRIMARY KEY (a, b)
);
CREATE TABLE IF NOT EXISTS erase_log (
    id INTEGER PRIMARY KEY, ts REAL NOT NULL, segments INTEGER NOT NULL, turns INTEGER NOT NULL, reason TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
-- 回应模块用关键词认出"记住"的消息（D29）；整理时含这些消息的片段算用户要求记住
CREATE TABLE IF NOT EXISTS remember_marks (turn_id INTEGER PRIMARY KEY);
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
CREATE TABLE IF NOT EXISTS profile_facts (
    id INTEGER PRIMARY KEY, person_id INTEGER REFERENCES people(id), category TEXT NOT NULL,
    fact TEXT NOT NULL, fact_key TEXT NOT NULL, first_ts REAL NOT NULL, last_ts REAL NOT NULL,
    mentions INTEGER NOT NULL, archived_ts REAL, superseded_by INTEGER REFERENCES profile_facts(id)
);
CREATE INDEX IF NOT EXISTS profile_by_key ON profile_facts(IFNULL(person_id, 0), category, fact_key);
CREATE TABLE IF NOT EXISTS profile_sources (
    fact_id INTEGER NOT NULL REFERENCES profile_facts(id), segment_id INTEGER NOT NULL REFERENCES segments(id),
    PRIMARY KEY (fact_id, segment_id)
);
CREATE TABLE IF NOT EXISTS profile_conflicts (
    id INTEGER PRIMARY KEY, old_id INTEGER NOT NULL REFERENCES profile_facts(id),
    new_id INTEGER NOT NULL REFERENCES profile_facts(id), ts REAL NOT NULL, firmness REAL NOT NULL,
    p_evoke REAL NOT NULL, p_overwrite REAL NOT NULL, roll REAL NOT NULL, outcome TEXT NOT NULL, reason TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evocations (
    id INTEGER PRIMARY KEY, fact_id INTEGER NOT NULL REFERENCES profile_facts(id),
    trigger_fact_id INTEGER REFERENCES profile_facts(id), ts REAL NOT NULL, consumed INTEGER NOT NULL DEFAULT 0
);
"""


class MemoryStore:
    def __init__(self, path: str | Path = DEFAULT_PATH):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(threads)")}
        if "exempt" not in cols:  # D18 之前建的库
            self.db.execute("ALTER TABLE threads ADD COLUMN exempt TEXT")
        if "forgotten_ts" not in cols:  # D25 之前建的库
            self.db.execute("ALTER TABLE threads ADD COLUMN forgotten_ts REAL")

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

    def conversation_ids(self) -> list[int]:
        return [r[0] for r in self.db.execute("SELECT id FROM conversations ORDER BY start_ts, id")]

    def all_turns(self) -> list[ChatTurn]:
        """全部消息按先后；熟悉期回应模块把它们都放进上下文。"""
        rows = self.db.execute("SELECT * FROM turns ORDER BY ts, id")
        return [ChatTurn(r["id"], r["conversation_id"], r["role"], r["content"], r["ts"]) for r in rows]

    def all_turn_texts(self) -> list[str]:
        return [r[0] for r in self.db.execute("SELECT content FROM turns ORDER BY id")]

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        r = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return r[0] if r else default

    def set_meta(self, key: str, value: str) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))

    def mark_remember(self, turn_id: int) -> None:
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO remember_marks VALUES (?)", (turn_id,))

    def remember_marks(self) -> set[int]:
        return {r[0] for r in self.db.execute("SELECT turn_id FROM remember_marks")}

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

    def turn_text(self, turn_ids: list[int]) -> str:
        marks = ",".join("?" * len(turn_ids)) or "NULL"
        rows = self.db.execute(f"SELECT role, content FROM turns WHERE id IN ({marks}) ORDER BY id", turn_ids)
        return "\n".join(f"{'他' if r[0] == 'user' else 'AI'}：{r[1]}" for r in rows)

    def log_sweep(self, ts: float, archived: list[int], kept: dict[int, list[str]]) -> None:
        """一次批量遗忘：哪些移进档案，哪些抽象成了关键词（D22）。"""
        self._insert(
            "INSERT INTO sweep_log (ts, archived, kept) VALUES (?, ?, ?)",
            (ts, json.dumps(archived), json.dumps(kept, ensure_ascii=False)),
        )

    def sweep_log(self) -> list[dict[str, Any]]:
        return [
            {"ts": r["ts"], "archived": json.loads(r["archived"]), "kept": json.loads(r["kept"])}
            for r in self.db.execute("SELECT * FROM sweep_log ORDER BY id")
        ]

    def iter_turns_newest_first(self) -> Iterator[ChatTurn]:
        """所有对话的消息，从最新往回逐条取；新鲜记忆用，取到不新鲜为止。"""
        for r in self.db.execute("SELECT * FROM turns ORDER BY ts DESC, id DESC"):
            yield ChatTurn(r["id"], r["conversation_id"], r["role"], r["content"], r["ts"])

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
                "emotion_reason, half_life, last_recall_ts, recall_count, adjustment, at_edge, state, trace, "
                "remember_requested) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                    int(seg.remember_requested),
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
            bool(r["remember_requested"]),
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

    # ---------- 记忆时钟用的活动记录（D21） ----------

    def add_busy_span(self, start_ts: float, end_ts: float, note: str = "") -> None:
        """智能体处理用户布置的任务的时段。这段时间里记忆时间照常流动，不因没有消息而减速。"""
        self._insert(
            "INSERT INTO busy_spans (start_ts, end_ts, note) VALUES (?, ?, ?)", (start_ts, end_ts, note)
        )

    def activity(self) -> list[tuple[float, float]]:
        """所有活动区间：每条消息是一个点，每个任务是一段。"""
        turns = [(r[0], r[0]) for r in self.db.execute("SELECT ts FROM turns")]
        return turns + [(r[0], r[1]) for r in self.db.execute("SELECT start_ts, end_ts FROM busy_spans")]

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
            bool(r["ongoing"]),
            r["resident_since"],
        )

    def update_thread(
        self, tid: int, *, title: str | None = None, summary: str | None = None, ongoing: bool | None = None
    ) -> None:
        with self.db:
            if title is not None:
                self.db.execute("UPDATE threads SET title = ? WHERE id = ?", (title, tid))
            if summary is not None:
                self.db.execute("UPDATE threads SET summary = ? WHERE id = ?", (summary, tid))
            if ongoing is not None:
                self.db.execute("UPDATE threads SET ongoing = ? WHERE id = ?", (int(ongoing), tid))

    # ---------- 常驻栏（D17）：单位是事件线 ----------

    def thread_stats(self, tid: int) -> dict[str, Any]:
        """常驻打分用的统计：线上片段的最高重要性和唤醒度、回忆总次数、出现在几场对话里、
        是否有用户要求记住的片段、最近更新时间、是否仍在进行。"""
        r = self.db.execute(
            "SELECT MAX(s.importance) AS importance, MAX(s.arousal) AS arousal, SUM(s.recall_count) AS recalls, "
            "COUNT(DISTINCT s.conversation_id) AS conversations, MAX(s.remember_requested) AS remember "
            "FROM thread_segments ts JOIN segments s ON s.id = ts.segment_id WHERE ts.thread_id = ?",
            (tid,),
        ).fetchone()
        t = self.db.execute("SELECT updated_ts, ongoing FROM threads WHERE id = ?", (tid,)).fetchone()
        return {
            "importance": r["importance"] or 0.0,
            "arousal": r["arousal"] or 0.0,
            "recalls": r["recalls"] or 0,
            "conversations": r["conversations"] or 0,
            "remember_requested": bool(r["remember"]),
            "updated_ts": t["updated_ts"],
            "ongoing": bool(t["ongoing"]),
        }

    def thread_has_profile_fact(self, tid: int) -> bool:
        """这条事件线上有没有片段是用户画像（仍在画像中的事实）的出处，即"涉及用户真实信息"。"""
        return (
            self.db.execute(
                "SELECT 1 FROM thread_segments ts JOIN profile_sources ps ON ps.segment_id = ts.segment_id "
                "JOIN profile_facts f ON f.id = ps.fact_id WHERE ts.thread_id = ? AND f.archived_ts IS NULL LIMIT 1",
                (tid,),
            ).fetchone()
            is not None
        )

    def resident_threads(self) -> list[int]:
        return [
            r[0]
            for r in self.db.execute(
                "SELECT id FROM threads WHERE resident_since IS NOT NULL AND merged_into IS NULL AND forgotten_ts IS NULL ORDER BY resident_since"
            )
        ]

    def set_resident(
        self, tid: int, ts: float, resident: bool, reason: str, score: float | None = None
    ) -> None:
        """进入或退出常驻栏，并记日志。退出后不重置遗忘：照常从最近一次被提及开始算。"""
        with self.db:
            self.db.execute(
                "UPDATE threads SET resident_since = ? WHERE id = ?", (ts if resident else None, tid)
            )
            self.db.execute(
                "INSERT INTO resident_log (thread_id, ts, action, reason, score) VALUES (?, ?, ?, ?, ?)",
                (tid, ts, "进入" if resident else "退出", reason, score),
            )

    def exempt_marks(self) -> dict[int, str]:
        """已记下的免竞争标记（D18）：原因；空字符串表示用户要求解除过，不再自动免竞争。"""
        return {
            r[0]: r[1]
            for r in self.db.execute(
                "SELECT id, exempt FROM threads WHERE exempt IS NOT NULL AND merged_into IS NULL AND forgotten_ts IS NULL"
            )
        }

    def mark_exempt(self, tid: int, reason: str) -> None:
        with self.db:
            self.db.execute("UPDATE threads SET exempt = ? WHERE id = ?", (reason, tid))

    def release_exempt(self, tid: int, ts: float, reason: str) -> None:
        """用户要求修改时解除免竞争（D18）。之后这条线和其他线一样竞争名额，不会再被自动标回免竞争。"""
        with self.db:
            self.db.execute("UPDATE threads SET exempt = '' WHERE id = ?", (tid,))
            self.db.execute(
                "INSERT INTO resident_log (thread_id, ts, action, reason, score) VALUES (?, ?, ?, ?, NULL)",
                (tid, ts, "解除免竞争", reason),
            )

    def resident_log(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.db.execute("SELECT * FROM resident_log ORDER BY id")]

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

    # ---------- 忘掉（D25）：用户选择存进档案，或彻底删除 ----------

    def forget_thread_to_archive(self, tid: int, ts: float) -> list[int]:
        """存进档案：这条线和它的片段不再出现在记忆系统里，原文留着，用户要求时可以搜索找回。
        只出自这条线的画像事实一起移进档案。返回移进档案的片段。"""
        sids = self.get_thread(tid).segment_ids
        marks = ",".join("?" * len(sids)) or "NULL"
        with self.db:
            self.db.execute(
                "UPDATE threads SET forgotten_ts = ?, resident_since = NULL WHERE id = ? OR merged_into = ?",
                (ts, tid, tid),
            )
            self.db.execute(f"UPDATE segments SET state = 'archived' WHERE id IN ({marks})", sids)
            self.db.execute(
                "INSERT INTO resident_log (thread_id, ts, action, reason, score) VALUES (?, ?, '退出', ?, NULL)",
                (tid, ts, "用户要求忘掉（存进档案）"),
            )
            for fid in self._facts_only_from(sids):
                self.db.execute(
                    "UPDATE profile_facts SET archived_ts = ? WHERE id = ? AND archived_ts IS NULL", (ts, fid)
                )
        return sids

    def restore_thread(self, tid: int) -> None:
        """从档案找回一条被用户忘掉的线：重新回到记忆系统（片段的状态由调用方处理）。"""
        with self.db:
            self.db.execute(
                "UPDATE threads SET forgotten_ts = NULL WHERE id = ? OR merged_into = ?", (tid, tid)
            )

    def forgotten_thread_ids(self) -> list[int]:
        return [r[0] for r in self.db.execute("SELECT id FROM threads WHERE forgotten_ts IS NOT NULL")]

    def _facts_only_from(self, sids: list[int]) -> list[int]:
        """出处全部在 sids 里的画像事实。"""
        marks = ",".join("?" * len(sids)) or "NULL"
        rows = self.db.execute(
            f"SELECT fact_id FROM profile_sources GROUP BY fact_id "
            f"HAVING SUM(segment_id NOT IN ({marks})) = 0",
            sids,
        )
        return [r[0] for r in rows]

    def erase_thread(self, tid: int, ts: float, reason: str) -> tuple[int, int]:
        """彻底删除：这条线、它的片段、只属于这些片段的消息原文、只出自它们的画像事实，以及相关的日志，
        全部从数据库删掉，不可恢复。只记一条不含内容的删除日志。返回 (删掉的片段数, 删掉的消息数)。"""
        tids = [tid, *[r[0] for r in self.db.execute("SELECT id FROM threads WHERE merged_into = ?", (tid,))]]
        sids = self.get_thread(tid).segment_ids
        sm = ",".join("?" * len(sids)) or "NULL"
        tm = ",".join("?" * len(tids))
        # 只属于这些片段的消息；同时属于别的片段的消息不删（那是别的记忆）
        turns = [
            r[0]
            for r in self.db.execute(
                f"SELECT turn_id FROM segment_turns GROUP BY turn_id HAVING SUM(segment_id NOT IN ({sm})) = 0 "
                f"AND SUM(segment_id IN ({sm})) > 0",
                [*sids, *sids],
            )
        ]
        um = ",".join("?" * len(turns)) or "NULL"
        facts = self._facts_only_from(sids)
        fm = ",".join("?" * len(facts)) or "NULL"
        with self.db:
            ex = self.db.execute
            ex(
                f"DELETE FROM evocations WHERE fact_id IN ({fm}) OR trigger_fact_id IN ({fm})",
                [*facts, *facts],
            )
            ex(f"DELETE FROM profile_conflicts WHERE old_id IN ({fm}) OR new_id IN ({fm})", [*facts, *facts])
            ex(f"UPDATE profile_facts SET superseded_by = NULL WHERE superseded_by IN ({fm})", facts)
            ex(f"DELETE FROM profile_sources WHERE segment_id IN ({sm})", sids)
            ex(f"DELETE FROM profile_facts WHERE id IN ({fm})", facts)
            ex(f"DELETE FROM tuning_log WHERE scope = 'memory' AND target IN ({sm})", [str(i) for i in sids])
            for i in sids:  # 依据这些片段做出的调整，理由里可能带着内容，一起删
                ex("DELETE FROM tuning_log WHERE json_extract(evidence, '$.segment') = ?", (i,))
            for r in list(ex("SELECT id, archived, kept FROM sweep_log")):
                arch = [i for i in json.loads(r[1]) if i not in sids]
                kept = {k: v for k, v in json.loads(r[2]).items() if int(k) not in sids}
                ex(
                    "UPDATE sweep_log SET archived = ?, kept = ? WHERE id = ?",
                    (json.dumps(arch), json.dumps(kept, ensure_ascii=False), r[0]),
                )
            ex(f"DELETE FROM segment_links WHERE from_id IN ({sm}) OR to_id IN ({sm})", [*sids, *sids])
            ex(f"DELETE FROM segment_people WHERE segment_id IN ({sm})", sids)
            ex(f"DELETE FROM segment_turns WHERE segment_id IN ({sm})", sids)
            ex(
                f"DELETE FROM thread_segments WHERE thread_id IN ({tm}) OR segment_id IN ({sm})",
                [*tids, *sids],
            )
            ex(f"DELETE FROM segments WHERE id IN ({sm})", sids)
            ex(f"DELETE FROM boundaries WHERE turn_id IN ({um})", turns)
            ex(f"DELETE FROM turns WHERE id IN ({um})", turns)
            ex(f"DELETE FROM thread_links WHERE a IN ({tm}) OR b IN ({tm})", [*tids, *tids])
            ex(f"DELETE FROM resident_log WHERE thread_id IN ({tm})", tids)
            ex(f"DELETE FROM threads WHERE id IN ({tm})", tids)
            ex(
                "INSERT INTO erase_log (ts, segments, turns, reason) VALUES (?, ?, ?, ?)",
                (ts, len(sids), len(turns), reason),
            )
        return len(sids), len(turns)

    def erase_log(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.db.execute("SELECT * FROM erase_log ORDER BY id")]

    def live_threads(self) -> list[int]:
        return [
            r[0]
            for r in self.db.execute(
                "SELECT id FROM threads WHERE merged_into IS NULL AND forgotten_ts IS NULL ORDER BY updated_ts DESC"
            )
        ]

    def thread_card(self, tid: int) -> dict[str, Any]:
        """给潜意识模型看的事件线卡片：只含标题、一句话摘要、最近片段标题、涉及人物。"""
        t = self.get_thread(tid)
        visible = [s for s in map(self.get_segment, t.segment_ids) if s.state != "archived"]  # D22
        latest = (
            "" if not visible else visible[-1].title if visible[-1].state == "active" else visible[-1].trace
        )
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

    # ---------- 用户画像（稳定事实：不改写原文、不参与遗忘；矛盾时可能被覆盖进档案，见 D16） ----------

    def add_profile_fact(
        self, person_id: int | None, category: str, fact: str, ts: float, segment_id: int | None = None
    ) -> int:
        """登记一条稳定事实。同一个人、同一类、内容相同（忽略标点和空格）的在画像中的事实合并：
        只增加提及次数、更新最近提及时间、补充出处，原文不改。已移进档案的旧事实不参与合并。"""
        key = _fact_key(fact)
        with self.db:
            row = self.db.execute(
                "SELECT id FROM profile_facts WHERE IFNULL(person_id, 0) = IFNULL(?, 0) AND category = ? "
                "AND fact_key = ? AND archived_ts IS NULL",
                (person_id, category, key),
            ).fetchone()
            if row:
                fid = row[0]
                self.db.execute(
                    "UPDATE profile_facts SET mentions = mentions + 1, last_ts = MAX(last_ts, ?) WHERE id = ?",
                    (ts, fid),
                )
            else:
                fid = self.db.execute(
                    "INSERT INTO profile_facts (person_id, category, fact, fact_key, first_ts, last_ts, mentions) "
                    "VALUES (?, ?, ?, ?, ?, ?, 1)",
                    (person_id, category, fact.strip(), key, ts, ts),
                ).lastrowid
            if segment_id is not None:
                self.db.execute("INSERT OR IGNORE INTO profile_sources VALUES (?, ?)", (fid, segment_id))
        return fid

    def profile(
        self, person_id: int | None = None, *, everyone: bool = False, archived: bool = False
    ) -> list[ProfileFact]:
        """某个人（默认用户本人）的画像；everyone=True 时返回所有人的。
        archived=False 只返回画像中的；archived=True 只返回已被覆盖、移进档案的。"""
        sql, args = (
            "SELECT * FROM profile_facts WHERE archived_ts IS " + ("NOT NULL" if archived else "NULL"),
            (),
        )
        if not everyone:
            sql, args = sql + " AND IFNULL(person_id, 0) = IFNULL(?, 0)", (person_id,)
        out = []
        for r in self.db.execute(sql + " ORDER BY category, first_ts", args):
            sources = [
                x[0]
                for x in self.db.execute(
                    "SELECT segment_id FROM profile_sources WHERE fact_id = ?", (r["id"],)
                )
            ]
            out.append(
                ProfileFact(
                    r["id"],
                    r["person_id"],
                    r["category"],
                    r["fact"],
                    r["first_ts"],
                    r["last_ts"],
                    r["mentions"],
                    sources,
                    r["archived_ts"],
                    r["superseded_by"],
                )
            )
        return out

    def fact_text(self, fact_id: int) -> str | None:
        r = self.db.execute("SELECT fact FROM profile_facts WHERE id = ?", (fact_id,)).fetchone()
        return r[0] if r else None

    def archive_profile_fact(self, old_id: int, new_id: int, ts: float) -> None:
        """覆盖：旧事实移进档案（不删除），记下是哪条新事实覆盖了它。"""
        with self.db:
            self.db.execute(
                "UPDATE profile_facts SET archived_ts = ?, superseded_by = ? WHERE id = ?",
                (ts, new_id, old_id),
            )

    def log_profile_conflict(
        self,
        old_id: int,
        new_id: int,
        ts: float,
        firmness: float,
        probs: dict[str, float],
        roll: float,
        outcome: str,
        reason: str,
    ) -> None:
        with self.db:
            self.db.execute(
                "INSERT INTO profile_conflicts (old_id, new_id, ts, firmness, p_evoke, p_overwrite, roll, outcome, "
                "reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (old_id, new_id, ts, firmness, probs["唤起"], probs["覆盖"], roll, outcome, reason),
            )

    def profile_conflicts(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.db.execute("SELECT * FROM profile_conflicts ORDER BY id")]

    def add_evocation(self, fact_id: int, trigger_fact_id: int | None, ts: float) -> None:
        """唤起：这条旧事实被重新想起，等浮现区（扫描器）或思考模型来取。"""
        with self.db:
            self.db.execute(
                "INSERT INTO evocations (fact_id, trigger_fact_id, ts) VALUES (?, ?, ?)",
                (fact_id, trigger_fact_id, ts),
            )

    def pending_evocations(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.db.execute("SELECT * FROM evocations WHERE consumed = 0 ORDER BY id")]

    def consume_evocation(self, evocation_id: int) -> None:
        with self.db:
            self.db.execute("UPDATE evocations SET consumed = 1 WHERE id = ?", (evocation_id,))

    # ---------- 事件线候选的查询（候选怎么取见 memory.candidates） ----------

    def threads_with_people(self, person_ids: list[int]) -> list[int]:
        if not person_ids:
            return []
        marks = ",".join("?" * len(person_ids))
        return [
            r[0]
            for r in self.db.execute(
                f"SELECT DISTINCT ts.thread_id FROM thread_segments ts JOIN segment_people sp ON sp.segment_id = ts.segment_id "
                f"JOIN threads t ON t.id = ts.thread_id WHERE t.merged_into IS NULL AND t.forgotten_ts IS NULL AND sp.person_id IN ({marks})",
                person_ids,
            )
        ]

    def threads_mentioning(self, text: str, exclude_turns: list[int] = ()) -> list[int]:
        """档案原文里出现过 text 的事件线（包括已经遗忘的片段，因为原文一直在）。"""
        excl = ",".join("?" * len(exclude_turns)) or "NULL"
        return [
            r[0]
            for r in self.db.execute(
                "SELECT DISTINCT ts.thread_id FROM turns tu JOIN segment_turns st ON st.turn_id = tu.id "
                "JOIN thread_segments ts ON ts.segment_id = st.segment_id JOIN threads t ON t.id = ts.thread_id "
                f"WHERE t.merged_into IS NULL AND t.forgotten_ts IS NULL AND instr(tu.content, ?) > 0 AND tu.id NOT IN ({excl})",
                (text, *exclude_turns),
            )
        ]

    def thread_text(self, tid: int) -> str:
        """事件线的文字：标题、摘要和片段标题（抽象化的用关键词，移进档案的不算），用于关键词检索。"""
        t = self.get_thread(tid)
        titles = [
            r[0]
            for r in self.db.execute(
                "SELECT CASE s.state WHEN 'abstract' THEN s.trace ELSE s.title END FROM segments s "
                "JOIN thread_segments ts ON ts.segment_id = s.id WHERE ts.thread_id = ? AND s.state != 'archived'",
                (tid,),
            )
        ]
        return " ".join([t.title, t.summary, *titles])


def _fact_key(fact: str) -> str:
    """合并用的键：去掉标点和空格。"""
    return re.sub(r"[\s，。！？、,.!?；;：:\"'“”（）()]", "", fact)
