"""一场对话、消息原文、元数据、记住标记与活动日志。"""

import json
from typing import Any

from contracts.memory import ChatTurn, Conversation

from ._db import Db


class ArchiveStore:
    def __init__(self, db: Db):
        self.db = db

    def new_conversation(self, start_ts: float) -> Conversation:
        cid = self.db._insert(
            "INSERT INTO conversations (start_ts, end_ts) VALUES (?, ?)", (start_ts, start_ts)
        )
        return Conversation(cid, start_ts, start_ts)

    def add_turn(self, conversation_id: int, role: str, content: str, ts: float) -> ChatTurn:
        with self.db.transaction():
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

    def all_turn_texts(self) -> list[str]:
        return [r[0] for r in self.db.execute("SELECT content FROM turns ORDER BY id")]

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        r = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return r[0] if r else default

    def set_meta(self, key: str, value: str) -> None:
        with self.db.transaction():
            self.db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))

    def visibility_revision(self) -> int:
        """主动遗忘发生后，丢弃后台模型基于旧内容算出的结果。"""
        return int(self.get_meta("visibility_revision", "0"))

    def mark_remember(self, turn_id: int) -> None:
        with self.db.transaction():
            self.db.execute("INSERT OR IGNORE INTO remember_marks VALUES (?)", (turn_id,))

    def remember_marks(self) -> set[int]:
        return {r[0] for r in self.db.execute("SELECT turn_id FROM remember_marks")}

    def last_conversation(self) -> Conversation | None:
        r = self.db.execute("SELECT id FROM conversations ORDER BY end_ts DESC LIMIT 1").fetchone()
        return self.get_conversation(r["id"]) if r else None

    def set_conversation_summary(self, cid: int, summary: str) -> None:
        with self.db.transaction():
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
        self.db._insert(
            "INSERT INTO sweep_log (ts, archived, kept) VALUES (?, ?, ?)",
            (ts, json.dumps(archived), json.dumps(kept, ensure_ascii=False)),
        )

    def sweep_log(self) -> list[dict[str, Any]]:
        return [
            {"ts": r["ts"], "archived": json.loads(r["archived"]), "kept": json.loads(r["kept"])}
            for r in self.db.execute("SELECT * FROM sweep_log ORDER BY id")
        ]

    def unsegmented_turns(self, conversation_id: int) -> list[ChatTurn]:
        """还没被切进任何片段的消息：第二层要处理的部分。"""
        rows = self.db.execute(
            "SELECT * FROM turns WHERE conversation_id = ? "
            "AND id NOT IN (SELECT turn_id FROM segment_turns) ORDER BY id",
            (conversation_id,),
        )
        return [ChatTurn(r["id"], r["conversation_id"], r["role"], r["content"], r["ts"]) for r in rows]

    def add_busy_span(self, start_ts: float, end_ts: float, note: str = "") -> None:
        """智能体处理用户布置的任务的时段。这段时间里记忆时间照常流动，不因没有消息而减速。"""
        self.db._insert(
            "INSERT INTO busy_spans (start_ts, end_ts, note) VALUES (?, ?, ?)", (start_ts, end_ts, note)
        )

    def activity(self) -> list[tuple[float, float]]:
        """所有活动区间：每条消息是一个点，每个任务是一段。"""
        turns = [(r[0], r[0]) for r in self.db.execute("SELECT ts FROM turns")]
        return turns + [(r[0], r[1]) for r in self.db.execute("SELECT start_ts, end_ts FROM busy_spans")]
