"""记忆存储：SQLite 单文件。默认放在 data/memory.db（不进版本库）。

只负责存取，不做判断：什么时候切段、挂到哪条事件线、怎么调参数，由潜意识的流程决定。
时间存 Unix 时间戳（秒）；调参日志沿用遗忘曲线的单位"天"。
"""

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from contracts.memory import ChatTurn, Conversation

from ._db import DEFAULT_PATH, SCHEMA, Db
from ._facade import _StoreMethods
from .archive import ArchiveStore
from .profile import ProfileStore
from .segments import SegmentStore
from .threads import ThreadStore
from .tuning import TuningStore

__all__ = ["DEFAULT_PATH", "SCHEMA", "MemoryStore"]


class MemoryStore(_StoreMethods):
    def __init__(self, path: str | Path = DEFAULT_PATH):
        self._db = Db(path)
        self._archive = ArchiveStore(self._db)
        self._segments = SegmentStore(self._db)
        self._threads = ThreadStore(self._db)
        self._profile = ProfileStore(self._db)
        self._tuning = TuningStore(self._db)

    @property
    def db(self) -> sqlite3.Connection:
        return self._db.conn

    def transaction(self) -> Iterator[None]:
        """可嵌套的短事务：阶段结果与进度一起提交，不跨模型调用持有写锁。"""
        return self._db.transaction()

    def close(self) -> None:
        self.db.close()

    def _insert(self, sql: str, args: tuple) -> int:
        with self.transaction():
            return self.db.execute(sql, args).lastrowid

    def new_conversation(self, start_ts: float) -> Conversation:
        return self._archive.new_conversation(start_ts)

    def add_turn(self, conversation_id: int, role: str, content: str, ts: float) -> ChatTurn:
        return self._archive.add_turn(conversation_id, role, content, ts)

    def get_conversation(self, cid: int) -> Conversation:
        return self._archive.get_conversation(cid)

    def conversation_ids(self) -> list[int]:
        return self._archive.conversation_ids()

    def all_turns(self, *, include_forgotten: bool = True) -> list[ChatTurn]:
        """全部消息按先后；熟悉期回应模块把它们都放进上下文。"""
        rows = self.db.execute("SELECT * FROM turns ORDER BY ts, id")
        hidden = set() if include_forgotten else self.forgotten_turn_ids()
        return [
            ChatTurn(r["id"], r["conversation_id"], r["role"], r["content"], r["ts"])
            for r in rows
            if r["id"] not in hidden
        ]

    def all_turn_texts(self) -> list[str]:
        return self._archive.all_turn_texts()

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        return self._archive.get_meta(key, default)

    def set_meta(self, key: str, value: str) -> None:
        return self._archive.set_meta(key, value)

    def visibility_revision(self) -> int:
        return self._archive.visibility_revision()

    def mark_remember(self, turn_id: int) -> None:
        return self._archive.mark_remember(turn_id)

    def remember_marks(self) -> set[int]:
        return self._archive.remember_marks()

    def last_conversation(self) -> Conversation | None:
        return self._archive.last_conversation()

    def set_conversation_summary(self, cid: int, summary: str) -> None:
        return self._archive.set_conversation_summary(cid, summary)

    def turns(self, conversation_id: int, after_id: int = 0) -> list[ChatTurn]:
        return self._archive.turns(conversation_id, after_id)

    def turn_text(self, turn_ids: list[int]) -> str:
        return self._archive.turn_text(turn_ids)

    def log_sweep(self, ts: float, archived: list[int], kept: dict[int, list[str]]) -> None:
        return self._archive.log_sweep(ts, archived, kept)

    def sweep_log(self) -> list[dict[str, Any]]:
        return self._archive.sweep_log()

    def iter_turns_newest_first(self) -> Iterator[ChatTurn]:
        """所有对话的消息，从最新往回逐条取；新鲜记忆用，取到不新鲜为止。"""
        hidden = self.forgotten_turn_ids()
        for r in self.db.execute("SELECT * FROM turns ORDER BY ts DESC, id DESC"):
            if r["id"] not in hidden:
                yield ChatTurn(r["id"], r["conversation_id"], r["role"], r["content"], r["ts"])

    def unsegmented_turns(self, conversation_id: int) -> list[ChatTurn]:
        return self._archive.unsegmented_turns(conversation_id)

    def add_busy_span(self, start_ts: float, end_ts: float, note: str = "") -> None:
        return self._archive.add_busy_span(start_ts, end_ts, note)

    def activity(self) -> list[tuple[float, float]]:
        return self._archive.activity()

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
