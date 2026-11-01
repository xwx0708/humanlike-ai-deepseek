"""共享 SQLite 连接、建表迁移与可嵌套事务。"""

import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from runtime import config

DEFAULT_PATH = config.data_dir() / "memory.db"  # 数据文件夹可在设置页选（D40）

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
CREATE TABLE IF NOT EXISTS consolidation_jobs (
    segment_id INTEGER PRIMARY KEY REFERENCES segments(id) ON DELETE CASCADE,
    batch_id INTEGER NOT NULL, stage TEXT NOT NULL, payload TEXT NOT NULL
);
"""


class Db:
    def __init__(self, path: str | Path = DEFAULT_PATH):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self._savepoint = 0
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(threads)")}
        if "exempt" not in cols:  # D18 之前建的库
            self.conn.execute("ALTER TABLE threads ADD COLUMN exempt TEXT")
        if "forgotten_ts" not in cols:  # D25 之前建的库
            self.conn.execute("ALTER TABLE threads ADD COLUMN forgotten_ts REAL")
        # 旧版整理失败可能留下未挂线的片段；重开数据库后也能恢复。
        with self.transaction():
            self.conn.execute(
                "INSERT OR IGNORE INTO consolidation_jobs "
                "SELECT s.id, s.id, 'thread', '{}' FROM segments s "
                "WHERE s.state = 'active' AND NOT EXISTS "
                "(SELECT 1 FROM thread_segments ts WHERE ts.segment_id = s.id)"
            )

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """可嵌套的短事务：阶段结果与进度一起提交，不跨模型调用持有写锁。"""
        self._savepoint += 1
        name = f"memory_{self._savepoint}"
        self.conn.execute(f"SAVEPOINT {name}")
        try:
            yield
        except BaseException:
            self.conn.execute(f"ROLLBACK TO {name}")
            self.conn.execute(f"RELEASE {name}")
            raise
        else:
            self.conn.execute(f"RELEASE {name}")

    def _insert(self, sql: str, args: tuple) -> int:
        with self.transaction():
            return self.conn.execute(sql, args).lastrowid

    def execute(self, sql: str, parameters: Iterable[Any] = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, parameters)

    def executemany(self, sql: str, parameters: Iterable[Iterable[Any]]) -> sqlite3.Cursor:
        return self.conn.executemany(sql, parameters)
