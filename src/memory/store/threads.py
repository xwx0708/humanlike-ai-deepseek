"""事件线、常驻栏、归档与彻底删除。"""

import json

from contracts.memory import Thread

from ._db import Db
from ._thread_queries import _ThreadQueries


class ThreadStore(_ThreadQueries):
    def __init__(self, db: Db):
        self.db = db

    def new_thread(self, title: str, summary: str, ts: float) -> Thread:
        tid = self.db._insert(
            "INSERT INTO threads (title, summary, created_ts, updated_ts) VALUES (?, ?, ?, ?)",
            (title, summary, ts, ts),
        )
        return Thread(tid, title, summary, ts, ts)

    def attach(self, thread_id: int, segment_id: int, confidence: float, ts: float) -> None:
        with self.db.transaction():
            self.db.execute(
                "INSERT OR REPLACE INTO thread_segments VALUES (?, ?, ?)", (thread_id, segment_id, confidence)
            )
            self.db.execute(
                "UPDATE threads SET updated_ts = MAX(updated_ts, ?) WHERE id = ?", (ts, thread_id)
            )

    def update_thread(
        self, tid: int, *, title: str | None = None, summary: str | None = None, ongoing: bool | None = None
    ) -> None:
        with self.db.transaction():
            if title is not None:
                self.db.execute("UPDATE threads SET title = ? WHERE id = ?", (title, tid))
            if summary is not None:
                self.db.execute("UPDATE threads SET summary = ? WHERE id = ?", (summary, tid))
            if ongoing is not None:
                self.db.execute("UPDATE threads SET ongoing = ? WHERE id = ?", (int(ongoing), tid))

    def set_resident(
        self, tid: int, ts: float, resident: bool, reason: str, score: float | None = None
    ) -> None:
        """进入或退出常驻栏，并记日志。退出后不重置遗忘：照常从最近一次被提及开始算。"""
        with self.db.transaction():
            self.db.execute(
                "UPDATE threads SET resident_since = ? WHERE id = ?", (ts if resident else None, tid)
            )
            self.db.execute(
                "INSERT INTO resident_log (thread_id, ts, action, reason, score) VALUES (?, ?, ?, ?, ?)",
                (tid, ts, "进入" if resident else "退出", reason, score),
            )

    def mark_exempt(self, tid: int, reason: str) -> None:
        with self.db.transaction():
            self.db.execute("UPDATE threads SET exempt = ? WHERE id = ?", (reason, tid))

    def release_exempt(self, tid: int, ts: float, reason: str) -> None:
        """用户要求修改时解除免竞争（D18）。之后这条线和其他线一样竞争名额，不会再被自动标回免竞争。"""
        with self.db.transaction():
            self.db.execute("UPDATE threads SET exempt = '' WHERE id = ?", (tid,))
            self.db.execute(
                "INSERT INTO resident_log (thread_id, ts, action, reason, score) VALUES (?, ?, ?, ?, NULL)",
                (tid, ts, "解除免竞争", reason),
            )

    def relate_threads(self, a: int, b: int, note: str) -> None:
        """两条事件线"有关但不是同一件事"，例如漂移路径把它们连在一起。"""
        a, b = min(a, b), max(a, b)
        with self.db.transaction():
            self.db.execute("INSERT OR REPLACE INTO thread_links VALUES (?, ?, ?)", (a, b, note))

    def merge_threads(self, keep: int, drop: int, ts: float) -> None:
        """夜间合并：把 drop 的片段移到 keep，drop 标记为已合并（不删除，便于追溯）。"""
        with self.db.transaction():
            self.db.execute(
                "INSERT OR IGNORE INTO thread_segments SELECT ?, segment_id, confidence "
                "FROM thread_segments WHERE thread_id = ?",
                (keep, drop),
            )
            self.db.execute("DELETE FROM thread_segments WHERE thread_id = ?", (drop,))
            self.db.execute("UPDATE threads SET merged_into = ? WHERE id = ?", (keep, drop))
            self.db.execute("UPDATE threads SET updated_ts = MAX(updated_ts, ?) WHERE id = ?", (ts, keep))

    def _invalidate_summaries(self, sids: list[int]) -> None:
        """主动遗忘后立即丢弃派生摘要；共享事件线仅用其余可见片段重建，不沿用旧摘要。"""
        marks = ",".join("?" * len(sids)) or "NULL"
        self.db.execute(
            f"UPDATE conversations SET summary = '' WHERE id IN "
            f"(SELECT conversation_id FROM segments WHERE id IN ({marks}))",
            sids,
        )
        tids = {
            r[0]
            for r in self.db.execute(
                f"SELECT thread_id FROM thread_segments WHERE segment_id IN ({marks})", sids
            )
        }
        for tid in tids:
            parts = [
                r[0]
                for r in self.db.execute(
                    "SELECT CASE s.state WHEN 'abstract' THEN s.trace ELSE s.title END FROM segments s "
                    "JOIN thread_segments ts ON ts.segment_id = s.id "
                    f"WHERE ts.thread_id = ? AND s.state != 'archived' AND s.id NOT IN ({marks}) "
                    "ORDER BY s.start_ts, s.id",
                    (tid, *sids),
                )
            ]
            self.db.execute(
                "UPDATE threads SET summary = ? WHERE id = ? AND forgotten_ts IS NULL",
                ("；".join(parts[-10:]), tid),
            )

    def forget_thread_to_archive(self, tid: int, ts: float) -> list[int]:
        """存进档案：这条线和它的片段不再出现在记忆系统里，原文留着，用户要求时可以搜索找回。
        只出自这条线的画像事实一起移进档案。返回移进档案的片段。"""
        sids = self.get_thread(tid).segment_ids
        marks = ",".join("?" * len(sids)) or "NULL"
        with self.db.transaction():
            self._set_meta("visibility_revision", str(self._visibility_revision() + 1))
            self.db.execute(
                "UPDATE threads SET forgotten_ts = ?, resident_since = NULL WHERE id = ? OR merged_into = ?",
                (ts, tid, tid),
            )
            self.db.execute(f"UPDATE segments SET state = 'archived' WHERE id IN ({marks})", sids)
            self._invalidate_summaries(sids)
            self.db.execute(
                "INSERT INTO resident_log (thread_id, ts, action, reason, score) VALUES (?, ?, '退出', ?, NULL)",
                (tid, ts, "用户要求忘掉（存进档案）"),
            )
            for fid in self._facts_only_from(sids):
                self.db.execute(
                    "UPDATE profile_facts SET archived_ts = ? WHERE id = ? AND archived_ts IS NULL", (ts, fid)
                )
                self.db.execute(
                    "UPDATE evocations SET consumed = 1 WHERE fact_id = ? OR trigger_fact_id = ?", (fid, fid)
                )
        return sids

    def restore_thread(self, tid: int) -> None:
        """从档案找回一条被用户忘掉的线：重新回到记忆系统（片段的状态由调用方处理）。"""
        with self.db.transaction():
            self.db.execute(
                "UPDATE threads SET forgotten_ts = NULL WHERE id = ? OR merged_into = ?", (tid, tid)
            )

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
        tids = [
            r[0]
            for r in self.db.execute(
                "WITH RECURSIVE family(id) AS (SELECT ? UNION "
                "SELECT t.id FROM threads t JOIN family f ON t.merged_into = f.id) SELECT id FROM family",
                (tid,),
            )
        ]
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
        with self.db.transaction():
            ex = self.db.execute
            self._set_meta("visibility_revision", str(self._visibility_revision() + 1))
            self._invalidate_summaries(sids)
            if ex("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'merge_log'").fetchone():
                ex(f"DELETE FROM merge_log WHERE keep_id IN ({tm}) OR drop_id IN ({tm})", [*tids, *tids])
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
            ex(f"DELETE FROM remember_marks WHERE turn_id IN ({um})", turns)
            ex(f"DELETE FROM thread_links WHERE a IN ({tm}) OR b IN ({tm})", [*tids, *tids])
            ex(f"DELETE FROM resident_log WHERE thread_id IN ({tm})", tids)
            ex(f"UPDATE threads SET merged_into = NULL WHERE id IN ({tm})", tids)
            ex(f"DELETE FROM threads WHERE id IN ({tm})", tids)
            ex(
                "INSERT INTO erase_log (ts, segments, turns, reason) VALUES (?, ?, ?, ?)",
                (ts, len(sids), len(turns), reason),
            )
        return len(sids), len(turns)

    # 主动遗忘跨表读写 meta；保留原 SQL 与嵌套事务，不引用其他小存储。
    def _get_meta(self, key: str, default: str | None = None) -> str | None:
        r = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return r[0] if r else default

    def _set_meta(self, key: str, value: str) -> None:
        with self.db.transaction():
            self.db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))

    def _visibility_revision(self) -> int:
        """主动遗忘发生后，丢弃后台模型基于旧内容算出的结果。"""
        return int(self._get_meta("visibility_revision", "0"))
