"""事件线存储的只读查询；与写入部分共用同一个 Db。"""

from typing import Any

from contracts.memory import Thread

from ._db import Db


class _ThreadQueries:
    db: Db

    def forgotten_turn_ids(self) -> set[int]:
        """用户主动存档的原消息；自然遗忘不在此列，显式恢复事件线后自动解除。"""
        return {
            r[0]
            for r in self.db.execute(
                "SELECT DISTINCT st.turn_id FROM segment_turns st "
                "JOIN thread_segments ts ON ts.segment_id = st.segment_id "
                "JOIN threads t ON t.id = ts.thread_id WHERE t.forgotten_ts IS NOT NULL"
            )
        }

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

    def exempt_marks(self) -> dict[int, str]:
        """已记下的免竞争标记（D18）：原因；空字符串表示用户要求解除过，不再自动免竞争。"""
        return {
            r[0]: r[1]
            for r in self.db.execute(
                "SELECT id, exempt FROM threads WHERE exempt IS NOT NULL AND merged_into IS NULL AND forgotten_ts IS NULL"
            )
        }

    def resident_log(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.db.execute("SELECT * FROM resident_log ORDER BY id")]

    def threads_of_segment(self, segment_id: int) -> list[int]:
        return [
            r[0]
            for r in self.db.execute(
                "SELECT thread_id FROM thread_segments WHERE segment_id = ?", (segment_id,)
            )
        ]

    def forgotten_thread_ids(self) -> list[int]:
        return [r[0] for r in self.db.execute("SELECT id FROM threads WHERE forgotten_ts IS NOT NULL")]

    def erase_log(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.db.execute("SELECT * FROM erase_log ORDER BY id")]

    def live_threads(self) -> list[int]:
        return [
            r[0]
            for r in self.db.execute(
                "SELECT id FROM threads WHERE merged_into IS NULL AND forgotten_ts IS NULL ORDER BY updated_ts DESC"
            )
        ]

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
