"""候选边界、片段、片段间链接与整理任务。"""

import json
from typing import Any

from contracts.memory import EmotionScore, Segment

from ._db import Db


class SegmentStore:
    def __init__(self, db: Db):
        self.db = db

    def add_boundary(self, turn_id: int, reason: str) -> None:
        with self.db.transaction():
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

    def add_segment(self, seg: Segment, *, job: dict[str, Any] | None = None) -> Segment:
        e = seg.emotion
        with self.db.transaction():
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
            if job is not None:
                self.save_consolidation_job(sid, job.get("batch_id") or sid, "facts", job)
        seg.id = sid
        return seg

    def save_consolidation_job(self, sid: int, batch_id: int, stage: str, payload: dict) -> None:
        with self.db.transaction():
            self.db.execute(
                "INSERT OR REPLACE INTO consolidation_jobs VALUES (?, ?, ?, ?)",
                (sid, batch_id, stage, json.dumps(payload, ensure_ascii=False)),
            )

    def consolidation_job(self, sid: int) -> tuple[int, str, dict]:
        r = self.db.execute("SELECT * FROM consolidation_jobs WHERE segment_id = ?", (sid,)).fetchone()
        return r["batch_id"], r["stage"], json.loads(r["payload"])

    def pending_consolidations(self, cid: int) -> list[int]:
        return [
            r[0]
            for r in self.db.execute(
                "SELECT j.segment_id FROM consolidation_jobs j JOIN segments s ON s.id = j.segment_id "
                "WHERE s.conversation_id = ? AND j.stage != 'done' AND s.state != 'archived' "
                "ORDER BY j.segment_id",
                (cid,),
            )
        ]

    def consolidation_batch(self, batch_id: int) -> set[int]:
        return {
            r[0]
            for r in self.db.execute(
                "SELECT segment_id FROM consolidation_jobs WHERE batch_id = ?", (batch_id,)
            )
        }

    def adjusted_in_batch(self, batch_id: int) -> set[str]:
        return {
            r[0]
            for r in self.db.execute(
                "SELECT target FROM tuning_log WHERE scope = 'memory' AND json_extract(evidence, '$.batch') = ?",
                (batch_id,),
            )
        }

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
        with self.db.transaction():
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
        with self.db.transaction():
            self.db.execute("INSERT OR IGNORE INTO segment_links VALUES (?, ?, ?)", (from_id, to_id, kind))

    def segment_links(self, conversation_id: int) -> list[tuple[int, int, str]]:
        rows = self.db.execute(
            "SELECT l.* FROM segment_links l JOIN segments s ON s.id = l.from_id WHERE s.conversation_id = ?",
            (conversation_id,),
        )
        return [(r["from_id"], r["to_id"], r["kind"]) for r in rows]
