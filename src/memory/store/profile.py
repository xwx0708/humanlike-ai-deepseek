"""人物别名、用户画像、画像冲突与唤起。"""

import re
from typing import Any

from contracts.memory import Person, ProfileFact

from ._db import Db


class ProfileStore:
    def __init__(self, db: Db):
        self.db = db

    def resolve(self, name: str) -> int | None:
        r = self.db.execute("SELECT person_id FROM aliases WHERE alias = ?", (name.strip(),)).fetchone()
        return r[0] if r else None

    def upsert_person(self, name: str, aliases: list[str] = ()) -> int:
        """登记一个人和他的叫法。任何一个叫法已经认识，就并到那个人；否则新建。"""
        names = [n.strip() for n in [name, *aliases] if n and n.strip()]
        known = {pid for n in names if (pid := self.resolve(n)) is not None}
        with self.db.transaction():
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

    def add_profile_fact(
        self, person_id: int | None, category: str, fact: str, ts: float, segment_id: int | None = None
    ) -> int:
        """登记一条稳定事实。同一个人、同一类、内容相同（忽略标点和空格）的在画像中的事实合并：
        只增加提及次数、更新最近提及时间、补充出处，原文不改。已移进档案的旧事实不参与合并。"""
        key = _fact_key(fact)
        with self.db.transaction():
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
        with self.db.transaction():
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
        with self.db.transaction():
            self.db.execute(
                "INSERT INTO profile_conflicts (old_id, new_id, ts, firmness, p_evoke, p_overwrite, roll, outcome, "
                "reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (old_id, new_id, ts, firmness, probs["唤起"], probs["覆盖"], roll, outcome, reason),
            )

    def profile_conflicts(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.db.execute("SELECT * FROM profile_conflicts ORDER BY id")]

    def add_evocation(self, fact_id: int, trigger_fact_id: int | None, ts: float) -> None:
        """唤起：这条旧事实被重新想起，等浮现区（扫描器）或思考模型来取。"""
        with self.db.transaction():
            self.db.execute(
                "INSERT INTO evocations (fact_id, trigger_fact_id, ts) VALUES (?, ?, ?)",
                (fact_id, trigger_fact_id, ts),
            )

    def pending_evocations(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.db.execute("SELECT * FROM evocations WHERE consumed = 0 ORDER BY id")]

    def consume_evocation(self, evocation_id: int) -> None:
        with self.db.transaction():
            self.db.execute("UPDATE evocations SET consumed = 1 WHERE id = ?", (evocation_id,))


def _fact_key(fact: str) -> str:
    """合并用的键：去掉标点和空格。"""
    return re.sub(r"[\s，。！？、,.!?；;：:\"'“”（）()]", "", fact)
