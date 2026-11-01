"""全局参数与调参日志。"""

import json
from dataclasses import asdict, fields, replace
from typing import Any

from memory.forgetting import ForgettingKnobs
from memory.tuning import GlobalChange
from runtime import config

from ._db import Db


class TuningStore:
    def __init__(self, db: Db):
        self.db = db

    def load_global_knobs(self) -> ForgettingKnobs:
        saved = {r["name"]: r["value"] for r in self.db.execute("SELECT * FROM global_knobs")}
        user = set(config.section("forgetting"))  # 设置页里改过的，以用户为准（D40）
        valid = {f.name for f in fields(ForgettingKnobs)} - user
        return replace(
            config.knobs(ForgettingKnobs(), "forgetting"), **{k: v for k, v in saved.items() if k in valid}
        )

    def save_global_knobs(self, knobs: ForgettingKnobs) -> None:
        with self.db.transaction():
            self.db.executemany(
                "INSERT OR REPLACE INTO global_knobs VALUES (?, ?)", list(asdict(knobs).items())
            )

    def log_tuning(
        self, scope: str, target: str, change: GlobalChange, evidence: dict[str, Any] | None = None
    ) -> None:
        """记一条调参日志。scope："memory"（单条记忆）或 "global"；被拒绝的请求也记。"""
        with self.db.transaction():
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
