"""和 AstrBot 合成一个进程（D29）：记忆的后台任务登记进共用调度，每次自己开关数据库连接。"""

from llm import FakeModel
from memory.lifecycle import report_context_used
from memory.store import MemoryStore
from response.background import make_scheduler, memory_job
from runtime.lifecycle import RUNNING, Lifecycle


def test_memory_job_starts_lifecycle_when_context_is_70_percent_full(tmp_path):
    db, life = tmp_path / "memory.db", tmp_path / "lifecycle.json"
    store = MemoryStore(db)
    c = store.new_conversation(0)
    store.add_turn(c.id, "user", "你好", 0)
    job = memory_job(FakeModel([]), context_limit=1000, db_path=db, lifecycle_path=life)
    assert job(60).started is None
    report_context_used(store, 700)  # 回应模块回报：上下文用到 70%
    store.close()
    seg = {
        "segments": [
            {
                "messages": [1],
                "title": "打招呼",
                "people": [],
                "arousal": 0.1,
                "valence": 0.1,
                "importance": 0.1,
                "emotion_reason": "",
            }
        ]
    }
    new = {
        "thread_ids": [],
        "new_thread": True,
        "new_thread_title": "打招呼",
        "related": [],
        "new_thread_ongoing": False,
        "confidence": 1,
        "reason": "",
    }
    job = memory_job(FakeModel([seg, new]), context_limit=1000, db_path=db, lifecycle_path=life)
    assert "记忆" in job(120).started  # 启动：整理熟悉期的全部对话
    assert Lifecycle(path=life).phase == RUNNING
    store = MemoryStore(db)
    assert [s.title for s in store.segments()] == ["打招呼"]
    store.close()


def test_scheduler_registers_memory_job(tmp_path):
    s = make_scheduler(FakeModel([]), 1000, db_path=tmp_path / "m.db", lifecycle_path=tmp_path / "l.json")
    assert [j.name for j in s.jobs] == ["记忆"] and s.run_pending() == ["记忆"] and s.errors == []
