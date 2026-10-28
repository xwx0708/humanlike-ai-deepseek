"""公共运行时（D27）：时钟、后台调度、全局生命周期。"""

from runtime.clock import FakeClock
from runtime.lifecycle import FAMILIAR, RUNNING, Lifecycle
from runtime.scheduler import Scheduler


def test_scheduler_runs_jobs_on_time_and_isolates_errors():
    clock = FakeClock(0)
    seen = []
    sch = Scheduler(clock)
    sch.add("记忆", 60, lambda now: seen.append(("记忆", now)))
    sch.add("情绪", 300, lambda now: 1 / 0)  # 出错不影响别的任务
    assert sch.run_pending() == ["记忆", "情绪"]
    clock.advance(30)
    assert sch.run_pending() == []
    clock.advance(30)
    assert sch.run_pending() == ["记忆"]
    assert seen == [("记忆", 0), ("记忆", 60)]
    assert sch.errors[0][0] == "情绪" and "ZeroDivisionError" in sch.errors[0][2]


def test_lifecycle_runs_every_module_hook_once(tmp_path):
    life = Lifecycle(path=tmp_path / "lifecycle.json")
    order = []
    life.on_start("记忆", lambda now: order.append("记忆") or "概率海")
    life.on_start("情绪", lambda now: order.append("情绪") or "基线")
    assert life.phase == FAMILIAR and life.maybe_start(1, 69, 100) is None  # 69% 还不启动
    assert life.maybe_start(2, 70, 100) == {"记忆": "概率海", "情绪": "基线"}
    assert order == ["记忆", "情绪"] and life.phase == RUNNING and life.started_ts == 2
    assert life.maybe_start(3, 100, 100) is None
    again = Lifecycle(path=tmp_path / "lifecycle.json")
    assert (again.phase, again.started_ts) == (RUNNING, 2)
