"""调参日志的查看与回滚（不联网）。"""

from dataclasses import replace

import pytest

from contracts.memory import Segment
from memory.prompts import Adjustment
from memory.store import MemoryStore
from memory.subconscious import apply_adjustments
from memory.tuning_review import format_entries, list_tuning, main, rollback

DAY = 86400.0


@pytest.fixture
def store():
    s = MemoryStore(":memory:")
    yield s
    s.close()


def _segment(store, title="他妈妈最近老咳嗽", finished=True):
    c = store.new_conversation(0)
    t = store.add_turn(c.id, "user", title, 0)
    seg = store.add_segment(Segment(None, c.id, title, 0, 0, [t.id], half_life=10.0))
    th = store.new_thread("妈妈的健康", "", 0)
    store.attach(th.id, seg.id, 1.0, 0)
    store.update_thread(th.id, ongoing=not finished)
    return seg


def test_list_and_format(store):
    seg = _segment(store)
    apply_adjustments(
        store, [Adjustment("memory", str(seg.id), 2, "还在担心")], 20 * DAY, {"segment": seg.id}
    )
    apply_adjustments(store, [Adjustment("global", "base_half_life", 1.05, "整体忘太快")], 20 * DAY)
    apply_adjustments(store, [Adjustment("global", "base_half_life", 1.05, "再拉长")], 21 * DAY)  # 冷却中

    all_ = list_tuning(store)
    assert [e.id for e in all_] == [1, 2, 3]
    e = all_[0]
    assert e.target_title == "他妈妈最近老咳嗽" and e.evidence_title == "他妈妈最近老咳嗽"
    assert "从 1 调到 2" in e.description and "理由：还在担心" in e.description
    assert [e.id for e in list_tuning(store, scope="global")] == [2, 3]
    assert [e.id for e in list_tuning(store, applied=False)] == [3]
    assert [e.id for e in list_tuning(store, since_ts=20.5 * DAY)] == [3]
    assert [e.id for e in list_tuning(store, until_ts=20.5 * DAY)] == [1, 2]
    text = format_entries(list_tuning(store))
    assert "第 1 条 [单条记忆·生效]" in text and "基础半衰期" in text and "未生效" in text
    assert format_entries([]) == "（没有调参日志）"


def test_rollback_memory_restores_adjustment_and_half_life(store):
    seg = _segment(store)
    apply_adjustments(store, [Adjustment("memory", str(seg.id), 2, "拉长")], 20 * DAY)
    assert store.get_segment(seg.id).half_life == pytest.approx(20.0)
    r = rollback(store, 1, 22 * DAY)
    s = store.get_segment(seg.id)
    assert s.adjustment == 1.0 and s.half_life == pytest.approx(10.0)
    # 回滚本身也记一条日志
    log = list_tuning(store)
    assert r.log_id == 2 and log[1].reason == "回滚第 1 条" and log[1].applied
    assert log[1].evidence == {"rollback_of": 1} and log[1].rollback_of == 1
    assert log[0].rolled_back_by == [2] and "已被第 2 条回滚" in log[0].description
    with pytest.raises(ValueError, match="已被第 2 条回滚"):
        rollback(store, 1, 23 * DAY)


def test_rollback_superseded_memory_undoes_factor(store):
    seg = _segment(store)
    apply_adjustments(store, [Adjustment("memory", str(seg.id), 2, "a")], 20 * DAY)
    apply_adjustments(store, [Adjustment("memory", str(seg.id), 1.5, "b")], 21 * DAY)  # 系数 3
    r = rollback(store, 1, 22 * DAY)
    s = store.get_segment(seg.id)
    assert s.adjustment == pytest.approx(1.5) and s.half_life == pytest.approx(15.0)
    assert "已被第 2 条后续调整覆盖" in r.change.note


def test_manual_rollback_ignores_d19_floor(store):
    """D26：人工回滚不受护栏限制，D19 也不管。"""
    seg = _segment(store, finished=False)  # 事件进行中
    apply_adjustments(store, [Adjustment("memory", str(seg.id), 2, "a")], 20 * DAY)  # 1 → 2
    apply_adjustments(store, [Adjustment("memory", str(seg.id), 0.5, "b")], 21 * DAY)  # 2 → 1
    r = rollback(store, 1, 22 * DAY)  # 撤销 ×2：得到 0.5
    assert store.get_segment(seg.id).adjustment == pytest.approx(0.5) and r.change.applied
    assert list_tuning(store)[-1].reason == "回滚第 1 条"


def test_rollback_global_ignores_cooldown(store):
    apply_adjustments(store, [Adjustment("global", "base_half_life", 1.1, "a")], 20 * DAY)
    assert store.load_global_knobs().base_half_life == pytest.approx(7.7)
    r = rollback(store, 1, 20.5 * DAY)  # 冷却期内也能回滚
    assert r.change.applied and store.load_global_knobs().base_half_life == pytest.approx(7.0)
    assert list_tuning(store, scope="global")[-1].reason == "回滚第 1 条"


def test_rollback_global_superseded_ignores_range_and_cooldown(store):
    # 先手动把参数放到上界附近，模拟多次调整后的状态
    k = store.load_global_knobs()
    store.save_global_knobs(replace(k, forget_line=0.06875))  # 默认 0.1 的 1/1.5 ≈ 0.0667 以上
    apply_adjustments(store, [Adjustment("global", "forget_line", 1.1, "抬高")], 0)  # 0.06875 → 0.075625
    apply_adjustments(store, [Adjustment("global", "forget_line", 0.95, "降低")], 8 * DAY)  # → 0.07184
    r = rollback(store, 1, 9 * DAY)  # 撤销 ×1.1 → 0.0653，低于下界 0.0667 也照样（D26：人工回滚不受护栏）
    assert store.load_global_knobs().forget_line == pytest.approx(0.071843750 / 1.1)
    assert "后续调整覆盖" in r.change.note
    # 回滚（第 9 天）不计入潜意识的冷却：第 16 天距上次潜意识修改（第 8 天）已过 7 天，可以再改
    out = apply_adjustments(store, [Adjustment("global", "forget_line", 1.05, "c")], 16 * DAY)
    assert out[0][2].applied


def test_rollback_rejects_unapplied_and_missing(store):
    apply_adjustments(store, [Adjustment("global", "base_half_life", 1.1, "a")], 20 * DAY)
    apply_adjustments(store, [Adjustment("global", "base_half_life", 1.1, "b")], 21 * DAY)  # 冷却，未生效
    with pytest.raises(ValueError, match="没有生效过"):
        rollback(store, 2, 22 * DAY)
    with pytest.raises(ValueError, match="没有第 99 条"):
        rollback(store, 99, 22 * DAY)


def test_cli_lists_and_rolls_back(tmp_path, capsys):
    path = tmp_path / "memory.db"
    s = MemoryStore(path)
    apply_adjustments(s, [Adjustment("global", "reinforce", 1.1, "回忆不够牢")], 20 * DAY)
    s.close()
    assert main(["--db", str(path), "--scope", "global"]) == 0
    assert "回忆加固" in capsys.readouterr().out
    assert main(["--db", str(path), "--rollback", "1"]) == 0
    assert "已回滚第 1 条" in capsys.readouterr().out
    s = MemoryStore(path)
    assert s.load_global_knobs().reinforce == pytest.approx(1.0)
    s.close()
    assert main(["--db", str(tmp_path / "none.db")]) == 1
