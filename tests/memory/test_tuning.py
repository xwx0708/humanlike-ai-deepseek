"""全局参数护栏：窄幅、慢速、留日志。"""

import pytest

from memory.forgetting import ForgettingKnobs
from memory.tuning import TUNABLE, adjust_global, knob_names

K = ForgettingKnobs()


def test_every_tunable_knob_exists():
    assert set(TUNABLE) <= set(knob_names())


def test_small_change_is_applied_and_logged():
    log = []
    k, c = adjust_global(K, "base_half_life", 1.05, now=0, reason="用户记性好", log=log)
    assert k.base_half_life == pytest.approx(7.35)
    assert c.applied and c.old == 7.0 and c.note == ""


def test_big_change_is_capped_per_step():
    k, c = adjust_global(K, "base_half_life", 3.0, now=0, reason="", log=[])
    assert k.base_half_life == pytest.approx(7.7)
    assert c.requested == pytest.approx(21.0)
    assert "单次限幅" in c.note


def test_cannot_drift_beyond_range_of_default():
    k, log = K, []
    for week in range(20):
        k, c = adjust_global(k, "emotion_max", 1.1, now=week * 7, reason="", log=log)
        log.append(c)
    assert k.emotion_max == pytest.approx(15.0)  # 默认 10 × 1.5
    assert not log[-1].applied


def test_cooldown_rejects_but_still_logs():
    log = []
    k, c = adjust_global(K, "forget_line", 0.95, now=0, reason="", log=log)
    log.append(c)
    k2, c2 = adjust_global(k, "forget_line", 0.95, now=3, reason="又想改", log=log)
    assert k2 == k and not c2.applied and "冷却中" in c2.note
    _, c3 = adjust_global(k, "forget_line", 0.95, now=7, reason="", log=log)
    assert c3.applied


def test_cooldown_is_per_knob():
    log = []
    _, c = adjust_global(K, "reinforce", 1.05, now=0, reason="", log=log)
    log.append(c)
    _, c2 = adjust_global(K, "importance_max", 1.05, now=1, reason="", log=log)
    assert c2.applied


def test_cannot_loosen_own_limits():
    with pytest.raises(ValueError):
        adjust_global(K, "adjust_total_max", 1.05, now=0, reason="", log=[])
