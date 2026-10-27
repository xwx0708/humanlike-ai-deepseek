"""遗忘曲线要满足设想文档里的"大概效果"。"""

import pytest

from memory.forgetting import (
    ForgettingKnobs,
    Trace,
    clamp_adjustment,
    days_until_forgotten,
    initial_half_life,
    recall,
    relative_arousal,
    strength,
)

K = ForgettingKnobs()
MONTH = 30.4


def forget_days(rel_arousal: float, importance: float = 0.0) -> float:
    return days_until_forgotten(initial_half_life(rel_arousal, importance, K), K)


@pytest.mark.parametrize(
    ("rel_arousal", "importance", "low", "high", "meaning"),
    [
        (0.0, 0.0, 18, 28, "平淡小事约三周"),
        (0.5, 0.0, 2.0 * MONTH, 2.8 * MONTH, "有点情绪约两个半月"),
        (0.9, 0.0, 5.5 * MONTH, 6.5 * MONTH, "很强烈约半年"),
        (1.0, 0.0, 7.0 * MONTH, 8.0 * MONTH, "极强烈七八个月"),
        (1.0, 1.0, 2 * 365, 5 * 365, "叠加重要性可保留好几年"),
    ],
)
def test_expected_effects(rel_arousal, importance, low, high, meaning):
    assert low <= forget_days(rel_arousal, importance) <= high, meaning


def test_half_life_means_half():
    assert strength(7.0, 7.0) == pytest.approx(0.5)
    assert strength(7.0, 0.0) == 1.0


def test_relative_arousal_subtracts_baseline():
    assert relative_arousal(0.3, 0.3) == 0.0
    assert relative_arousal(0.2, 0.3) == 0.0  # 低于基线不为负
    assert relative_arousal(1.0, 0.3) == 1.0
    assert relative_arousal(0.65, 0.3) == pytest.approx(0.5)


def test_recall_right_away_barely_reinforces():
    t = Trace(half_life=7.0, last_recall=0.0)
    assert recall(t, 0.01, K).half_life == pytest.approx(7.0, rel=0.01)


def test_recall_late_reinforces_up_to_double():
    t = Trace(half_life=7.0, last_recall=0.0)
    late = recall(t, 60.0, K)
    assert 13.0 < late.half_life <= 14.0
    assert late.strength_at(60.0) == 1.0
    assert late.recall_count == 1


def test_spaced_recall_outlasts_unrecalled():
    plain = Trace(half_life=initial_half_life(0.0, 0.0, K), last_recall=0.0)
    recalled = plain
    for day in (5, 15, 40):
        recalled = recall(recalled, day, K)
    assert plain.is_forgotten(90, K)
    assert not recalled.is_forgotten(90, K)


def test_adjustment_is_clamped():
    assert clamp_adjustment(1.0, 100.0, K) == 3.0
    assert clamp_adjustment(4.0, 3.0, K) == 5.0
    assert clamp_adjustment(1.0, 0.0001, K) == pytest.approx(1 / 3)
