"""可替换点（改进三，D42）：遗忘曲线和浮现策略能换成内置的别的实现，也能换成 data/plugins/ 里用户自己写的。"""

import random

import pytest

from contracts.memory import EmotionScore, Segment
from contracts.plugins import Candidate
from memory import forgetting, scanner
from memory.forgetting import ExponentialDecay, NoDecay, PowerLawDecay, Trace, recall, strength
from memory.scanner import NoSurfacing, ProbabilisticSurfacing, RelevanceSurfacing, Scanner, ScannerKnobs
from memory.store import MemoryStore
from runtime import plugins

K = scanner.ScannerKnobs()


def seg(title, priming=0.0):
    return Candidate(Segment(1, 1, title, 0, 0, emotion=EmotionScore(0.5, 0.0, 0.5)), 0.9, priming, 1.0)


def test_builtin_decay_models_agree_at_half_life_and_differ_after():
    for m in (ExponentialDecay(), PowerLawDecay()):
        assert m.strength(7.0, 0.0) == 1.0 and m.strength(7.0, 7.0) == pytest.approx(0.5)
    assert PowerLawDecay().strength(7.0, 70.0) > ExponentialDecay().strength(7.0, 70.0)  # 幂律尾巴长
    assert NoDecay().strength(7.0, 10_000.0) == 1.0
    assert NoDecay().reinforced_half_life(7.0, 0.1, 1.0) == 7.0


def test_default_is_exponential_and_recall_uses_the_active_model(monkeypatch):
    assert forgetting.DECAY.name == "指数衰减"
    assert strength(7.0, 14.0) == pytest.approx(0.25)
    monkeypatch.setattr(forgetting, "DECAY", NoDecay())
    assert strength(7.0, 14.0) == 1.0
    assert recall(Trace(7.0, 0.0), 60.0, forgetting.ForgettingKnobs()).half_life == 7.0


def test_surface_policies():
    cands = [seg("无关"), seg("相关", priming=0.4), seg("更相关", priming=0.5)]
    rng = random.Random(1)
    assert RelevanceSurfacing().surface(cands, 0.2, ScannerKnobs(buffer_size=1), rng)[0].title == "更相关"
    assert NoSurfacing().surface(cands, 0.2, K, rng) == []
    p = ProbabilisticSurfacing()
    assert p.probability(cands[2], 0.2, K) > p.probability(cands[0], 0.2, K)


def test_scanner_uses_the_chosen_policy():
    store = MemoryStore(":memory:")
    c = store.new_conversation(0)
    t = store.add_turn(c.id, "user", "去爬山", 0)
    store.add_segment(Segment(None, c.id, "去爬山", 0, 0, [t.id], emotion=EmotionScore(0.5, 0.2, 0.5)))
    sc = Scanner(store, policy=NoSurfacing(), rng=random.Random(0))
    assert all(sc.tick(60.0 * i) == [] for i in range(1, 200))
    sc = Scanner(store, policy=RelevanceSurfacing(), rng=random.Random(0))
    assert sc.tick(60.0) == []  # 没设话题：无关，不冒头
    sc.set_topic("周末去爬山")
    assert [s.text for s in sc.tick(120.0)] == ["去爬山"]
    store.close()


def test_user_plugin_file_is_loaded_and_selectable(tmp_path):
    """验收：用一个"不遗忘"插件跑通——放一个 Python 文件进 plugins/，按名字选用。"""
    (tmp_path / "plugins").mkdir()
    (tmp_path / "plugins" / "my_decay.py").write_text(
        """
class NeverForget:
    plugin_kind = "decay"
    name = "我的不遗忘"

    def strength(self, half_life, elapsed_days):
        return 1.0

    def reinforced_half_life(self, half_life, strength_now, reinforce):
        return half_life * 2
""",
        encoding="utf-8",
    )
    (tmp_path / "plugins" / "broken.py").write_text("this is not python", encoding="utf-8")
    found = plugins.load_plugins(tmp_path / "plugins")
    assert list(found["decay"]) == ["我的不遗忘"] and found["surface"] == {}
    assert plugins.errors and plugins.errors[0][0] == "broken.py"  # 坏文件只跳过
    assert plugins.names("decay", forgetting.DECAY_MODELS) == ["指数衰减", "幂律衰减", "不遗忘", "我的不遗忘"]
    chosen = plugins.choose("decay", "我的不遗忘", forgetting.DECAY_MODELS)
    assert chosen.strength(7.0, 10_000.0) == 1.0 and chosen.reinforced_half_life(7.0, 0.5, 1.0) == 14.0
    assert plugins.choose("decay", "不存在", forgetting.DECAY_MODELS).name == "指数衰减"  # 回落到默认
    plugins.load_plugins(tmp_path / "nowhere")  # 还原：别的测试用空目录
