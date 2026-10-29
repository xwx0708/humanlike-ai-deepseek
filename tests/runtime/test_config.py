"""用户设置（D40）：settings.json 覆盖默认参数、功能开关、自定义词表、数据文件夹。"""

import json

import pytest

from memory.scanner import ScannerKnobs
from runtime import config


@pytest.fixture
def settings(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    monkeypatch.setenv("HUMANLIKE_SETTINGS", str(path))
    monkeypatch.delenv("HUMANLIKE_DATA_DIR", raising=False)

    def write(data):
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    return write


def test_no_file_means_defaults(settings):
    assert config.knobs(ScannerKnobs(), "scanner") == ScannerKnobs()
    assert config.value("lifecycle", "start_ratio", 0.7) == 0.7
    assert config.enabled("web_search") is True


def test_overrides_are_cast_and_unknown_keys_ignored(settings):
    settings(
        {"scanner": {"buffer_size": "6", "threshold": 0.5, "nonsense": 1}, "lifecycle": {"start_ratio": 0.8}}
    )
    k = config.knobs(ScannerKnobs(), "scanner")
    assert (k.buffer_size, k.threshold, k.rate) == (6, 0.5, ScannerKnobs().rate)
    assert config.value("lifecycle", "start_ratio", 0.7) == 0.8


def test_broken_file_falls_back_to_defaults(settings, tmp_path):
    (tmp_path / "settings.json").write_text("{坏了", encoding="utf-8")
    assert config.load() == {}


def test_features_and_word_lists(settings):
    settings(
        {
            "features": {"web_search": False},
            "keywords": {"remember": {"add": ["刻在脑子里", " "], "disabled": ["记一下"]}},
        }
    )
    assert config.enabled("web_search") is False and config.enabled("read_files") is True
    assert config.words("remember", ("记住", "记一下")) == ("记住", "刻在脑子里")


def test_data_dir(settings, tmp_path, monkeypatch):
    assert config.data_dir() == config.ROOT / "data"
    settings({"general": {"data_dir": "我的记忆"}})
    assert config.data_dir() == config.ROOT / "我的记忆"  # 相对路径按项目根目录
    settings({"general": {"data_dir": str(tmp_path / "abs")}})
    assert config.data_dir() == tmp_path / "abs"
    monkeypatch.setenv("HUMANLIKE_DATA_DIR", str(tmp_path / "env"))
    assert config.data_dir() == tmp_path / "env"  # 环境变量优先


def test_model_settings_page_layer(settings, monkeypatch, tmp_path):
    """模型配置：环境变量 > 设置页 > .env > 默认。"""
    from llm.settings import load_settings

    monkeypatch.delenv("SUBCONSCIOUS_MODEL", raising=False)
    dotenv = tmp_path / ".env"
    dotenv.write_text("SUBCONSCIOUS_MODEL=from-dotenv\nDEEPSEEK_THINKING=false\n", encoding="utf-8")
    assert load_settings(dotenv).subconscious_model == "from-dotenv"
    settings({"models": {"subconscious_model": "from-page", "deepseek_thinking": True}})
    s = load_settings(dotenv)
    assert s.subconscious_model == "from-page" and s.deepseek_thinking is True
    monkeypatch.setenv("SUBCONSCIOUS_MODEL", "from-env")
    assert load_settings(dotenv).subconscious_model == "from-env"


def test_user_forgetting_settings_win_over_learned_knobs(settings):
    """全局遗忘参数：设置页写了的以用户为准，没写的用潜意识调过的值。"""
    from memory.forgetting import ForgettingKnobs
    from memory.store import MemoryStore

    store = MemoryStore(":memory:")
    store.save_global_knobs(ForgettingKnobs(base_half_life=9.0, forget_line=0.12))
    assert store.load_global_knobs().base_half_life == 9.0
    settings({"forgetting": {"base_half_life": 3.0}})
    k = store.load_global_knobs()
    assert (k.base_half_life, k.forget_line) == (3.0, 0.12)
