"""设置页（D40、D42）：参数由各模块登记到注册表，每项都有名称和解释；提交的设置经过校验，只存和默认不同的。"""

from dataclasses import dataclass

from contracts.knobs import REGISTRY, knob
from runtime import config
from settings_page.schema import ORDER, schema
from settings_page.server import clean

FEATURES = {
    "keyword_requests", "persona_reminder", "mid_chat_split", "nightly_merge", "batch_sweep", "read_images",
    "read_files", "web_search", "qq_set_avatar", "qq_set_nickname", "qq_set_signature", "qq_post_moment",
    "qq_delete_moment",
}  # fmt: skip
WORD_LISTS = {"remember", "forget", "archive", "choice_erase", "choice_archive", "choice_cancel"}


def test_every_module_registered_its_settings():
    """config.load_modules() 要能走到每个登记参数的文件：分区、功能开关、词表一个不少。"""
    s = schema()
    assert [x["id"] for x in s["sections"]][: len(ORDER)] == list(ORDER)
    assert {f["key"] for f in s["features"]} == FEATURES
    assert {w["key"] for w in s["word_lists"]} == WORD_LISTS


def test_every_knob_has_label_and_help():
    for sec in schema()["sections"]:
        assert sec["title"] and sec["help"], sec["id"]
        for f in sec["fields"]:
            assert f["label"] and f["help"], f"{sec['id']}.{f['key']} 缺少名称或解释"
            assert f["type"] in ("int", "float", "bool", "text", "select", "path")
    assert all(f["help"] for f in schema()["features"])
    assert all(w["help"] and w["default"] for w in schema()["word_lists"])


def test_secrets_stay_off_the_page():
    keys = {f["key"] for f in next(s for s in schema()["sections"] if s["id"] == "models")["fields"]}
    assert "thinking_context_limit" in keys and not any("api_key" in k for k in keys)


def test_numeric_defaults_match_code():
    from memory.scanner import ScannerKnobs

    f = {x["key"]: x for x in next(s for s in schema()["sections"] if s["id"] == "scanner")["fields"]}
    assert f["buffer_size"]["default"] == ScannerKnobs().buffer_size and f["buffer_size"]["type"] == "int"


def test_new_field_appears_without_touching_the_page():
    """在参数类里新加一个字段，不改设置页代码，页面上它就出现了（改进二的验收标准）。"""

    @dataclass(frozen=True)
    class TrialKnobs:
        depth: int = knob(3, "深度", "试验用。", lo=1, hi=9)
        hidden_key: str = knob("", secret=True)

    try:
        assert config.knobs(TrialKnobs(), "trial", "试验", "只在测试里出现。").depth == 3
        sec = next(s for s in schema()["sections"] if s["id"] == "trial")
        assert sec["title"] == "试验" and [f["key"] for f in sec["fields"]] == ["depth"]
        assert sec["fields"][0]["min"] == 1 and sec["fields"][0]["type"] == "int"
    finally:
        REGISTRY.sections.pop("trial", None)


def test_clean_casts_clamps_and_drops_defaults():
    out = clean(
        {
            "scanner": {"buffer_size": "6.4", "threshold": 99, "rate": 0.02, "bogus": 1},
            "models": {"subconscious_provider": "nonsense", "deepseek_thinking": True},
            "action": {"post_visibility": "64"},
            "general": {"data_dir": "  我的记忆  "},
            "features": {"web_search": False, "read_files": True},
            "keywords": {
                "remember": {
                    "add": ["记住", "刻在脑子里", "刻在脑子里"],
                    "disabled": ["记一下", "不存在的词"],
                }
            },
            "unknown": {"x": 1},
        }
    )
    assert out["scanner"] == {"buffer_size": 6, "threshold": 1.5}  # 取整、裁剪到上限；等于默认的不存
    assert out["models"] == {"deepseek_thinking": True}  # 不在选项里的丢掉
    assert out["action"] == {"post_visibility": 64}
    assert out["general"] == {"data_dir": "我的记忆"}
    assert out["features"] == {"web_search": False}
    assert out["keywords"] == {"remember": {"add": ["刻在脑子里"], "disabled": ["记一下"]}}
    assert "unknown" not in out


def test_clean_empty_is_empty():
    assert clean({}) == {}
