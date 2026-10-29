"""设置页（D40）：所有参数都有条目和解释；提交的设置经过校验，只存和默认不同的。"""

from settings_page.schema import FEATURES, KNOB_CLASSES, SECTIONS, WORD_LISTS, class_fields
from settings_page.server import clean


def test_every_knob_field_is_on_the_page_with_help():
    by_id = {s["id"]: {f["key"]: f for f in s["fields"]} for s in SECTIONS}
    for sec, cls in KNOB_CLASSES.items():
        for name in class_fields(cls):
            assert name in by_id[sec], f"{sec}.{name} 没有出现在设置页"
            assert by_id[sec][name]["help"], f"{sec}.{name} 缺少解释"
    for s in SECTIONS:
        assert s["help"]
        for f in s["fields"]:
            assert f["help"] and f["label"]
    assert all(f["help"] for f in FEATURES) and all(w["help"] and w["default"] for w in WORD_LISTS)


def test_numeric_defaults_match_code():
    from memory.scanner import ScannerKnobs

    f = {x["key"]: x for x in next(s for s in SECTIONS if s["id"] == "scanner")["fields"]}
    assert f["buffer_size"]["default"] == ScannerKnobs().buffer_size and f["buffer_size"]["type"] == "int"


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
