"""各家模型的连接项只在选了那家时显示：schema 里带 show_when，页面据此隐藏（用户 2026-11-02 指出）。
思考模型在 AstrBot 里选，不上设置页（D44）。"""

from settings_page.schema import schema


def test_vendor_fields_depend_on_provider_choice():
    f = {x["key"]: x for x in next(s for s in schema()["sections"] if s["id"] == "models")["fields"]}
    assert "thinking_provider" not in f and "thinking_model" not in f
    assert f["ollama_base_url"]["show_when"] == {"subconscious_provider": ["ollama"]}
    assert f["deepseek_thinking"]["show_when"] == {"subconscious_provider": ["deepseek"]}
    assert f["openai_compat_base_url"]["show_when"] == {"subconscious_provider": ["openai_compat"]}
    assert "show_when" not in f["subconscious_model"]
    assert (f["thinking_context_limit"]["min"], f["thinking_context_limit"]["max"]) == (8000, 1000000)
