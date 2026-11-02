"""Docker/AstrBot 部署时思考模型以 AstrBot 为准（D29）：设置页读 AstrBot 配置，显示它启用的模型；没有配置就空。"""

import json

from settings_page.server import astrbot_thinking


def test_lists_enabled_astrbot_models_without_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("HUMANLIKE_DATA_DIR", str(tmp_path))
    assert astrbot_thinking() == []
    (tmp_path / "astrbot").mkdir()
    (tmp_path / "astrbot" / "cmd_config.json").write_text(
        json.dumps(
            {
                "provider": [
                    {
                        "id": "deepseek/deepseek-flash",
                        "enable": True,
                        "provider_source_id": "deepseek",
                        "model": "deepseek-flash",
                    },
                    {"id": "old", "enable": False, "provider_source_id": "x", "model": "gone"},
                ],
                "provider_sources": [{"id": "deepseek", "key": ["sk-secret"]}],
            }
        ),
        encoding="utf-8",
    )
    out = astrbot_thinking()
    assert out == [{"id": "deepseek/deepseek-flash", "model": "deepseek-flash", "source": "deepseek"}]
    assert "sk-secret" not in json.dumps(out)
