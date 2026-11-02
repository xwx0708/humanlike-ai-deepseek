"""设置页里直接写人设或从文本文件读入，存到数据文件夹的 persona.md（用户 2026-11-02 定）。"""

from settings_page.server import persona_path, read_persona, write_persona


def test_persona_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("HUMANLIKE_DATA_DIR", str(tmp_path))
    assert read_persona() == {"path": str(tmp_path / "persona.md"), "text": ""}
    out = write_persona("你叫大肥鱼。\r\n说话慢。")
    assert (
        out["path"] == str(persona_path())
        and (tmp_path / "persona.md").read_text(encoding="utf-8") == "你叫大肥鱼。\n说话慢。"
    )
    assert read_persona()["text"] == "你叫大肥鱼。\n说话慢。"
