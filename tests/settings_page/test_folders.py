"""数据文件夹用选择器选，不手动输入（用户 2026-11-02 定）：服务端列目录，只列子文件夹。"""

from settings_page.server import list_folders


def test_list_folders_lists_only_directories(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "file.txt").write_text("x", encoding="utf-8")
    d = list_folders(str(tmp_path))
    assert (
        d["dirs"] == ["a", "b"]
        and d["path"] == str(tmp_path.resolve())
        and d["parent"] == str(tmp_path.resolve().parent)
    )
    assert list_folders(str(tmp_path / "nowhere"))["dirs"] == []
    top = list_folders("")
    assert top["parent"] is None and top["dirs"]
