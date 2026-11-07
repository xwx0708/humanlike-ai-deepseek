"""自己上网找图（用户 2026-11-07）：只收真图片、记出处、限频、自己找的图有上限。不联网，用假的下载函数。"""

import json

import pytest

from action import image_finder
from action.image_finder import PREFIX, ImageFinder, image_suffix
from action.qq_self import QQSelf, Refused
from runtime.clock import FakeClock

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 20
JPG = b"\xff\xd8\xff" + b"0" * 20


def fake_web(results, files):
    """results：图库返回的条目；files：地址 → 内容（缺的地址当作下载失败）。"""
    calls = []

    def fetch(url, limit):
        calls.append(url)
        if url.startswith(image_finder.API):
            return json.dumps({"results": results}).encode()
        if url not in files:
            raise OSError("下不来")
        return files[url]

    fetch.calls = calls
    return fetch


def item(i, url, thumb=None):
    return {
        "id": f"id{i}",
        "title": f"猫 {i}",
        "tags": [{"name": "cat"}, {"name": "orange"}],
        "creator": "某人",
        "license": "by",
        "license_version": "2.0",
        "url": url,
        "thumbnail": thumb,
        "foreign_landing_url": f"https://example.org/{i}",
    }


def test_downloads_real_images_and_records_source(tmp_path):
    web = fake_web(
        [item(1, "u1"), item(2, "u2"), item(3, "u3", thumb="t3")],
        {"u1": PNG, "u2": b"<html>not an image</html>", "t3": JPG},  # u2 不是图片；u3 原图下不来，退回缩略图
    )
    found = ImageFinder(web).find("orange cat", tmp_path)
    assert [f.name.endswith(s) for f, s in zip(found, (".png", ".jpg"), strict=True)] == [True, True]
    assert all(f.name.startswith(PREFIX) and (tmp_path / f.name).is_file() for f in found)
    assert found[0].license == "CC BY 2.0" and found[0].tags == ["cat", "orange"]
    index = json.loads((tmp_path / "found.json").read_text(encoding="utf-8"))
    assert [e["name"] for e in index] == [f.name for f in found]
    assert "mature=false" in web.calls[0]  # 默认滤掉成人内容


def test_keeps_at_most_keep_found_images(tmp_path, monkeypatch):
    monkeypatch.setattr(image_finder, "KEEP", 3)
    (tmp_path / "用户的图.png").write_bytes(PNG)
    for batch in range(2):
        web = fake_web([item(batch * 10 + i, f"u{i}") for i in range(2)], {f"u{i}": PNG for i in range(2)})
        ImageFinder(web).find(f"第{batch}批", tmp_path)
    mine = sorted(p.name for p in tmp_path.iterdir() if p.name.startswith(PREFIX))
    assert len(mine) == 3 and (tmp_path / "用户的图.png").is_file()  # 用户自己放的不动


def test_suffix_by_file_header():
    assert image_suffix(PNG) == ".png" and image_suffix(JPG) == ".jpg"
    assert image_suffix(b"RIFF1234WEBPxxxx") == ".webp"
    assert image_suffix(b"GIF89a...") == ".gif"
    assert image_suffix(b"<svg>") is None


@pytest.fixture
def qq(tmp_path):
    web = fake_web([item(1, "u1")], {"u1": PNG})
    return QQSelf(tmp_path / "log.json", tmp_path / "images", FakeClock(1000.0), ImageFinder(web))


def test_found_images_can_become_avatar(qq):
    [f] = qq.find_images("cat")
    assert f.name in qq.list_images()
    assert qq.set_avatar(f.name).action == "set_qq_avatar"


def test_finding_is_rate_limited_and_empty_query_refused(qq):
    with pytest.raises(Refused, match="找什么"):
        qq.find_images("  ")
    for _ in range(5):
        qq.find_images("cat")
    with pytest.raises(Refused, match="太频繁"):
        qq.find_images("cat")
