"""小d 在 QQ 上自己的事（D35）：可以自己决定，但严格限制频率；图片只用 data/images 里的；只能删自己发的说说。"""

import base64

import pytest

from action.qq_self import DAY, POST, QQSelf, Refused
from runtime.clock import FakeClock


@pytest.fixture
def qq(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "猫.png").write_bytes(b"png-bytes")
    (images / "说明.txt").write_text("不是图片", encoding="utf-8")
    return QQSelf(tmp_path / "log.json", images, FakeClock(1000.0))


def test_post_builds_napcat_call_and_is_friends_only(qq):
    c = qq.post("今天天气不错", ["猫.png"])
    assert c.action == "send_qzone_msg"
    assert c.params["content"] == "今天天气不错" and c.params["ugc_right"] == 4  # 默认好友可见
    assert c.params["images"] == ["base64://" + base64.b64encode(b"png-bytes").decode()]
    assert "发了一条说说" in c.note


def test_post_at_most_once_a_day(qq):
    qq.record(qq.post("第一条"), {"tid": "t1"})
    with pytest.raises(Refused, match="太频繁"):
        qq.post("第二条")
    qq.clock.advance(DAY + 1)
    assert qq.post("第二天").kind == POST


def test_checking_alone_does_not_use_up_the_quota(qq):
    """只有真正做成（record）才算一次；调用失败不占名额。"""
    qq.post("试试")
    assert qq.post("再试").kind == POST


def test_avatar_once_a_week_and_only_from_images_folder(qq):
    with pytest.raises(Refused, match="没有这张图"):
        qq.set_avatar("../../secret.png")
    with pytest.raises(Refused, match="能用的图片：猫.png"):
        qq.set_avatar("说明.txt")
    qq.record(qq.set_avatar("猫.png"))
    qq.clock.advance(6 * DAY)
    with pytest.raises(Refused, match="7 天内最多 1 次"):
        qq.set_avatar("猫.png")
    assert qq.list_images() == ["猫.png"]


def test_text_limits(qq):
    with pytest.raises(Refused, match="最多 20 个字"):
        qq.set_nickname("名" * 21)
    with pytest.raises(Refused, match="不能为空"):
        qq.set_signature("  ")
    assert qq.set_signature("摸鱼中").params == {"longNick": "摸鱼中"}


def test_can_only_delete_own_posts(qq):
    qq.record(qq.post("发错了"), {"tid": "t1"})
    with pytest.raises(Refused, match="只能删自己发的"):
        qq.delete_post("别人的")
    c = qq.delete_post("t1")
    assert c.params == {"tid": "t1"} and "发错了" in c.note
    assert qq.my_posts() == [{"tid": "t1", "content": "发错了"}]
