"""小d 在 QQ 上自己的事：换头像、改昵称、改签名、发说说、删说说（D35）。

- 用户定："在严格限制频率的情况下可以自我决定。" 所以这些动作由思考模型自己决定做不做，不先问用户，
  但每种动作都有频率上限（暂定，见 LIMITS），超过就拒绝，并告诉模型下次最早什么时候能做。
- 这里只负责把关（频率、长度、图片来源）和生成要交给 NapCat 的接口名与参数；真正调用由 AstrBot 插件做，
  调用成功后再 record()。这样这一层不依赖 AstrBot，可以单独测试。
- 图片只能用 data/images/ 里的（暂定）：小d 自己画不了图，用户把候选图片放进这个文件夹。
  以 base64 交给 NapCat，因为 NapCat 在另一个容器里，读不到本项目的文件。
- 删说说只能删自己用这个工具发的。
"""

import base64
import json
from dataclasses import dataclass
from pathlib import Path

from runtime import config
from runtime.clock import Clock, SystemClock

ROOT = Path(__file__).resolve().parents[2]
DAY = 86400.0

AVATAR = "换头像"
NICKNAME = "改昵称"
SIGNATURE = "改签名"
POST = "发说说"
DELETE = "删说说"


@dataclass(frozen=True)
class Limit:
    times: int  # 在 window 秒内最多做几次
    window: float


# 暂定，偏严（用户要求"严格限制频率"）；以后测试时再调
def _limit(key: str, times: int, days: float) -> Limit:
    return Limit(
        config.value("action", f"{key}_times", times), config.value("action", f"{key}_days", days) * DAY
    )


LIMITS = {
    AVATAR: _limit("avatar", 1, 7),
    NICKNAME: _limit("nickname", 1, 7),
    SIGNATURE: _limit("signature", 1, 3),
    POST: _limit("post", 1, 1),
    DELETE: _limit("delete", 2, 1),
}
MAX_CHARS = {NICKNAME: 20, SIGNATURE: 60, POST: 500}
MAX_IMAGES = 4  # 一条说说最多配几张图（暂定）
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".gif", ".webp")
VISIBLE_TO_FRIENDS = config.value(
    "action", "post_visibility", 4
)  # 说说默认好友可见（暂定）；1 所有人、4 好友、64 仅自己


class Refused(Exception):
    """不能做：超过频率、内容不合规、图片不存在等。消息是给思考模型看的。"""


@dataclass(frozen=True)
class Call:
    kind: str  # 动作，例如 发说说
    action: str  # NapCat 接口名
    params: dict
    note: str  # 做成以后记进记忆的一句话（AI 自己做过的事）


class QQSelf:
    def __init__(
        self,
        log_path: Path | None = None,
        images_dir: Path | None = None,
        clock: Clock | None = None,
    ):
        self.log_path = Path(log_path or config.data_dir() / "action_log.json")
        self.images_dir = Path(images_dir or config.data_dir() / "images")
        self.clock = clock or SystemClock()

    # ---------- 生成调用（先检查，不记录） ----------

    def set_avatar(self, image: str) -> Call:
        self._check_rate(AVATAR)
        return Call(AVATAR, "set_qq_avatar", {"file": self._image(image)}, f"我把 QQ 头像换成了图片 {image}")

    def set_nickname(self, nickname: str) -> Call:
        text = self._text(NICKNAME, nickname)
        self._check_rate(NICKNAME)
        return Call(NICKNAME, "set_qq_profile", {"nickname": text}, f"我把 QQ 昵称改成了「{text}」")

    def set_signature(self, signature: str) -> Call:
        text = self._text(SIGNATURE, signature)
        self._check_rate(SIGNATURE)
        return Call(SIGNATURE, "set_self_longnick", {"longNick": text}, f"我把 QQ 签名改成了「{text}」")

    def post(self, content: str, images: list[str] = ()) -> Call:
        text = self._text(POST, content)
        if len(images) > MAX_IMAGES:
            raise Refused(f"一条说说最多配 {MAX_IMAGES} 张图")
        self._check_rate(POST)
        params = {
            "content": text,
            "images": [self._image(i) for i in images],
            "ugc_right": VISIBLE_TO_FRIENDS,
        }
        pics = f"，配图 {'、'.join(images)}" if images else ""
        return Call(POST, "send_qzone_msg", params, f"我在 QQ 空间发了一条说说（好友可见{pics}）：{text}")

    def delete_post(self, tid: str) -> Call:
        posts = {e.get("tid"): e for e in self._log() if e["kind"] == POST}
        if tid not in posts:
            raise Refused("只能删自己发的说说；这个编号不是我发的")
        self._check_rate(DELETE)
        return Call(
            DELETE, "delete_qzone_msg", {"tid": tid}, f"我删掉了之前发的说说：{posts[tid]['content']}"
        )

    # ---------- 做成以后 ----------

    def record(self, call: Call, result: dict | None = None) -> None:
        entry = {"kind": call.kind, "ts": self.clock.now()}
        if call.kind == POST:
            entry |= {"content": call.params["content"], "tid": (result or {}).get("tid")}
        log = self._log() + [entry]
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")

    def list_images(self) -> list[str]:
        if not self.images_dir.is_dir():
            return []
        return sorted(p.name for p in self.images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)

    def my_posts(self) -> list[dict]:
        """自己发过的说说（编号和正文），删说说时用。"""
        return [
            {"tid": e["tid"], "content": e["content"]}
            for e in self._log()
            if e["kind"] == POST and e.get("tid")
        ]

    # ---------- 内部 ----------

    def _log(self) -> list[dict]:
        if not self.log_path.is_file():
            return []
        return json.loads(self.log_path.read_text(encoding="utf-8"))

    def _check_rate(self, kind: str) -> None:
        lim, now = LIMITS[kind], self.clock.now()
        recent = sorted(e["ts"] for e in self._log() if e["kind"] == kind and now - e["ts"] < lim.window)
        if len(recent) >= lim.times:
            wait = recent[0] + lim.window - now
            raise Refused(
                f"「{kind}」太频繁了：{_span(lim.window)}内最多 {lim.times} 次，约 {_span(wait)}后才能再做"
            )

    def _text(self, kind: str, text: str) -> str:
        text = (text or "").strip()
        if not text:
            raise Refused("内容不能为空")
        if len(text) > MAX_CHARS[kind]:
            raise Refused(f"{kind}最多 {MAX_CHARS[kind]} 个字，现在 {len(text)} 个")
        return text

    def _image(self, name: str) -> str:
        p = (self.images_dir / name).resolve()
        if p.parent != self.images_dir.resolve() or p.suffix.lower() not in IMAGE_SUFFIXES or not p.is_file():
            raise Refused(
                f"没有这张图：{name}。能用的图片：{'、'.join(self.list_images()) or '（文件夹里还没有图）'}"
            )
        return "base64://" + base64.b64encode(p.read_bytes()).decode()


def _span(seconds: float) -> str:
    if seconds >= DAY:
        return f"{seconds / DAY:.0f} 天"
    if seconds >= 3600:
        return f"{seconds / 3600:.0f} 小时"
    return f"{max(1, seconds // 60):.0f} 分钟"
