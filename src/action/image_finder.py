"""自己上网找图（用户 2026-11-07："我希望它可以有自己给自己换头像，找图片的能力"）。

补充（暂定，待用户确认）：
- 来源用 Openverse（api.openverse.org）：不用密钥、只收知识共享（CC）许可和公有领域的图，默认滤掉成人内容。
  国内能连上（2026-11-07 从 AstrBot 容器实测约 5 秒）。
- 找到的图下载进数据文件夹的 images/，和用户放的图在一起，换头像、给说说配图都能用。文件名以"找到_"开头，
  出处、作者、许可记在 images/found.json，方便以后署名或追溯。
- 只收真正的图片：看文件头（JPEG、PNG、GIF、WEBP），单张不超过 MAX_BYTES。原图太大时退回用缩略图。
- 自己找来的图最多留 KEEP 张，超出时删最早找到的（用户自己放的图不动）。
- 思考模型看不到图本身，只能按标题和标签挑（限制，见 04-行动/README.md）。
"""

import json
import logging
import re
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

API = "https://api.openverse.org/v1/images/"
USER_AGENT = "humanlike-ai/0.1 (https://github.com/xwx0708/humanlike-ai-deepseek)"
PREFIX = "找到_"
MAX_BYTES = 5 * 1024 * 1024  # 单张上限（暂定）
KEEP = 40  # 自己找来的图最多留几张（暂定）
TIMEOUT = 20
log = logging.getLogger(__name__)

MAGIC = {b"\xff\xd8\xff": ".jpg", b"\x89PNG\r\n\x1a\n": ".png", b"GIF87a": ".gif", b"GIF89a": ".gif"}

Fetch = Callable[[str, int], bytes]  # (地址, 最多读多少字节) → 内容


@dataclass(frozen=True)
class Found:
    name: str  # images/ 里的文件名，换头像、配图时用
    title: str
    tags: list[str]
    creator: str
    license: str  # 例如 "CC BY 2.0"
    source: str  # 原图所在网页


def http_fetch(url: str, limit: int) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        data = r.read(limit + 1)
    if len(data) > limit:
        raise ValueError("太大")
    return data


def image_suffix(data: bytes) -> str | None:
    """按文件头认图片格式；不是图片返回 None。"""
    for magic, suffix in MAGIC.items():
        if data.startswith(magic):
            return suffix
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    return None


class ImageFinder:
    def __init__(self, fetch: Fetch = http_fetch):
        self.fetch = fetch

    def find(self, query: str, images_dir: Path, count: int = 4) -> list[Found]:
        """搜 query，最多下载 count 张进 images_dir。网络不通时抛出异常，由调用方转告思考模型。"""
        url = API + "?" + urllib.parse.urlencode({"q": query, "page_size": count * 3, "mature": "false"})
        results = json.loads(self.fetch(url, 2 * 1024 * 1024)).get("results", [])
        images_dir.mkdir(parents=True, exist_ok=True)
        out = []
        for r in results:
            if len(out) >= count:
                break
            data = self._download(r)
            suffix = image_suffix(data) if data else None
            if not suffix:
                continue
            name = f"{PREFIX}{_slug(query)}_{_slug(str(r.get('id', '')))[:8]}{suffix}"
            (images_dir / name).write_bytes(data)
            out.append(
                Found(
                    name,
                    r.get("title") or "",
                    [t["name"] for t in r.get("tags") or [] if t.get("name")][:8],
                    r.get("creator") or "",
                    f"CC {str(r.get('license', '')).upper()} {r.get('license_version') or ''}".strip(),
                    r.get("foreign_landing_url") or r.get("url") or "",
                )
            )
        self._remember(images_dir, out)
        return out

    def _download(self, r: dict) -> bytes | None:
        for url in (r.get("url"), r.get("thumbnail")):  # 原图太大或下不来，退回缩略图
            if not url:
                continue
            try:
                return self.fetch(url, MAX_BYTES)
            except Exception as e:  # noqa: BLE001  一张下不来就换下一张
                log.info("找图：下载 %s 失败（%s）", url, e)
        return None

    def _remember(self, images_dir: Path, found: list[Found]) -> None:
        """出处记进 found.json；自己找来的超过 KEEP 张时删最早的。"""
        index_path = images_dir / "found.json"
        index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else []
        index = [e for e in index if e["name"] not in {f.name for f in found}] + [asdict(f) for f in found]
        while len(index) > KEEP:
            (images_dir / index.pop(0)["name"]).unlink(missing_ok=True)
        index = [e for e in index if (images_dir / e["name"]).is_file()]
        index_path.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")


def _slug(text: str) -> str:
    """文件名里只留字母、数字、汉字，最多 12 个字。"""
    return re.sub(r"[^\w一-鿿]+", "", text)[:12] or "图"
