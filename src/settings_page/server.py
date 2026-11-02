"""设置页的本地服务器（D40），只用标准库。

GET  /              设置页
GET  /api/settings  参数说明表 + 当前设置
POST /api/settings  校验后写入 settings.json（只存和默认值不同的项）
"""

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from runtime import config
from settings_page.schema import schema

PAGE = Path(__file__).with_name("page.html")


def list_folders(raw: str) -> dict[str, Any]:
    """数据文件夹的选择器用：列出一个目录下的子文件夹。空路径时列出各个盘（Windows）或根目录。
    只列目录名，不读文件；读不了的目录（无权限）当作空。"""
    import os
    import string

    if not raw:
        drives = (
            [f"{d}:\\" for d in string.ascii_uppercase if Path(f"{d}:\\").exists()]
            if os.name == "nt"
            else ["/"]
        )
        return {"path": "", "parent": None, "dirs": drives}
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = config.ROOT / p
    try:
        p = p.resolve()
        dirs = sorted(
            (c.name for c in p.iterdir() if c.is_dir() and not c.name.startswith(".")), key=str.lower
        )
    except OSError:
        dirs = []
    parent = None if p.parent == p else str(p.parent)
    if os.name == "nt" and p.parent == p:
        parent = ""  # 盘的上一级：盘列表
    return {"path": str(p), "parent": parent, "dirs": dirs}


def persona_path() -> Path:
    """人设文件：数据文件夹里的 persona.md（和 response.persona 的位置一致，改了立即生效）。"""
    return config.data_dir() / "persona.md"


def read_persona() -> dict[str, str]:
    p = persona_path()
    return {"path": str(p), "text": p.read_text(encoding="utf-8") if p.is_file() else ""}


def write_persona(text: str) -> dict[str, str]:
    p = persona_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text.replace("\r\n", "\n"), encoding="utf-8")
    return {"path": str(p), "chars": str(len(text))}


def astrbot_thinking() -> list[dict[str, str]]:
    """Docker/AstrBot 部署时思考模型以 AstrBot 为准（D29）：读 AstrBot 的配置，列出它启用的模型给设置页看。
    文件在数据文件夹的 astrbot/cmd_config.json（docker-compose.yml 挂载的位置）；没有就返回空。只读模型名，不碰密钥。"""
    p = config.data_dir() / "astrbot" / "cmd_config.json"
    try:
        providers = json.loads(p.read_text(encoding="utf-8")).get("provider") or []
    except (OSError, ValueError):
        return []
    return [
        {
            "id": str(x.get("id", "")),
            "model": str(x.get("model", "")),
            "source": str(x.get("provider_source_id", "")),
        }
        for x in providers
        if isinstance(x, dict) and x.get("enable", True) and x.get("model")
    ]


EN = Path(__file__).with_name("en.json")  # 英文文案；页面默认中文，按钮切换


def clean(raw: dict[str, Any]) -> dict[str, Any]:
    """按说明表校验用户提交的设置：类型转换、范围裁剪、去掉不认识的项、只保留和默认值不同的。"""
    spec = schema()
    out: dict[str, Any] = {}
    for sec in spec["sections"]:
        got = raw.get(sec["id"]) or {}
        kept = {}
        for f in sec["fields"]:
            if f["key"] not in got or got[f["key"]] in (None, ""):
                continue
            v = got[f["key"]]
            t = f["type"]
            try:
                if t in ("int", "float"):
                    v = round(float(v)) if t == "int" else float(v)
                    if f.get("min") is not None:
                        v = max(f["min"], v)
                    if f.get("max") is not None:
                        v = min(f["max"], v)
                elif t == "bool":
                    v = bool(v)
                elif t == "select":
                    opts = f["options"]
                    v = next((o for o in opts if str(o) == str(v)), None)
                    if v is None:
                        continue
                else:
                    v = str(v).strip()
            except (TypeError, ValueError):
                continue
            if v != f["default"]:
                kept[f["key"]] = v
        if kept:
            out[sec["id"]] = kept
    feats = raw.get("features") or {}
    off = {f["key"]: False for f in spec["features"] if feats.get(f["key"]) is False}
    if off:
        out["features"] = off
    words = raw.get("keywords") or {}
    kw = {}
    for w in spec["word_lists"]:
        got = words.get(w["key"]) or {}
        add = [
            s.strip()
            for s in got.get("add") or []
            if isinstance(s, str) and s.strip() and s.strip() not in w["default"]
        ]
        disabled = [s for s in got.get("disabled") or [] if s in w["default"]]
        if add or disabled:
            kw[w["key"]] = {"add": list(dict.fromkeys(add)), "disabled": disabled}
    if kw:
        out["keywords"] = kw
    return out


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, data: Any) -> None:
        self._send(
            code, json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8"
        )

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/settings":
            self._json(
                200,
                {
                    "schema": schema(),
                    "values": config.load(),
                    "path": str(config.settings_path()),
                    "en": json.loads(EN.read_text(encoding="utf-8")),
                    "astrbot": astrbot_thinking(),
                },
            )
        elif self.path == "/api/persona" or self.path == "/api/persona":
            self._json(200, read_persona())
        elif self.path.startswith("/api/folders"):
            from urllib.parse import parse_qs, urlparse

            raw = parse_qs(urlparse(self.path).query).get("path", [""])[0]
            self._json(200, list_folders(raw))
        else:
            self._json(404, {"error": "没有这个地址"})

    def do_POST(self) -> None:
        if self.path not in ("/api/settings", "/api/persona"):
            self._json(404, {"error": "没有这个地址"})
            return
        try:
            raw = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "提交的内容不是合法的 JSON"})
            return
        if self.path == "/api/persona":
            text = raw.get("text") if isinstance(raw, dict) else None
            if not isinstance(text, str):
                self._json(400, {"error": "缺少人设文字"})
                return
            self._json(200, write_persona(text))
            return
        data = clean(raw if isinstance(raw, dict) else {})
        config.save(data)
        self._json(200, {"saved": data, "path": str(config.settings_path())})

    def log_message(self, *args) -> None:  # 不刷屏
        pass


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="类人 AI 设置页")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8791)
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(f"设置页：http://127.0.0.1:{args.port}  （设置文件：{config.settings_path()}）")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
