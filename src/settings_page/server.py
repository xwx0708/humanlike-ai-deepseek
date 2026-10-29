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
from settings_page.schema import FEATURES, SECTIONS, WORD_LISTS, schema

PAGE = Path(__file__).with_name("page.html")


def clean(raw: dict[str, Any]) -> dict[str, Any]:
    """按说明表校验用户提交的设置：类型转换、范围裁剪、去掉不认识的项、只保留和默认值不同的。"""
    out: dict[str, Any] = {}
    for sec in SECTIONS:
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
    off = {f["key"]: False for f in FEATURES if feats.get(f["key"]) is False}
    if off:
        out["features"] = off
    words = raw.get("keywords") or {}
    kw = {}
    for w in WORD_LISTS:
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
                200, {"schema": schema(), "values": config.load(), "path": str(config.settings_path())}
            )
        else:
            self._json(404, {"error": "没有这个地址"})

    def do_POST(self) -> None:
        if self.path != "/api/settings":
            self._json(404, {"error": "没有这个地址"})
            return
        try:
            raw = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "提交的内容不是合法的 JSON"})
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
