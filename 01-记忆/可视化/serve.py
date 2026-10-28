"""记忆可视化的本地服务器：提供本文件夹里的静态文件，文件一有改动，所有打开的页面自动刷新。

用法：python serve.py [端口]，默认 8790。只用标准库。
另有只读的 JSON 接口 /api/memory、/api/docs，见 api.py。
"""

import http.server
import json
import sys
import threading
import time
from pathlib import Path

sys.dont_write_bytecode = True  # 不在本文件夹写 __pycache__，否则会触发页面刷新
import api  # noqa: E402

ROOT = Path(__file__).resolve().parent
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8790

# 注入到每个 html 页面末尾：记住第一次收到的版本号，之后版本变了就刷新。
# 服务器重启后 EventSource 会自动重连，拿到新版本号同样会刷新。
RELOAD_SNIPPET = b"""
<script>
(() => {
  let first = null;
  new EventSource('/__reload').onmessage = e => {
    if (first === null) first = e.data;
    else if (e.data !== first) location.reload();
  };
})();
</script>
"""

_version = "0"
_changed = threading.Condition()


def _snapshot() -> tuple:
    return tuple(sorted((str(p), p.stat().st_mtime_ns) for p in ROOT.rglob("*") if p.is_file()))


def _watch() -> None:
    global _version
    last = _snapshot()
    while True:
        time.sleep(0.3)
        try:
            now = _snapshot()
        except OSError:  # 文件正在被写入或删除，下一轮再看
            continue
        if now != last:
            time.sleep(0.2)  # 等编辑器把一次保存的多个步骤写完
            last = _snapshot()
            with _changed:
                _version = str(time.time_ns())
                _changed.notify_all()


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format, *args) -> None:
        pass  # 用 pythonw 后台运行时没有控制台，不输出日志

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/__reload":
            return self._stream_reload()
        if path in ("/api/memory", "/api/docs"):
            return self._send_json(api.memory_snapshot if path == "/api/memory" else api.docs_snapshot)
        if path == "/":
            path = "/overview.html"
        if path.endswith(".html"):
            return self._send_html(path)
        super().do_GET()

    def _send_html(self, path: str) -> None:
        file = (ROOT / path.lstrip("/")).resolve()
        if ROOT not in file.parents or not file.is_file():
            return self.send_error(404)
        body = file.read_bytes()
        if b"</body>" in body:
            body = body.replace(b"</body>", RELOAD_SNIPPET + b"</body>", 1)
        else:
            body += RELOAD_SNIPPET
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, make) -> None:
        try:
            body = json.dumps(make(), ensure_ascii=False, default=str).encode()
            code = 200
        except Exception as e:  # 接口出错不影响静态页面和自动刷新
            body = json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False).encode()
            code = 500
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _stream_reload(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        try:
            sent = None
            while True:
                if _version != sent:
                    sent = _version
                    self.wfile.write(f"data: {sent}\n\n".encode())
                else:
                    self.wfile.write(b": keepalive\n\n")  # 让断开的连接尽快暴露
                self.wfile.flush()
                with _changed:
                    _changed.wait(timeout=15)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass


if __name__ == "__main__":
    _version = str(time.time_ns())
    threading.Thread(target=_watch, daemon=True).start()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    server.daemon_threads = True
    server.serve_forever()
