"""设置的 MCP 服务：让用户自己的智能体（Claude、Cherry Studio 等）读懂设置、给用户解释，并在用户同意后修改。

用户 2026-11-08："我希望用户在使用它的时候可以使用智能体理解它，并修改它。并给用户做解释。"选了"给外部智能体的接口"。

补充（做法）：
- 走 MCP 的 stdio 传输（每行一条 JSON-RPC 消息），只用标准库，和设置页一样不加依赖。
  用法：在 src 目录下 `python -m settings_page.mcp_server`；客户端配置见 03-回应/部署.md。
- 改动分两步：propose_* 只校验、给出改前改后的对照，不写文件；用户同意后 apply_change 才写。
  校验和设置页用同一套（settings_page.server.clean）：类型转换、范围裁剪、不认识的项丢掉。
- 每次写入都记进数据文件夹的 settings_history.json（改前、改后、理由），可以 propose_undo 撤销最近一次。
- 提议之后设置文件被别处改过（例如有人在设置页点了保存），apply 会拒绝，要重新提议。
- 设置改完要重启 AstrBot 才生效（docker compose restart astrbot）；人设改完立即生效。
"""

import copy
import hashlib
import json
import sys
import time
import uuid
from typing import Any

from runtime import config
from settings_page.schema import schema
from settings_page.server import clean, read_persona, write_persona

PROTOCOL = "2025-06-18"
INSTRUCTIONS = """这是"类人 AI"项目的设置。用户想了解或调整它的行为时用这些工具。
- 先用 find_settings / list_settings 找到相关的项，用用户听得懂的话解释它是什么、现在是多少、改大改小会怎样。
- 要改时先 propose_change（或 propose_persona / propose_undo），把返回的改前改后对照和提醒原样告诉用户，
  得到用户明确同意后再 apply_change。不要替用户做决定，也不要一次改很多项。
- 设置改完要重启 AstrBot 才生效（在项目目录运行 docker compose restart astrbot），告诉用户这一点；人设改完立即生效。
- 不确定的就照实说不确定，不要编参数的作用。"""

TOOLS = [
    {
        "name": "list_settings",
        "description": "列出设置。不给 section 时列出全部分区（标题、说明、哪些项改过）；给了 section 时列出这个分区每一项的"
        "名称、说明、当前值、默认值、范围和单位。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "section": {"type": "string", "description": "分区 id，例如 forgetting；不给就列全部分区"}
            },
        },
    },
    {
        "name": "find_settings",
        "description": '按用户的说法找相关的设置项（在名称和说明里搜），例如"忘事""头像""说说"。',
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "关键词，可以用空格分开多个"}},
            "required": ["query"],
        },
    },
    {
        "name": "propose_change",
        "description": "提议修改设置：只校验、给出改前改后的对照，不写文件。用户同意后用 apply_change 应用。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "changes": {
                    "type": "array",
                    "description": "要改的数值项",
                    "items": {
                        "type": "object",
                        "properties": {
                            "section": {"type": "string"},
                            "key": {"type": "string"},
                            "value": {"description": "新值；填 null 表示恢复默认"},
                        },
                        "required": ["section", "key", "value"],
                    },
                },
                "features": {
                    "type": "object",
                    "description": "功能开关：{开关名: true/false}",
                    "additionalProperties": {"type": "boolean"},
                },
                "reason": {"type": "string", "description": "为什么改（用户的原话或目的），记进修改历史"},
            },
        },
    },
    {
        "name": "get_persona",
        "description": "读人设（它是谁、怎么说话）。",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "propose_persona",
        "description": "提议把人设整段换成新的文字：只给出对照，不写文件。用户同意后用 apply_change 应用。",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}, "reason": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "apply_change",
        "description": "应用一个提议（propose_change / propose_persona / propose_undo 返回的编号）。只在用户明确同意后调用。",
        "inputSchema": {
            "type": "object",
            "properties": {"proposal_id": {"type": "string"}},
            "required": ["proposal_id"],
        },
    },
    {
        "name": "list_history",
        "description": "最近几次修改（时间、理由、改了什么）。",
        "inputSchema": {"type": "object", "properties": {"limit": {"type": "integer"}}},
    },
    {
        "name": "propose_undo",
        "description": "提议撤销最近一次修改（设置或人设），给出对照；用户同意后用 apply_change 应用。",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


class Settings:
    """工具的实现。提议放在内存里，服务重启就作废（重新提议即可）。"""

    def __init__(self):
        self.proposals: dict[str, dict[str, Any]] = {}

    # ---------- 读 ----------

    def list_settings(self, section: str = "") -> str:
        spec, current = schema(), config.load()
        if not section:
            lines = []
            for g in spec["groups"]:
                lines.append(f"【{g['title']}】")
                for sid in g["ids"]:
                    sec = _section(spec, sid)
                    if sec:
                        changed = list((current.get(sid) or {}).keys())
                        mark = f"（改过：{'、'.join(changed)}）" if changed else ""
                        lines.append(f"- {sid}：{sec['title']}{mark}。{sec.get('help', '')}")
            off = [f["key"] for f in spec["features"] if not config.enabled(f["key"])]
            lines.append(
                f"功能开关：{len(spec['features'])} 个，关掉的：{'、'.join(off) or '无'}（list_settings section=features 看全部）"
            )
            return "\n".join(lines)
        if section == "features":
            return "\n".join(
                f"- {f['key']}：{f['label']}，{'开' if config.enabled(f['key']) else '关'}。{f.get('help', '')}"
                for f in spec["features"]
            )
        sec = _section(spec, section)
        if sec is None:
            return f"没有分区 {section}。可选：{'、'.join(s['id'] for s in spec['sections'])}、features"
        mine = current.get(section) or {}
        out = [f"{sec['title']}（{section}）：{sec.get('help', '')}"]
        out += [_describe(f, mine.get(f["key"], f["default"])) for f in sec["fields"]]
        return "\n".join(out)

    def find_settings(self, query: str) -> str:
        words = [w for w in query.split() if w]
        spec, current = schema(), config.load()
        hits = []
        for sec in spec["sections"]:
            for f in sec["fields"]:
                text = f"{sec['title']} {f['label']} {f.get('help', '')} {f['key']}"
                if any(w in text for w in words):
                    now = (current.get(sec["id"]) or {}).get(f["key"], f["default"])
                    hits.append(f"[{sec['id']}.{f['key']}] {sec['title']} · " + _describe(f, now)[2:])
        for f in spec["features"]:
            if any(w in f"{f['label']} {f.get('help', '')} {f['key']}" for w in words):
                hits.append(
                    f"[开关 {f['key']}] {f['label']}，{'开' if config.enabled(f['key']) else '关'}。{f.get('help', '')}"
                )
        return (
            "\n".join(hits[:20])
            if hits
            else f"没找到和「{query}」有关的设置。可以用 list_settings 看全部分区。"
        )

    def get_persona(self) -> str:
        p = read_persona()
        return f"人设文件：{p['path']}\n" + (p["text"] or "（还没写人设）")

    def list_history(self, limit: int = 5) -> str:
        hist = _history()[-limit:]
        if not hist:
            return "还没有通过智能体改过设置。"
        return "\n".join(
            f"- {h['time']}：{h['summary']}（理由：{h.get('reason') or '无'}）" for h in reversed(hist)
        )

    # ---------- 提议 ----------

    def propose_change(
        self, changes: list | None = None, features: dict | None = None, reason: str = ""
    ) -> str:
        spec, before = schema(), config.load()
        raw = copy.deepcopy(before)
        problems = []
        for c in changes or []:
            sec = _section(spec, c.get("section", ""))
            field = next((f for f in sec["fields"] if f["key"] == c.get("key")), None) if sec else None
            if field is None:
                problems.append(
                    f"没有这一项：{c.get('section')}.{c.get('key')}（用 find_settings 找正确的名字）"
                )
                continue
            sid = sec["id"]
            raw.setdefault(sid, {})
            if c.get("value") is None:
                raw[sid].pop(field["key"], None)
            else:
                raw[sid][field["key"]] = c["value"]
        for key, on in (features or {}).items():
            if not any(f["key"] == key for f in spec["features"]):
                problems.append(f"没有这个开关：{key}")
                continue
            raw.setdefault("features", {})[key] = bool(on)
        after = clean(raw)
        diff = _diff(spec, before, after)
        for c in changes or []:  # 被范围裁剪、或不合法被丢掉的，要告诉用户
            sec, key = c.get("section"), c.get("key")
            field = next(
                (f for s in spec["sections"] if s["id"] == sec for f in s["fields"] if f["key"] == key), None
            )
            if field and c.get("value") is not None:
                got = (after.get(sec) or {}).get(key, field["default"])
                if str(got) != str(c["value"]):
                    problems.append(f"{field['label']} 要求 {c['value']}，按允许范围实际会是 {got}")
                for v, warn in (field.get("warnings") or {}).items():
                    if str(v) == str(got):
                        problems.append(f"{field['label']}：{warn}")
        if not diff:
            return "\n".join(problems + ["这样改和现在一样，没有要应用的改动。"])
        pid = self._keep("settings", before, after, reason, diff)
        return _proposal_text(pid, diff, problems, "设置改完要重启 AstrBot 才生效。")

    def propose_persona(self, text: str, reason: str = "") -> str:
        old = read_persona()["text"]
        if text.strip() == old.strip():
            return "新人设和现在的一样，没有要应用的改动。"
        diff = [
            f"人设：{len(old)} 字 → {len(text)} 字",
            f"新人设开头：{text[:120]}{'……' if len(text) > 120 else ''}",
        ]
        pid = self._keep("persona", old, text, reason, diff)
        return _proposal_text(pid, diff, [], "人设改完立即生效。")

    def propose_undo(self) -> str:
        hist = _history()
        if not hist:
            return "没有可以撤销的修改。"
        last = hist[-1]
        current = config.load() if last["kind"] == "settings" else read_persona()["text"]
        diff = [f"撤销 {last['time']} 的修改：{last['summary']}"]
        pid = self._keep(
            last["kind"], current, last["before"], f"撤销 {last['time']} 的修改", diff, undo=True
        )
        return _proposal_text(pid, diff, [], "设置要重启 AstrBot 才生效；人设立即生效。")

    def apply_change(self, proposal_id: str) -> str:
        p = self.proposals.pop(proposal_id, None)
        if p is None:
            return "没有这个提议（可能已经应用过，或服务重启过）。请重新提议。"
        now = config.load() if p["kind"] == "settings" else read_persona()["text"]
        if _digest(now) != p["base"]:
            return "提议之后设置又被改过了（可能有人在设置页保存过），为了不覆盖，请重新提议。"
        if p["kind"] == "settings":
            config.save(p["after"])
        else:
            write_persona(p["after"])
        hist = _history()
        if p["undo"] and hist:
            hist.pop()  # 撤销的那次不再留着，免得再撤销一次又改回去
        else:
            hist.append(
                {
                    "time": time.strftime("%Y-%m-%d %H:%M"),
                    "kind": p["kind"],
                    "before": p["before"],
                    "reason": p["reason"],
                    "summary": "；".join(p["diff"])[:300],
                }
            )
        _save_history(hist)
        tail = (
            "已生效。"
            if p["kind"] == "persona"
            else "要重启 AstrBot 才生效：在项目目录运行 docker compose restart astrbot。"
        )
        return "已应用：" + "；".join(p["diff"]) + "。" + tail

    def _keep(
        self, kind: str, before: Any, after: Any, reason: str, diff: list[str], undo: bool = False
    ) -> str:
        pid = uuid.uuid4().hex[:8]
        self.proposals[pid] = {
            "kind": kind,
            "before": before,
            "after": after,
            "base": _digest(before),
            "reason": reason,
            "diff": diff,
            "undo": undo,
        }
        return pid


def _section(spec: dict, sid: str) -> dict | None:
    return next((s for s in spec["sections"] if s["id"] == sid), None)


def _describe(f: dict, now: Any) -> str:
    rng = f" 范围 {f['min']}～{f['max']}" if f.get("min") is not None and f.get("max") is not None else ""
    opts = f"，可选 {f['options']}" if f.get("options") else ""
    unit = f" {f['unit']}" if f.get("unit") else ""
    changed = "（改过）" if now != f["default"] else ""
    return f"- {f['key']}：{f['label']}，现在 {now}{unit}{changed}，默认 {f['default']}{rng}{opts}。{f.get('help', '')}"


def _diff(spec: dict, before: dict, after: dict) -> list[str]:
    out = []
    for sec in spec["sections"]:
        b, a = before.get(sec["id"]) or {}, after.get(sec["id"]) or {}
        for f in sec["fields"]:
            old, new = b.get(f["key"], f["default"]), a.get(f["key"], f["default"])
            if old != new:
                unit = f" {f['unit']}" if f.get("unit") else ""
                out.append(f"{sec['title']} · {f['label']}：{old}{unit} → {new}{unit}")
    bf, af = before.get("features") or {}, after.get("features") or {}
    for f in spec["features"]:
        old, new = bf.get(f["key"], True), af.get(f["key"], True)
        if old != new:
            out.append(f"开关 · {f['label']}：{'开' if old else '关'} → {'开' if new else '关'}")
    return out


def _proposal_text(pid: str, diff: list[str], notes: list[str], effect: str) -> str:
    lines = [f"提议编号 {pid}（还没应用）：", *[f"- {d}" for d in diff]]
    lines += [f"注意：{n}" for n in notes]
    lines.append(f"{effect}请把这些告诉用户，用户同意后再调用 apply_change。")
    return "\n".join(lines)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _history_path():
    return config.data_dir() / "settings_history.json"


def _history() -> list[dict]:
    p = _history_path()
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else []
    except (OSError, ValueError):
        return []


def _save_history(hist: list[dict]) -> None:
    p = _history_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(hist[-50:], ensure_ascii=False, indent=1), encoding="utf-8")


# ---------- MCP（stdio，JSON-RPC 2.0） ----------


def handle(tools: Settings, msg: dict) -> dict | None:
    """处理一条消息，返回回复；通知（没有 id）不回复。"""
    mid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
    if mid is None:
        return None
    if method == "initialize":
        result = {
            "protocolVersion": params.get("protocolVersion") or PROTOCOL,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "humanlike-settings", "version": "0.1.0"},
            "instructions": INSTRUCTIONS,
        }
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        name, args = params.get("name"), params.get("arguments") or {}
        fn = getattr(tools, name, None) if any(t["name"] == name for t in TOOLS) else None
        if fn is None:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": f"没有工具 {name}"}}
        try:
            text, err = fn(**args), False
        except TypeError as e:
            text, err = f"参数不对：{e}", True
        except Exception as e:  # noqa: BLE001  工具出错告诉智能体，服务不能停
            text, err = f"出错了：{type(e).__name__}: {e}", True
        result = {"content": [{"type": "text", "text": text}], "isError": err}
    else:
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"不支持 {method}"}}
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def main() -> None:
    config.load_modules()  # 让各模块登记自己的参数（schema 里才有全部项）
    tools = Settings()
    # MCP 规定 UTF-8；Windows 控制台默认编码不是，所以按文件描述符重新打开
    with (
        open(sys.stdin.fileno(), encoding="utf-8", closefd=False) as stdin,
        open(sys.stdout.fileno(), "w", encoding="utf-8", closefd=False) as stdout,
    ):
        for line in stdin:
            if not line.strip():
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                reply = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "不是合法的 JSON"},
                }
            else:
                reply = handle(tools, msg) if isinstance(msg, dict) else None
            if reply is not None:
                stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
                stdout.flush()


if __name__ == "__main__":
    main()
