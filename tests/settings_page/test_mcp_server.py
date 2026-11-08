"""设置的 MCP 服务（用户 2026-11-08）：智能体能读懂、解释设置；改动先提议、用户同意才应用，可撤销。"""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from runtime import config
from settings_page.mcp_server import Settings, handle

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def tools(tmp_path, monkeypatch):
    monkeypatch.setenv("HUMANLIKE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("HUMANLIKE_DATA_DIR", str(tmp_path / "data"))
    return Settings()


def pid(text):
    return re.search(r"提议编号 (\w+)", text).group(1)


def test_lists_and_finds_settings_in_plain_words(tools):
    overview = tools.list_settings()
    assert "forgetting：遗忘曲线" in overview and "功能开关" in overview
    detail = tools.list_settings("forgetting")
    assert "base_half_life：基础半衰期，现在 7.0 天" in detail and "范围 0.5～60" in detail
    assert "[forgetting.base_half_life]" in tools.find_settings("半衰期")
    assert "qq_set_avatar" in tools.find_settings("头像")
    assert "没找到" in tools.find_settings("火星")


def test_propose_does_not_write_until_applied(tools):
    text = tools.propose_change(
        [{"section": "forgetting", "key": "base_half_life", "value": 14}], reason="老忘事"
    )
    assert "基础半衰期：7.0 天 → 14.0 天" in text and "还没应用" in text
    assert config.load() == {}  # 只是提议
    assert "重启 AstrBot" in tools.apply_change(pid(text))
    assert config.load() == {"forgetting": {"base_half_life": 14.0}}
    assert "没有这个提议" in tools.apply_change(pid(text))  # 用过就作废
    assert "老忘事" in tools.list_history()


def test_out_of_range_is_clipped_and_reported(tools):
    text = tools.propose_change([{"section": "forgetting", "key": "base_half_life", "value": 999}])
    assert "实际会是 60" in text
    assert "没有这一项" in tools.propose_change([{"section": "forgetting", "key": "nope", "value": 1}])


def test_feature_switch_and_undo(tools):
    tools.apply_change(pid(tools.propose_change(features={"qq_set_avatar": False})))
    assert not config.enabled("qq_set_avatar")
    undo = tools.propose_undo()
    assert "撤销" in undo and config.load() == {"features": {"qq_set_avatar": False}}
    tools.apply_change(pid(undo))
    assert config.enabled("qq_set_avatar") and config.load() == {}
    assert "没有可以撤销" in tools.propose_undo()


def test_refuses_if_settings_changed_after_proposal(tools):
    text = tools.propose_change([{"section": "forgetting", "key": "base_half_life", "value": 10}])
    config.save({"scanner": {"buffer_size": 6}})  # 有人在设置页保存过
    assert "又被改过" in tools.apply_change(pid(text))
    assert config.load() == {"scanner": {"buffer_size": 6}}


def test_persona_proposal(tools):
    text = tools.propose_persona("你叫大肥鱼，说话慢。", reason="换个性格")
    assert "人设：0 字" in text and "（还没写人设）" in tools.get_persona()
    assert "已生效" in tools.apply_change(pid(text))
    assert "大肥鱼" in tools.get_persona()


def test_mcp_protocol_over_stdio(tmp_path):
    """真的起一个进程，按 MCP 的方式说话：初始化、列工具、调用工具。"""
    msgs = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "find_settings", "arguments": {"query": "半衰期"}},
        },
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "nope", "arguments": {}}},
    ]
    env = {"HUMANLIKE_SETTINGS": str(tmp_path / "s.json"), "HUMANLIKE_DATA_DIR": str(tmp_path), "PYTHONPATH": str(ROOT / "src"),
           "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", "")}  # fmt: skip
    out = subprocess.run(
        [sys.executable, "-m", "settings_page.mcp_server"],
        input="\n".join(json.dumps(m, ensure_ascii=False) for m in msgs) + "\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        timeout=60,
        check=True,
    ).stdout
    replies = {r["id"]: r for r in map(json.loads, out.splitlines())}
    assert set(replies) == {1, 2, 3, 4}  # 通知不回复
    assert "apply_change" in replies[1]["result"]["instructions"]
    assert {t["name"] for t in replies[2]["result"]["tools"]} >= {
        "propose_change",
        "apply_change",
        "propose_undo",
    }
    assert "base_half_life" in replies[3]["result"]["content"][0]["text"]
    assert "error" in replies[4]


def test_notifications_get_no_reply():
    assert handle(Settings(), {"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
