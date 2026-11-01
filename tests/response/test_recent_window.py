"""接口成立的证据（改进一）：把记忆换成"只留最近 N 条"的对照组，回应模块照常工作、一行不改。"""

from memory import RecentWindowMemory
from response.responder import PERSONA_REMINDER, Responder


def test_recent_window_sends_last_n_turns():
    r = Responder(RecentWindowMemory(window=2), lambda: "人设")
    r.on_user_message("一", 0)
    r.on_reply("二", 1)
    r.on_user_message("三", 2)
    r.on_reply("四", 3)
    p = r.on_user_message("五", 4)
    assert p.contexts == [
        {"role": "user", "content": "三"},
        {"role": "assistant", "content": "四"},
        {"role": "system", "content": PERSONA_REMINDER},
    ]
    assert p.prompt == "五" and p.system == "人设"


def test_recent_window_has_no_memory_requests():
    """没有事件线、没有档案：要求忘掉时只能问是哪件，翻档案什么都找不到。"""
    r = Responder(RecentWindowMemory(), lambda: "")
    assert "没想起是哪件" in r.on_user_message("把昨天的事忘掉", 0).notes[0]
    assert "没找到" in r.on_user_message("翻翻档案，昨天说了什么", 1).notes[0]
    r.on_action("发了一条说说", 2)
    assert r.on_user_message("在吗", 3).contexts[-1]["content"] == "（发了一条说说）"
