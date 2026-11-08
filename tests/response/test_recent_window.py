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
    assert p.prompt == "五" and p.system.split("\n\n")[0] == "人设"  # 后面是现在的时间


def test_recent_window_has_no_memory_requests():
    """没有事件线、没有档案：要求忘掉时只能问是哪件，翻档案什么都找不到。"""
    r = Responder(RecentWindowMemory(), lambda: "")
    assert "没想起是哪件" in r.on_user_message("把昨天的事忘掉", 0).notes[0]
    assert "没找到" in r.on_user_message("翻翻档案，昨天说了什么", 1).notes[0]
    r.on_action("发了一条说说", 2)
    assert r.on_user_message("在吗", 3).contexts[-1]["content"] == "（发了一条说说）"


def test_recent_window_can_be_cut_by_tokens():
    """评估时和记忆系统的上下文预算对齐（项目总评第三步）：按 token 从最近的往前放。"""
    r = Responder(RecentWindowMemory(window=100, token_budget=30), lambda: "")
    for i in range(10):
        r.on_user_message(f"第{i}句" + "话" * 8, i)
        r.on_reply("嗯", i + 0.5)
    p = r.on_user_message("现在", 20)
    kept = [c["content"] for c in p.contexts]
    assert kept[-1] == "嗯" and len(kept) < 20 and "第0句话话话话话话话话" not in kept
