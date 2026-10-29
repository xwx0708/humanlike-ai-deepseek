"""回应模块：每轮组织的内容（D29、D30）、关键词识别的要求（D29、D25、D26）。不调用模型。"""

import pytest

from contracts.memory import EmotionScore, Segment
from llm import FakeModel
from memory.lifecycle import context_used
from memory.scanner import Scanner, Surfaced
from memory.store import MemoryStore
from memory.subconscious import consolidate
from response import requests
from response.responder import PENDING_KEY, PERSONA_REMINDER, Responder
from runtime.lifecycle import FAMILIAR, RUNNING

MIN = 60.0
PERSONA = "你叫小满，说话直接，有点懒。"


@pytest.fixture
def store():
    s = MemoryStore(":memory:")
    yield s
    s.close()


def responder(store, phase=RUNNING):
    return Responder(store, Scanner(store), lambda: PERSONA, phase=lambda: phase)


def old_thread(store, title="他妈妈住院", ts=0.0):
    """一件早先聊过、已经整理成片段并挂上事件线的事。"""
    c = store.new_conversation(ts)
    t = store.add_turn(c.id, "user", "我妈住院了", ts)
    seg = store.add_segment(
        Segment(None, c.id, title, ts, ts, [t.id], [], EmotionScore(0.6, -0.6, 0.6), last_recall_ts=ts)
    )
    tid = store.new_thread(title, title, ts).id
    store.attach(tid, seg.id, 1.0, ts)
    return seg, tid


# ---------- 熟悉期：上下文就是上下文 ----------


def test_familiar_phase_sends_whole_history(store):
    r = responder(store, FAMILIAR)
    r.on_user_message("你好", 0)
    r.on_reply("嗯，你好", 10)
    p = r.on_user_message("今天好累", 20)
    assert p.contexts == [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "嗯，你好"},
        {"role": "system", "content": PERSONA_REMINDER},  # 人设提醒在旧记录之后，免得口气被带偏
    ]
    assert p.prompt == "今天好累" and p.system == PERSONA


def test_reply_reports_real_context_usage(store):
    """D24：接入后改用思考模型实际回报的用量。"""
    r = responder(store, FAMILIAR)
    r.on_user_message("你好", 0)
    r.on_reply("嗯", 10, prompt_tokens=50_000)
    assert context_used(store) == 50_000


# ---------- 运行期：上下文就是记忆系统 ----------


def test_running_phase_has_no_chat_history_only_memory(store):
    r = responder(store)
    r.on_user_message("今天加班到十点", 0)
    r.on_reply("辛苦了", 10)
    p = r.on_user_message("老板还骂我", 20)
    assert p.contexts == []  # D30：不另带聊天记录
    assert p.prompt == "老板还骂我"
    assert p.system.startswith(PERSONA)
    assert "一直记得的事" in p.system  # 常驻栏
    assert "对方：今天加班到十点" in p.system and "我：辛苦了" in p.system  # 新鲜记忆
    assert "对方：老板还骂我" not in p.system  # 这一条本身单独给


def test_surfaced_memories_shown_but_not_duplicated(store):
    seg, _ = old_thread(store)
    store.add_busy_span(MIN, 5 * 3600)  # 之后做了 5 小时任务，记忆时间照常流动：那句话不再新鲜
    r = responder(store)
    r.on_user_message("在吗", 86400)
    now_turn = store.all_turns()[-1]
    fresh_seg = store.add_segment(Segment(None, now_turn.conversation_id, "他问在不在", 0, 0, [now_turn.id]))
    r.scanner.buffer = [
        Surfaced(seg.id, seg.title, 0.4, -0.6, 0),
        Surfaced(fresh_seg.id, "他问在不在", 1.0, 0, 0),
    ]
    p = r.on_user_message("周末去看看她", 86400 + MIN)
    assert "我妈住院了" not in p.system
    assert "（清晰度 0.4）他妈妈住院" in p.system
    assert "他问在不在" not in p.system  # 消息全在新鲜记忆里，不再浮现一次


def test_after_a_long_break_last_words_are_still_fresh(store):
    """不聊天时几乎不遗忘（D21）：十天后回来，上次最后说的话仍是新鲜记忆。"""
    r = responder(store)
    r.on_user_message("我明天面试", 0)
    assert "对方：我明天面试" in r.on_user_message("我回来了", 10 * 86400).system


# ---------- 关键词识别的要求 ----------


def test_detect_keywords():
    assert requests.detect("这个你记住啊") == requests.REMEMBER
    assert requests.detect("这件事不用记住") == requests.FORGET  # 否定说法先认成忘掉
    assert requests.detect("帮我翻翻档案，去年说过什么") == requests.ARCHIVE
    assert requests.detect("今天天气不错") is None
    assert requests.detect_choice("那就存进档案吧") == "存进档案"
    assert requests.detect_choice("彻底删除") == "彻底删除"
    assert requests.detect_choice("算了不用了") == "取消"


def test_remember_keyword_marks_segment_on_consolidation(store):
    r = responder(store)
    p = r.on_user_message("我对芒果过敏，你记住", 0)
    assert "记下了" in p.notes[0]
    seg = {
        "segments": [
            {
                "messages": [1],
                "title": "他对芒果过敏",
                "people": [],
                "arousal": 0.3,
                "valence": -0.2,
                "importance": 0.8,
                "emotion_reason": "",
                "remember_requested": False,
            }
        ]
    }
    new = {
        "thread_ids": [],
        "new_thread": True,
        "new_thread_title": "过敏",
        "related": [],
        "new_thread_ongoing": False,
        "confidence": 1,
        "reason": "",
    }
    consolidate(store, FakeModel([seg, new]), store.last_conversation().id, 30 * MIN, sweep_forgotten=False)
    assert store.segments()[0].remember_requested  # 切段模型没认出，关键词认出了，取并集


def test_forget_asks_first_then_follows_users_choice(store):
    """D25：先问用户选彻底删除还是存进档案，选了才执行。"""
    seg, _ = old_thread(store)
    r = responder(store)
    p = r.on_user_message("妈妈住院那件事你忘掉吧", 86400)
    assert "「他妈妈住院」" in p.notes[0] and "彻底删除" in p.notes[0]
    assert store.get_segment(seg.id).state == "active"  # 还没选，什么都不做
    p = r.on_user_message("存进档案吧", 86400 + MIN)
    assert "已按对方的选择处理" in p.notes[0]
    assert store.get_segment(seg.id).state == "archived"


def test_forget_pending_is_dropped_if_user_moves_on(store):
    seg, _ = old_thread(store)
    r = responder(store)
    r.on_user_message("妈妈住院的事忘掉吧", 86400)
    r.on_user_message("对了今天吃什么", 86400 + MIN)
    assert store.get_meta(PENDING_KEY) == ""
    r.on_user_message("彻底删除", 86400 + 2 * MIN)  # 已经聊别的了，这句不再当作选择
    assert store.get_segment(seg.id).state == "active"


def test_archive_only_searched_when_asked(store):
    seg, _ = old_thread(store)
    seg.state = "archived"
    store.save_segment_state(seg)
    r = responder(store)
    assert r.on_user_message("我妈之前住院是怎么回事", 86400).notes == []  # 没要求，不翻（D26）
    p = r.on_user_message("帮我翻翻档案，我妈住院那次", 86400 + MIN)
    assert "他妈妈住院" in p.notes[0] and "我妈住院了" in p.notes[0]


def test_own_actions_are_remembered(store):
    """D35：AI 自己做的事（发说说等）作为它说过的一句话记进记忆，之后在新鲜记忆里能看到。"""
    r = responder(store)
    r.on_user_message("今天好无聊", 0)
    r.on_action("我在 QQ 空间发了一条说说（好友可见）：摸鱼中", 30)
    p = r.on_user_message("你刚才干嘛了", 60)
    assert "我：（我在 QQ 空间发了一条说说（好友可见）：摸鱼中）" in p.system


def test_persona_reminder_in_running_phase(store):
    r = responder(store)
    assert r.on_user_message("在吗", 0).system.endswith(PERSONA_REMINDER)


def test_evoked_old_fact_surfaces_once(store):
    """D16：改口时旧记录被唤起，下一轮出现在浮现区，只出现一次。"""
    old = store.add_profile_fact(None, "饮食", "他不吃香菜", 0)
    new = store.add_profile_fact(None, "饮食", "他现在爱吃香菜", 100)
    store.add_evocation(old, new, 100)
    r = responder(store)
    p = r.on_user_message("今晚吃火锅", 200)
    assert "你想起他以前说过：他不吃香菜（和他后来说的「他现在爱吃香菜」不一样）" in p.system
    assert "他不吃香菜（" not in r.on_user_message("要不要加香菜", 260).system
