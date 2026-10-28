"""每轮怎么组织交给思考模型的内容（D29、D30），以及把对话交给记忆。

- 熟悉期（记忆系统还没启动）：上下文就是上下文。人设 + 全部聊天记录（D24；用户说明"启动之前，上下文就是上下文"）。
- 运行期：上下文就是记忆系统（D29 第 7 条，用户原话"现在我则需要 AI 的上下文变成我们搭建的记忆系统"）。
  不另带聊天记录（D30），只放：人设、常驻栏、浮现区、新鲜记忆（刚说过的话）。
- 浮现的记忆说不说由思考模型按当下语境判断（D29），这里只负责给它看。
- 用户对记忆的要求用关键词识别（D29），处理结果作为"此刻要注意的事"告诉思考模型，由它用自己的话说。

模型调用不在这里：插件拿 Prompt 去调 AstrBot 里配置的思考模型（D29），再把回复和用量交回 on_reply。
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field

from memory.archive import forget_thread, search_archive
from memory.boundaries import ingest_turn
from memory.fresh import drop_fresh, fresh_block, fresh_memories
from memory.lifecycle import report_context_used
from memory.resident import resident_block
from memory.scanner import Scanner, focus_of, search_memory
from memory.store import MemoryStore
from response import requests
from runtime.lifecycle import FAMILIAR, Lifecycle

SURFACED_HEADER = (
    "以下是你此刻自己想起来的事。说不说、怎么说，由你根据当下的语境判断，不必每条都提；"
    "清晰度低的只是模糊印象，可以说记不太清。"
)
PENDING_KEY = "pending_forget"  # 要求忘掉、等用户选方式时，记下是哪件事


@dataclass
class Prompt:
    system: str  # 系统提示词
    contexts: list[dict] = field(default_factory=list)  # 之前的对话（只在熟悉期有）
    prompt: str = ""  # 这一条用户消息
    notes: list[str] = field(
        default_factory=list
    )  # 此刻要注意的事（已经放进 system，单独留一份便于测试和日志）


class Responder:
    def __init__(
        self,
        store: MemoryStore,
        scanner: Scanner,
        persona: Callable[[], str],
        phase: Callable[[], str] = lambda: Lifecycle().phase,
    ):
        self.store = store
        self.scanner = scanner
        self.persona = persona
        self.phase = phase  # 生命周期由后台任务推进，每轮重新读

    def on_user_message(self, text: str, ts: float) -> Prompt:
        turn = ingest_turn(self.store, "user", text, ts)
        notes = self._handle_requests(text, turn.id, ts)
        persona = self.persona()
        if self.phase() == FAMILIAR:
            history = [t for t in self.store.all_turns() if t.id != turn.id]
            contexts = [{"role": t.role, "content": t.content} for t in history]
            return Prompt(_join(persona, _notes_block(notes)), contexts, text, notes)
        # 运行期：话题和专注程度交给扫描器，下一次扫描按新的话题充电、按专注程度定温度（D26）
        self.scanner.set_topic(text)
        self.scanner.set_focus(focus_of(self.store, turn.conversation_id))
        fresh = [m for m in fresh_memories(self.store, ts) if m.turn.id != turn.id]  # 这一条本身单独给
        surfaced = drop_fresh(self.store, self.scanner.buffer, fresh)
        system = _join(
            persona,
            resident_block(self.store),
            _surfaced_block(surfaced),
            fresh_block(fresh),
            _notes_block(notes),
        )
        return Prompt(system, [], text, notes)

    def on_reply(self, text: str, ts: float, prompt_tokens: int | None = None) -> None:
        """思考模型回复以后：存进记忆；回报实际用掉的上下文（熟悉期据此判断何时启动，D24）。"""
        ingest_turn(self.store, "assistant", text, ts)
        if prompt_tokens:
            report_context_used(self.store, prompt_tokens)

    # ---------- 用户对记忆的要求（D29：关键词识别） ----------

    def _handle_requests(self, text: str, turn_id: int, ts: float) -> list[str]:
        pending = self.store.get_meta(PENDING_KEY)
        self.store.set_meta(PENDING_KEY, "")  # 只等紧接着的一条：用户没选、聊别的了，就算放下了
        if pending:
            choice = requests.detect_choice(text)
            target = json.loads(pending)
            if choice == "取消":
                return [f"对方决定不忘掉「{target['title']}」了，照常记着。"]
            if choice:  # D25：由用户选，选了才执行
                said = forget_thread(self.store, target["thread"], choice, ts)
                return [f"已按对方的选择处理「{target['title']}」：{said}用你自己的话告诉对方。"]
        kind = requests.detect(text)
        if kind == requests.REMEMBER:
            self.store.mark_remember(turn_id)  # 整理时含这条消息的片段算用户要求记住（D17 免竞争）
            return ["对方要你记住这件事，已经记下了。自然地答应就好。"]
        if kind == requests.FORGET:
            return self._ask_forget(text, ts)
        if kind == requests.ARCHIVE:  # D26：只有用户要求才翻档案
            return [_archive_note(search_archive(self.store, requests.strip_keywords(text) or text))]
        return []

    def _ask_forget(self, text: str, ts: float) -> list[str]:
        """暂定：用去掉关键词后的话在记忆里搜，取最相关的片段所在的事件线，请思考模型跟用户确认。"""
        for hit in search_memory(self.store, requests.strip_keywords(text) or text, ts, limit=3):
            tids = self.store.threads_of_segment(hit.segment_id)
            if tids:
                title = self.store.thread_card(tids[0])["title"]
                self.store.set_meta(
                    PENDING_KEY, json.dumps({"thread": tids[0], "title": title}, ensure_ascii=False)
                )
                ask = (
                    f"对方要你忘掉一件事，你理解为「{title}」。先跟对方确认是不是这件，并问清楚要"
                    "「彻底删除」（再也找不回来）还是「存进档案」（以后要找还能找到）。"
                    "对方明确选择之前，不要说已经忘了。"
                )
                return [ask]
        return ["对方要你忘掉一件事，但你没想起是哪件。问清楚是哪件事。"]


def _archive_note(hits: list) -> str:
    if not hits:
        return "对方要你翻以前的记录，档案里没找到相关的。如实告诉对方。"
    lines = ["对方要你翻以前的记录，档案里找到这些原文（按相关程度排列）："]
    for h in hits:
        lines.append(f"- {h.title}：{h.excerpt}")
    return "\n".join(lines)


def _surfaced_block(items: list) -> str:
    if not items:
        return ""
    lines = [SURFACED_HEADER]
    for s in items:
        lines.append(f"- （清晰度 {s.clarity:.1f}）{s.text}")
    return "\n".join(lines)


def _notes_block(notes: list[str]) -> str:
    return "此刻要注意的事：\n" + "\n".join(f"- {n}" for n in notes) if notes else ""


def _join(*parts: str) -> str:
    return "\n\n".join(p for p in parts if p)
