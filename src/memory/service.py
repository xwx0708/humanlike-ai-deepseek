"""记忆模块对外的门：contracts.memory.MemoryService 的实现（《模块化改进建议》改进一）。

- DefaultMemoryService：概率海。内部持有 MemoryStore 和 Scanner，把回应模块以前直接调的函数收进来，
  逻辑一行不改，只是换了入口。
- RecentWindowMemory：对照组，只保留最近 N 条聊天记录。它是接口成立的证据，也是以后模拟对话测试里
  "普通上下文窗口"的对照。
- background_job：后台整理（每分钟一次）。整理要调本地模型、可能几分钟，在另一个线程跑，
  SQLite 连接不能跨线程，所以每次自己开一个连接、用完关掉；扫描器则长期存在，用收发消息线程的连接。
"""

import json
from collections import deque
from collections.abc import Callable
from pathlib import Path

from contracts.llm import ChatModel
from contracts.memory import ArchiveHit, ChatTurn, ForgetTarget, MemoryContext
from memory.archive import forget_thread, search_archive
from memory.boundaries import ingest_turn
from memory.fresh import drop_fresh, fresh_block, fresh_memories
from memory.lifecycle import register, report_context_used
from memory.profile import take_evocations
from memory.resident import resident_block
from memory.runner import TickReport, tick
from memory.scanner import Scanner, focus_of, search_memory
from memory.store import DEFAULT_PATH, MemoryStore
from runtime.lifecycle import DEFAULT_PATH as LIFECYCLE_PATH
from runtime.lifecycle import FAMILIAR, Lifecycle

SURFACED_HEADER = (
    "以下是你此刻自己想起来的事。说不说、怎么说，由你根据当下的语境判断，不必每条都提；"
    "清晰度低的只是模糊印象，可以说记不太清。"
)
PENDING_KEY = "pending_forget"  # 要求忘掉、等用户选方式时，记下是哪件事


class DefaultMemoryService:
    def __init__(
        self,
        store: MemoryStore | None = None,
        scanner: Scanner | None = None,
        phase: Callable[[], str] = lambda: Lifecycle().phase,
    ):
        self.store = store or MemoryStore()
        self.scanner = scanner or Scanner(self.store)
        self._phase = phase  # 生命周期由后台任务推进，每轮重新读

    def ingest(self, role: str, content: str, ts: float) -> ChatTurn:
        return ingest_turn(self.store, role, content, ts)

    def context_for_turn(self, turn: ChatTurn, ts: float) -> MemoryContext:
        self.scanner.refresh_buffer(ts)
        phase = self._phase()  # 整轮只读一次：后台线程可能正好在这一轮里把阶段切到运行期
        if phase == FAMILIAR:
            history = [t for t in self.store.all_turns(include_forgotten=False) if t.id != turn.id]
            return MemoryContext(phase, history=history)
        # 运行期：话题和专注程度交给扫描器，下一次扫描按新的话题充电、按专注程度定温度（D26）
        self.scanner.set_topic(turn.content)
        self.scanner.set_focus(focus_of(self.store, turn.conversation_id))
        fresh = [m for m in fresh_memories(self.store, ts) if m.turn.id != turn.id]  # 这一条本身单独给
        surfaced = drop_fresh(self.store, self.scanner.buffer, fresh)
        return MemoryContext(
            phase,
            resident=resident_block(self.store),
            surfaced=_surfaced_block(surfaced, take_evocations(self.store)),
            fresh=fresh_block(fresh),
            surfaced_ids=[s.segment_id for s in surfaced],
        )

    def scan(self, ts: float) -> None:
        self.scanner.tick(ts)

    def report_context_used(self, prompt_tokens: int) -> None:
        report_context_used(self.store, prompt_tokens)

    def mark_remember(self, turn_id: int) -> None:
        self.store.mark_remember(turn_id)  # 整理时含这条消息的片段算用户要求记住（D17 免竞争）

    def find_forget_target(self, query: str, ts: float) -> ForgetTarget | None:
        """暂定：用去掉关键词后的话在记忆里搜，取最相关的片段所在的事件线。"""
        for hit in search_memory(self.store, query, ts, limit=3):
            tids = self.store.threads_of_segment(hit.segment_id)
            if tids:
                return ForgetTarget(tids[0], self.store.thread_card(tids[0])["title"])
        return None

    def take_pending_forget(self) -> ForgetTarget | None:
        raw = self.store.get_meta(PENDING_KEY)
        self.store.set_meta(PENDING_KEY, "")  # 先清空再解析：记录坏了也只影响这一条，不会每条都报错
        if not raw:
            return None
        data = json.loads(raw)
        return ForgetTarget(data["thread"], data["title"])

    def set_pending_forget(self, target: ForgetTarget) -> None:
        self.store.set_meta(
            PENDING_KEY, json.dumps({"thread": target.thread_id, "title": target.title}, ensure_ascii=False)
        )

    def forget(self, thread_id: int, mode: str, ts: float) -> str:
        return forget_thread(self.store, thread_id, mode, ts)

    def search_archive(self, query: str) -> list[ArchiveHit]:
        return search_archive(self.store, query)

    def set_meta(self, key: str, value: str) -> None:
        self.store.set_meta(key, value)

    def close(self) -> None:
        self.store.close()


def _surfaced_block(items: list, evoked: list[str] = ()) -> str:
    """浮现区：自己冒头的记忆，加上被唤起的旧事实（D16）。"""
    if not items and not evoked:
        return ""
    lines = [SURFACED_HEADER]
    for s in items:
        lines.append(f"- （清晰度 {s.clarity:.1f}）{s.text}")
    lines += [f"- {e}" for e in evoked]
    return "\n".join(lines)


def background_job(
    model: ChatModel,
    context_limit: int,
    db_path: str | Path = DEFAULT_PATH,
    lifecycle_path: Path | None = LIFECYCLE_PATH,
) -> Callable[[float], TickReport]:
    """记忆模块每分钟要做的事：熟悉期检查是否该启动；运行期停顿后整理、对话中途分批整理、夜间合并。"""

    def run(now_ts: float) -> TickReport:
        store = MemoryStore(db_path)
        try:
            life = Lifecycle(path=lifecycle_path)
            register(life, store, model)
            return tick(store, model, now_ts, life, context_limit)
        finally:
            store.close()

    return run


class RecentWindowMemory:
    """对照组：没有记忆系统，只把最近 N 条聊天记录当上下文。什么都不忘、什么都不想起。"""

    def __init__(self, window: int = 50):
        self.window = window
        self.turns: deque[ChatTurn] = deque(maxlen=window + 1)  # 窗口 + 这一条本身，更早的直接丢
        self.count = 0
        self.meta: dict[str, str] = {}
        self._pending: ForgetTarget | None = None

    def ingest(self, role: str, content: str, ts: float) -> ChatTurn:
        self.count += 1
        turn = ChatTurn(self.count, 1, role, content, ts)
        self.turns.append(turn)
        return turn

    def context_for_turn(self, turn: ChatTurn, ts: float) -> MemoryContext:
        history = [t for t in self.turns if t.id != turn.id]
        return MemoryContext(FAMILIAR, history=history[-self.window :])  # 永远是"上下文就是上下文"

    def scan(self, ts: float) -> None:
        pass

    def report_context_used(self, prompt_tokens: int) -> None:
        pass

    def mark_remember(self, turn_id: int) -> None:
        pass

    def find_forget_target(self, query: str, ts: float) -> ForgetTarget | None:
        return None

    def take_pending_forget(self) -> ForgetTarget | None:
        pending, self._pending = self._pending, None
        return pending

    def set_pending_forget(self, target: ForgetTarget) -> None:
        self._pending = target

    def forget(self, thread_id: int, mode: str, ts: float) -> str:
        raise ValueError("最近窗口没有事件线可忘")

    def search_archive(self, query: str) -> list[ArchiveHit]:
        return []

    def set_meta(self, key: str, value: str) -> None:
        self.meta[key] = value

    def close(self) -> None:
        pass
