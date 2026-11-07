"""命令行对话平台（D44）：不经 AstrBot，直接在终端里和它聊。

- 思考模型用本项目 .env 里的 THINKING_PROVIDER / THINKING_MODEL（AstrBot 部署时这两项不起作用，以 AstrBot 为准）。
- 记忆、人设、后台整理和 AstrBot 插件完全一样；没有工具调用、联网搜索和看图（那些是 AstrBot 提供的）。
- 用法：在 src 目录下 `python -m response.cli`，输入 /quit 退出。
"""

import sys
import time
from collections.abc import Callable, Iterable, Sequence

from contracts.llm import ChatModel, Message
from contracts.memory import MemoryService
from contracts.platform import ThinkingBackend
from response.persona import load_persona
from response.responder import Responder


class ModelBackend:
    """contracts.platform.ThinkingBackend 的实现：直接调一个 ChatModel。"""

    def __init__(self, model: ChatModel):
        self.model = model

    def reply(
        self, system: str, contexts: list[dict[str, str]], prompt: str, image_paths: Sequence[str] = ()
    ) -> tuple[str, int | None]:
        messages = [Message("system", system)] if system else []
        messages += [Message(c["role"], c["content"]) for c in contexts]
        messages.append(Message("user", prompt))
        r = self.model.chat(messages)
        return r.text, r.prompt_tokens or None


def run(
    memory: MemoryService,
    backend: ThinkingBackend,
    lines: Iterable[str],
    *,
    persona: Callable[[], str] = load_persona,
    out: Callable[[str], None] = print,
    clock: Callable[[], float] = time.time,
    tick: Callable[[float], object] | None = None,
) -> None:
    """逐行读输入、逐条回复。tick：每条消息前跑一次的后台步骤（扫描、整理），可为空。"""
    responder = Responder(memory, persona)
    for line in lines:
        text = line.strip()
        if not text:
            continue
        if text in ("/quit", "/exit"):
            break
        now = clock()
        if tick is not None:
            tick(now)
        memory.scan(now)
        p = responder.on_user_message(text, now)
        reply, used = backend.reply(p.system, p.contexts, p.prompt, p.image_paths)
        out(responder.on_reply(reply, clock(), used))  # 〔用到：记N〕标记已删掉


def main() -> None:
    from llm import load_settings, subconscious_model, thinking_model
    from memory import DefaultMemoryService, background_job
    from response.background import make_scheduler

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    memory = DefaultMemoryService()
    scheduler = make_scheduler(background_job(subconscious_model(), load_settings().thinking_context_limit))
    print("命令行平台：输入消息回车，/quit 退出。思考模型：", thinking_model().name)
    try:
        run(
            memory,
            ModelBackend(thinking_model()),
            iter(input, "/quit"),
            tick=lambda _now: scheduler.run_pending(),
        )
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        memory.close()


if __name__ == "__main__":
    main()
