"""AstrBot 插件：把本项目接进 AstrBot（D28、D29）。

- 插件接管整轮（D29）：收到消息后由本项目组织提示词，再用 AstrBot 里配置的思考模型回答，跳过 AstrBot 默认流程。
- 只陪一个人、不考虑群聊（D29）：只回复配置里那个 QQ 号的私聊，其他私聊和群聊一律不回。
- 行动用 AstrBot 自带的 MCP（D29）：把 AstrBot 登记的全部工具（含 MCP）交给模型，用 tool_loop_agent 跑工具调用。
- 和 AstrBot 合成一个进程（D29）：记忆的后台任务每分钟在另一个线程里跑一次；扫描器每分钟扫一格。

安装方法见 03-回应/部署.md。这个文件只能在 AstrBot 里运行，没有单元测试；AstrBot 的接口按 2026-10-28 的文档和源码写，
装好后要实测。
"""

import asyncio
import sys
import time
from pathlib import Path

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star

DEFAULT_ROOT = Path(__file__).resolve().parents[3]  # 插件放在项目的 src/response/astrbot_plugin/ 里时


class HumanlikePlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        root = Path(config.get("project_root") or DEFAULT_ROOT)
        if str(root / "src") not in sys.path:
            sys.path.insert(0, str(root / "src"))

        from llm import subconscious_model
        from llm.settings import load_settings
        from memory.scanner import Scanner
        from memory.store import MemoryStore
        from response.background import make_scheduler
        from response.persona import load_persona
        from response.responder import Responder

        self.owner = str(config.get("owner_qq", "")).strip()
        self.store = MemoryStore()
        self.scanner = Scanner(self.store)
        self.responder = Responder(self.store, self.scanner, load_persona)
        # 潜意识模型仍在本项目 .env；思考模型用 AstrBot 里填的（D29）
        self.scheduler = make_scheduler(subconscious_model(), load_settings().thinking_context_limit)
        self._task: asyncio.Task | None = None

    async def initialize(self):
        self._ensure_background()

    def _ensure_background(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._background())

    async def _background(self) -> None:
        reported = 0
        while True:
            try:
                self.scanner.tick(time.time())  # 扫描器用的是收发消息这个线程的连接，在这里跑
                await asyncio.to_thread(self.scheduler.run_pending)  # 整理可能要几分钟，放到别的线程
                for name, ts, err in self.scheduler.errors[reported:]:
                    logger.error(f"[类人AI] 后台任务「{name}」出错：{err}")
                reported = len(self.scheduler.errors)
            except Exception:  # noqa: BLE001  后台出错不能让插件停下
                logger.exception("[类人AI] 后台循环出错")
            await asyncio.sleep(60)

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def on_group(self, event: AstrMessageEvent):
        event.stop_event()  # 不考虑群聊（D29）：群里一律不回

    @filter.event_message_type(filter.EventMessageType.PRIVATE_MESSAGE)
    async def on_private(self, event: AstrMessageEvent):
        event.stop_event()  # 接管整轮，AstrBot 默认流程不再回复（D29）
        self._ensure_background()
        if not self.owner or event.get_sender_id() != self.owner:
            return  # 只陪一个人（D29）
        text = event.message_str.strip()
        if not text:
            return  # 图片、语音等非文字消息怎么处理还没定（待议问题 11）
        self.store.set_meta("owner_umo", event.unified_msg_origin)  # 以后主动发消息用（D29）
        prompt = self.responder.on_user_message(text, time.time())
        try:
            provider_id = await self.context.get_current_chat_provider_id(event.unified_msg_origin)
            resp = await self.context.tool_loop_agent(
                event=event,
                chat_provider_id=provider_id,
                prompt=prompt.prompt,
                system_prompt=prompt.system,
                contexts=prompt.contexts,
                tools=self.context.get_llm_tool_manager().get_full_tool_set(),
            )
        except Exception:  # noqa: BLE001
            logger.exception("[类人AI] 调用思考模型失败")
            return  # 出错时说什么还没定（待议问题 22），暂时不回
        reply = (resp.completion_text or "").strip()
        if not reply:
            return
        used = resp.usage.input_other + resp.usage.input_cached if resp.usage else None
        self.responder.on_reply(reply, time.time(), used)
        yield event.plain_result(reply)

    async def terminate(self):
        if self._task is not None:
            self._task.cancel()
        self.store.close()
