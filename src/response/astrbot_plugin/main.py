"""AstrBot 插件：把本项目接进 AstrBot（D28、D29）。

- 插件接管整轮（D29）：收到消息后由本项目组织提示词，再用 AstrBot 里配置的思考模型回答，跳过 AstrBot 默认流程。
- 只陪一个人、不考虑群聊（D29）：只回复配置里那个 QQ 号的私聊，其他私聊和群聊一律不回。
- 行动用 AstrBot 自带的 MCP（D29）：把 AstrBot 登记的全部工具（含 MCP）交给模型，用 tool_loop_agent 跑工具调用。
- 小d 自己的事（D35）：换头像、改昵称和签名、发说说、删说说登记成工具，由模型自己决定做不做；
  频率、长度、图片来源由 action.qq_self 把关，做成以后记进记忆。
- 联网搜索（D36）：用 AstrBot 自带的，在它网页里开启、选来源、填密钥；插件把它加进工具。
- 和 AstrBot 合成一个进程（D29）：记忆的后台任务每分钟在另一个线程里跑一次；扫描器每分钟扫一格。

安装方法见 03-回应/部署.md。这个文件只能在 AstrBot 里运行，没有单元测试；AstrBot 的接口按 2026-10-28 的文档和源码写，
装好后要实测。
"""

import asyncio
import os
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
        # 插件配置优先；Docker 部署时由环境变量给出（docker-compose.yml）
        root = Path(config.get("project_root") or os.environ.get("HUMANLIKE_ROOT") or DEFAULT_ROOT)
        if str(root / "src") not in sys.path:
            sys.path.insert(0, str(root / "src"))

        from action.qq_self import QQSelf
        from llm import subconscious_model
        from llm.settings import load_settings
        from memory.scanner import Scanner
        from memory.store import MemoryStore
        from response.background import make_scheduler
        from response.persona import load_persona
        from response.responder import Responder

        self.owner = str(config.get("owner_qq") or os.environ.get("HUMANLIKE_OWNER_QQ") or "").strip()
        self.store = MemoryStore()
        self.scanner = Scanner(self.store)
        self.responder = Responder(self.store, self.scanner, load_persona)
        self.qq = QQSelf()
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
        attachments = await self._attachments(event)
        if not text and not attachments:
            return  # 表情、戳一戳等没有文字也没有文件的消息，暂不回
        self.store.set_meta("owner_umo", event.unified_msg_origin)  # 以后主动发消息用（D29）
        prompt = self.responder.on_user_message(text, time.time(), attachments)
        try:
            provider_id = await self.context.get_current_chat_provider_id(event.unified_msg_origin)
            resp = await self.context.tool_loop_agent(
                event=event,
                chat_provider_id=provider_id,
                prompt=prompt.prompt,
                system_prompt=prompt.system,
                contexts=prompt.contexts,
                image_urls=prompt.image_paths or None,
                tools=await self._tools(event),
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

    async def _attachments(self, event: AstrMessageEvent) -> list:
        """这条消息里的图片和文件（D38）：下载到本地，图片交给模型看，文件提取成文字。MP4 等视频、音频不支持。"""
        import astrbot.api.message_components as Comp

        from response.attachments import read_file, read_image, unsupported

        out = []
        for comp in event.message_obj.message:
            try:
                if isinstance(comp, Comp.Image):
                    out.append(read_image(await comp.convert_to_file_path(), "图片"))
                elif isinstance(comp, Comp.File):
                    name = comp.name or "文件"
                    out.append(read_file(await comp.get_file(), name))
                elif isinstance(comp, Comp.Video):
                    out.append(unsupported("视频", "视频还看不了"))
                elif isinstance(comp, Comp.Record):
                    out.append(unsupported("语音", "语音还听不了"))
            except Exception as e:  # noqa: BLE001  下载失败等，如实告诉模型
                logger.exception("[类人AI] 读取附件失败")
                out.append(unsupported(getattr(comp, "name", "") or "附件", f"没拿到（{type(e).__name__}）"))
        return out

    async def _tools(self, event: AstrMessageEvent):
        """交给模型的工具：AstrBot 登记的全部工具（含 MCP 和本插件的），再加上 AstrBot 自带的联网搜索（D36）。
        联网搜索只在 AstrBot 默认流程里才会加，插件接管整轮后要自己补；是否开启、用哪家，按 AstrBot 网页里的设置。"""
        tools = self.context.get_llm_tool_manager().get_full_tool_set()
        try:
            from types import SimpleNamespace

            from astrbot.core.astr_main_agent import _apply_web_search_tools

            await _apply_web_search_tools(event, SimpleNamespace(func_tool=tools), self.context)
        except Exception:  # noqa: BLE001  AstrBot 改了内部接口时退回不带搜索，不影响聊天
            logger.exception("[类人AI] 加联网搜索工具失败，这一轮不带搜索")
        return tools

    # ---------- 小d 自己的事（D35：严格限制频率，可以自己决定） ----------

    async def _do(self, event: AstrMessageEvent, build) -> str:
        from action.qq_self import Refused

        bot = getattr(event, "bot", None)
        if bot is None or not hasattr(bot, "call_action"):
            return "没做成：这件事只能在 QQ 上做"
        try:
            call = build(self.qq)
        except Refused as e:
            return f"没做成：{e}"
        try:
            result = await bot.call_action(call.action, **call.params)
        except Exception as e:  # noqa: BLE001
            logger.exception(f"[类人AI] {call.kind}失败")
            return f"没做成：QQ 那边出错了（{e}）"
        self.qq.record(call, result if isinstance(result, dict) else None)
        self.responder.on_action(call.note, time.time())
        logger.info(f"[类人AI] {call.note}")
        return f"做好了：{call.note}"

    @filter.llm_tool(name="qq_list_images")
    async def qq_list_images(self, event: AstrMessageEvent):
        """列出可以用来换头像、给说说配图的图片文件名。"""
        names = self.qq.list_images()
        return "能用的图片：" + "、".join(names) if names else "图片文件夹里还没有图"

    @filter.llm_tool(name="qq_set_avatar")
    async def qq_set_avatar(self, event: AstrMessageEvent, image: str):
        """换自己的 QQ 头像。想换就可以自己决定换，但一周最多一次。

        Args:
            image(string): 图片文件名，只能用 qq_list_images 列出的
        """
        return await self._do(event, lambda q: q.set_avatar(image))

    @filter.llm_tool(name="qq_set_nickname")
    async def qq_set_nickname(self, event: AstrMessageEvent, nickname: str):
        """改自己的 QQ 昵称。一周最多一次。

        Args:
            nickname(string): 新昵称，最多 20 个字
        """
        return await self._do(event, lambda q: q.set_nickname(nickname))

    @filter.llm_tool(name="qq_set_signature")
    async def qq_set_signature(self, event: AstrMessageEvent, signature: str):
        """改自己的 QQ 个性签名。三天最多一次。

        Args:
            signature(string): 新签名，最多 60 个字
        """
        return await self._do(event, lambda q: q.set_signature(signature))

    @filter.llm_tool(name="qq_post_moment")
    async def qq_post_moment(self, event: AstrMessageEvent, content: str, images: str = ""):
        """在自己的 QQ 空间发一条说说（好友可见）。想分享心情或日常时可以自己决定发，但一天最多一条。

        Args:
            content(string): 说说正文，最多 500 字
            images(string): 配图文件名，多张用逗号分开，只能用 qq_list_images 列出的；不配图就留空
        """
        names = [i.strip() for i in images.replace("，", ",").split(",") if i.strip()]
        return await self._do(event, lambda q: q.post(content, names))

    @filter.llm_tool(name="qq_delete_moment")
    async def qq_delete_moment(self, event: AstrMessageEvent, tid: str):
        """删掉自己之前发的一条说说。一天最多两次，只能删自己发的。

        Args:
            tid(string): 说说编号；不知道编号时先留空调用一次，会列出自己发过的说说
        """
        if not tid.strip():
            posts = self.qq.my_posts()
            return (
                "我发过的说说：" + "；".join(f"{p['tid']}：{p['content'][:30]}" for p in posts)
                if posts
                else "我还没发过说说"
            )
        return await self._do(event, lambda q: q.delete_post(tid.strip()))

    async def terminate(self):
        if self._task is not None:
            self._task.cancel()
        self.store.close()
