# 回应模块

> 状态：第一版代码已写（2026-10-28，总纲对话按用户"继续"代写），还没在真正的 AstrBot 里跑过。部署见[部署.md](部署.md)。回应对话开始后接手。职责范围见 [架构总图](../00-总纲/架构总图.md)。

## 平台（D28）

**我的设想**：首先可以用 NapCat 挂一个 QQ 小号，再用 AstrBot 填入 DeepSeek 的 API。不一定是 DeepSeek 的 API key，这里也要设置成通用接口。

补充（链路）：

```
QQ 小号 ⇄ NapCat（OneBot v11）──反向 WebSocket──▶ AstrBot（aiocqhttp 适配器，默认端口 6199）
                                                    │
                                                    └─ 插件 ──▶ 本项目：回应模块 ⇄ 记忆 / 情绪 / 行动
模型：AstrBot 的"OpenAI 兼容"提供方，或本项目 src/llm 的 openai_compat 提供方，填地址 + 密钥 + 模型名即可
```

- NapCat：基于 NTQQ 的 OneBot v11 实现，负责登录 QQ 小号、收发消息。
- AstrBot：机器人框架，接 NapCat 收发消息，自带模型提供方、人设、对话历史、插件系统。本项目通过 AstrBot 插件接进去。
- 插件里能用的钩子（查自 AstrBot 文档 docs.astrbot.app，2026-10-28）：
  - `@filter.on_llm_request()`：调用模型前拿到请求（系统提示词、历史、本轮内容），可以改；
  - `@filter.on_llm_response()`：模型回答后拿到结果；
  - `@filter.llm_tool(name=...)`：给模型登记可调用的工具（适合"主动检索记忆""翻档案"）；
  - `event.stop_event()` 加 `yield event.plain_result(...)`：插件自己回复，跳过 AstrBot 默认的模型调用；
  - `self.context.get_using_provider_async(umo=...)`：插件里调用 AstrBot 当前配置的模型。

## 上下文就是记忆系统（D29 第 7 条）

**我的设想**："一般上下文对应的是 AI 在回答时会综合前面所有的聊天记录，现在我则需要 AI 的上下文变成我们搭建的记忆系统。"

补充（每轮组织提示词）：

| 阶段 | 放进上下文的 |
|---|---|
| 熟悉期（记忆系统未启动，D24） | 人设 + 全部聊天记录 |
| 运行期 | 人设 + 常驻栏 + 浮现区（由模型按语境决定提不提）+ 检索结果（需要时）+ 新鲜记忆（刚说过的话，几乎未衰减，D30；不另带聊天记录） |

其他已定：插件接管整轮、思考模型用 AstrBot 里填的、合成一个进程、只陪一个人、人设由用户输入、浮现说不说由模型判断、用户要求靠关键词识别、行动用 AstrBot 自带 MCP、语气由人设和模型决定（D29）。限制汇总见 [使用说明与限制](../00-总纲/使用说明与限制.md)。

记忆模块已提供新鲜记忆：`memory.fresh.fresh_memories` / `fresh_block`（见接口约定）。

## 代码（2026-10-28 第一版）

| 文件 | 做什么 |
|---|---|
| `src/response/responder.py` | 每轮组织内容：熟悉期 = 人设 + 全部聊天记录；运行期 = 人设 + 常驻栏 + 浮现区（去掉和新鲜记忆重复的）+ 新鲜记忆；把用户和 AI 的话存进记忆；回报实际用量 |
| `src/response/requests.py` | 关键词识别：记住、忘掉（及之后的选择）、翻档案 |
| `src/response/persona.py` | 读人设 `data/persona.md` |
| `src/response/background.py` | 记忆后台任务登记进共用调度，和 AstrBot 合成一个进程 |
| `src/response/astrbot_plugin/` | AstrBot 插件：只回那一个 QQ 号的私聊，接管整轮，用 AstrBot 的模型和全部工具（含 MCP） |

2026-10-29 补充：每轮加一句人设提醒（熟悉期放在旧记录之后）；被唤起的旧事实进浮现区（D16）；能读图片和文件（D38）。

还没做：记忆使用回执（待议问题 14）、主动发消息（10）、非文字消息（11）、出错时说什么（22）、"要求修改"（17）。

## 暂定职责

- 思考模型：处理"当下"的对话和推理
- 每轮查看记忆的浮现区，决定提不提起；需要时主动检索记忆；用户要求时请记忆模块翻档案（D26：AI 不能自己翻）
- 语气由人设和模型共同决定（D29）；以后情绪模块做出来，再看要不要参考它给出的信息
- 决定何时调用行动模块
- 用到某条记忆时回报给记忆模块（用于加固，`memory.scanner.use_memory`）

## 记忆模块已经提供的接口

见 [接口约定](../00-总纲/接口约定.md)：常驻栏 `resident_block`、浮现区 `Scanner`、记忆使用回执 `use_memory`、主动检索 `search_memory`、档案 `forget_thread` / `search_archive`、全局生命周期 `Lifecycle().phase`。

## 还没想清楚的事

见 [待议问题.md](待议问题.md)。
