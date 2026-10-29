# 行动模块（手脚）

> 状态：第一批代码已写（2026-10-29，总纲对话按用户要求代写），还没有专门的行动对话。职责范围见 [架构总图](../00-总纲/架构总图.md)。

## 暂定职责

- 通过 MCP 等协议调用外部工具，即 AI 的"手脚"。第一版直接用 AstrBot 自带的 MCP（D29）。
- 把执行结果交还给回应模块。
- 把"做过什么"作为事件交给记忆模块（对应记忆文档中的问题："AI 自己说过的话、做过的承诺要不要记？"）。

## 联网搜索（D36）

用 AstrBot 自带的网页搜索，在 AstrBot 网页的配置里开启、选来源（暂定 DeepSeek Search API）、填密钥。插件接管整轮后 AstrBot 不会自己加搜索工具，由插件补上。搜不搜由模型判断。

## 小d 在 QQ 上自己的事（D35）

**我的设想**："如果我想让他有权限更改自己的头像发动态之类的呢""在严格限制频率的情况下可以自我决定。"

| 工具 | 做什么 | NapCat 接口 | 频率上限（暂定） |
|---|---|---|---|
| `qq_list_images` | 列出能用的图片 | — | — |
| `qq_set_avatar` | 换头像 | `set_qq_avatar` | 一周一次 |
| `qq_set_nickname` | 改昵称（≤20 字） | `set_qq_profile` | 一周一次 |
| `qq_set_signature` | 改签名（≤60 字） | `set_self_longnick` | 三天一次 |
| `qq_post_moment` | 发说说（≤500 字、≤4 张图，好友可见） | `send_qzone_msg` | 一天一条 |
| `qq_delete_moment` | 删自己发的说说 | `delete_qzone_msg` | 一天两次 |

- 代码：`src/action/qq_self.py` 负责把关（频率、字数、图片只能来自 `data/images/`、只能删自己发的），生成要调用的接口和参数；`src/response/astrbot_plugin/main.py` 登记成工具、调用 NapCat，做成以后记一次，并把"我做了什么"记进记忆。
- 只有真正做成才算一次；被拒或调用失败不占名额。记录在 `data/action_log.json`。
- 补充（查证）：NapCat 官方文档没写发说说，但镜像（2026-10-29）里有 `send_qzone_msg` / `delete_qzone_msg`，参数是从源码读的。

还没验证：在 QQ 上真的让小d 换一次头像、发一条说说。其余暂定做法见[待你确认](../00-总纲/待你确认.md) A1～A6。
