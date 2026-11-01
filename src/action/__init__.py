"""行动模块（手脚）：AI 自己做的事。第一版是 QQ 上的"自己的事"：换头像、改昵称和签名、发说说（D35）。

外部工具（MCP）直接用 AstrBot 自带的（D29）；这里只放需要本项目自己把关的动作。
"""

from action import qq_self  # 加载时登记参数与功能开关（D42）
from action.qq_self import QQSelf, Refused  # 组装处（插件）从这里取

__all__ = ["QQSelf", "Refused", "qq_self"]
