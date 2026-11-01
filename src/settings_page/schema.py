"""设置页的说明表（D40、D42）：纯展示层。

参数的名称、解释、范围由各模块在加载时登记到 contracts.knobs.REGISTRY（字段上的 knob()、config.value() 等），
这里只定分区的显示顺序，然后把注册表交给页面。新增一个参数、换掉一个模块，这里一行都不用改。
"""

from typing import Any

from contracts.knobs import REGISTRY
from runtime import config

# 分区的显示顺序；没列出的分区按登记顺序排在后面
ORDER = (
    "general",
    "models",
    "lifecycle",
    "forgetting",
    "clock",
    "scanner",
    "fresh",
    "resident",
    "sweep",
    "archive",
    "runner",
    "merge",
    "candidates",
    "boundaries",
    "profile",
    "attachments",
    "action",
)

config.load_modules()  # 让各模块登记自己的参数


def schema() -> dict[str, Any]:
    return REGISTRY.schema(ORDER)
