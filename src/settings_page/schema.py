"""设置页的说明表（D40、D42）：纯展示层。

参数的名称、解释、范围由各模块在加载时登记到 contracts.knobs.REGISTRY（字段上的 knob()、config.value() 等），
这里只定分区的显示顺序，然后把注册表交给页面。新增一个参数、换掉一个模块，这里一行都不用改。
"""

from typing import Any

from contracts.knobs import REGISTRY
from runtime import config

# 分区归成几个大项（用户 2026-11-02：栏目太多，合并成大项看着舒服）；没列出的分区按登记顺序排在最后一组
GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("基础", ("general", "persona", "models", "lifecycle")),
    ("记忆", ("forgetting", "clock", "scanner", "fresh", "resident", "sweep", "archive")),
    ("整理", ("runner", "merge", "candidates", "boundaries", "profile")),
    ("对话与行动", ("attachments", "action", "features", "keywords")),
)
ORDER = tuple(
    i for _, ids in GROUPS for i in ids
)  # persona / features / keywords 是页面自己画的，不在注册表里

config.load_modules()  # 让各模块登记自己的参数


def schema() -> dict[str, Any]:
    s = REGISTRY.schema(ORDER)
    s["groups"] = [{"title": title, "ids": list(ids)} for title, ids in GROUPS]
    return s
