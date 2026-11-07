"""运行期上下文的总预算（项目总评 2026-11-02 第一步，用户 2026-11-07 要求做）。

运行期每轮交给思考模型的记忆分三段：常驻栏、浮现区、新鲜记忆。原先三段都没有长度上限：常驻栏带全部画像和全部
常驻事件线（免竞争的没有名额限制），新鲜记忆只按强度筛。用久了会重新变成"越来越长的上下文"，和愿景
"不靠长上下文"相反。这里给每段一个 token 配额，三段之和就是记忆部分的总预算。

超出时怎么取舍（补充，暂定）：
- 常驻栏：用户画像在前，其次免竞争的事件线，最后竞争得来的；放不下的不列，只注明还有几条。
- 浮现区：按浮现的先后放，放不下的这一轮不给。
- 新鲜记忆：从最近一条往前放，越早的越先让出位置；最近一条单独就超出时截短。
让出位置的不会丢：它们照常在概率海里遗忘、浮现，未整理的照常整理。
token 数按 memory.tokens 粗估（宁可估多）。
"""

from dataclasses import dataclass

from contracts.knobs import knob
from memory.tokens import estimate_tokens
from runtime import config


@dataclass(frozen=True)
class BudgetKnobs:
    resident: int = knob(
        3000,
        "常驻栏预算",
        "常驻栏（画像 + 一直记得的事）最多占多少 token。超出时先保留画像，其次免竞争的事件线。",
        lo=200,
        hi=100000,
        step=100,
        unit="token",
    )  # 暂定
    surfaced: int = knob(
        1500, "浮现区预算", "浮现区最多占多少 token。", lo=100, hi=50000, step=100, unit="token"
    )  # 暂定
    fresh: int = knob(
        8000,
        "新鲜记忆预算",
        "刚说过的话最多占多少 token。超出时从最早的开始让出位置（它们照常整理进记忆）。",
        lo=500,
        hi=200000,
        step=500,
        unit="token",
    )  # 暂定：约等于一场几十个来回的对话


DEFAULT_KNOBS = config.knobs(
    BudgetKnobs(),
    "budget",
    "上下文预算",
    "运行期每轮交给思考模型的记忆分三段，各有上限，三段之和就是记忆部分的总预算。",
)


def fit_lines(lines: list[str], budget: int, more: str = "（另有 {n} 条因篇幅没列出）") -> list[str]:
    """按先后放进预算，放不下的不列；有没放下的，末尾注明几条（注明这一行也算在预算里）。"""
    costs = [estimate_tokens(line) + 1 for line in lines]  # +1：换行
    if sum(costs) <= budget:
        return list(lines)
    room = budget - (estimate_tokens(more.format(n=len(lines))) + 1)
    if room < 10:  # 连截短的一条都放不下，只注明
        return [more.format(n=len(lines))]
    out, used = [], 0
    for line, cost in zip(lines, costs, strict=True):
        if used + cost > room:
            break
        out.append(line)
        used += cost
    if not out:  # 第一条单独就放不下：截短
        out = [truncate(lines[0], room)]
    left = len(lines) - len(out)
    return out + [more.format(n=left)] if left else out


def truncate(text: str, budget: int) -> str:
    """截到不超过预算，末尾标"……"。"""
    if estimate_tokens(text) <= budget:
        return text
    lo, hi = 0, len(text)
    while lo < hi:  # 二分找最长的前缀
        mid = (lo + hi + 1) // 2
        if estimate_tokens(text[:mid]) + 1 <= budget:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] + "……"
