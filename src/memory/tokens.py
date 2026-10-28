"""粗估文字占多少 token。偏保守（宁可估多）：中日韩字符每个按 1 个算，其余每 3 个字符按 1 个算。

暂行：等回应模块接入后，改用思考模型实际回报的 token 数（Reply.tokens）。
"""

import re

_CJK = re.compile(r"[\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\uff00-\uffef]")


def estimate_tokens(text: str) -> int:
    cjk = len(_CJK.findall(text))
    return cjk + (len(text) - cjk + 2) // 3
