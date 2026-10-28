"""真实调用：本地 Qwen 和 DeepSeek API。默认不运行，用 `pytest -m live` 手动运行。

DeepSeek 每次调用会花一点钱（很少的 token）；Qwen 首次调用要加载模型，约 1.5 分钟。
"""

import pytest

from contracts.llm import user
from llm import subconscious_model, thinking_model

pytestmark = pytest.mark.live

SCHEMA = {
    "type": "object",
    "properties": {"topic_changed": {"type": "boolean"}, "confidence": {"type": "number"}},
    "required": ["topic_changed", "confidence"],
}
QUESTION = user("上一句：我妈最近老咳嗽。这一句：对了，周末去爬山吗？话题是否变了？给出 0～1 的把握值。")


def test_qwen_structured_output():
    r = subconscious_model().chat([QUESTION], schema=SCHEMA, temperature=0.2)
    assert r.data["topic_changed"] is True
    assert 0 <= r.data["confidence"] <= 1


def test_deepseek_structured_output():
    r = thinking_model().chat([QUESTION], schema=SCHEMA, max_tokens=60)
    assert r.data["topic_changed"] is True
