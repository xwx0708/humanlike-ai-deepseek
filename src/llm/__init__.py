"""模型调用的具体实现：DeepSeek（思考模型）、Ollama（潜意识模型）、FakeModel（测试用）。

一般用法：
    from llm import thinking_model, subconscious_model
    from contracts.llm import user
    reply = subconscious_model().chat([user("...")], schema={...})
"""

from llm.deepseek import DeepSeekModel
from llm.fake import FakeModel
from llm.ollama import OllamaModel
from llm.settings import load_settings


def thinking_model() -> DeepSeekModel:
    s = load_settings()
    if not s.deepseek_api_key:
        raise RuntimeError("没有找到 DEEPSEEK_API_KEY：请设为环境变量，或写进项目根目录的 .env")
    return DeepSeekModel(
        api_key=s.deepseek_api_key,
        model=s.deepseek_model,
        base_url=s.deepseek_base_url,
        thinking=s.deepseek_thinking,
    )


def subconscious_model() -> OllamaModel:
    s = load_settings()
    return OllamaModel(
        model=s.ollama_model,
        base_url=s.ollama_base_url,
        keep_alive=s.ollama_keep_alive,
        num_ctx=s.ollama_num_ctx,
    )


__all__ = ["DeepSeekModel", "FakeModel", "OllamaModel", "subconscious_model", "thinking_model"]
