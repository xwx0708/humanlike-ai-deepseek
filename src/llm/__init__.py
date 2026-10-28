"""模型调用的具体实现：DeepSeek、Ollama、FakeModel（测试用）。

思考模型和潜意识模型都只依赖 contracts.llm.ChatModel 这个通用接口，各自用哪家、哪个模型
由配置决定，可以随意搭配（决策 D13）。默认：思考模型用 DeepSeek，潜意识模型用本地 Ollama 上的 Qwen。

一般用法：
    from llm import thinking_model, subconscious_model
    from contracts.llm import user
    reply = subconscious_model().chat([user("...")], schema={...})
"""

from contracts.llm import ChatModel
from llm.deepseek import DeepSeekModel
from llm.fake import FakeModel
from llm.ollama import OllamaModel
from llm.settings import PROVIDERS, Settings, load_settings


def make_model(provider: str, model: str, settings: Settings | None = None) -> ChatModel:
    """按"哪家 + 哪个模型"创建模型。新增一家时，在这里加一个分支。"""
    s = settings or load_settings()
    if provider == "deepseek":
        if not s.deepseek_api_key:
            raise RuntimeError("没有找到 DEEPSEEK_API_KEY：请设为环境变量，或写进项目根目录的 .env")
        return DeepSeekModel(
            api_key=s.deepseek_api_key,
            model=model,
            base_url=s.deepseek_base_url,
            thinking=s.deepseek_thinking,
        )
    if provider == "ollama":
        return OllamaModel(
            model=model, base_url=s.ollama_base_url, keep_alive=s.ollama_keep_alive, num_ctx=s.ollama_num_ctx
        )
    raise ValueError(f"不认识的模型提供方 {provider!r}，可选：{', '.join(PROVIDERS)}")


def thinking_model() -> ChatModel:
    s = load_settings()
    return make_model(s.thinking_provider, s.thinking_model, s)


def subconscious_model() -> ChatModel:
    s = load_settings()
    return make_model(s.subconscious_provider, s.subconscious_model, s)


__all__ = ["DeepSeekModel", "FakeModel", "OllamaModel", "make_model", "subconscious_model", "thinking_model"]
