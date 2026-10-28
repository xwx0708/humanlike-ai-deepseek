"""DeepSeek API：OpenAI 兼容接口，多一个"先思考再回答"的开关。"""

from llm.openai_compat import OpenAICompatModel


class DeepSeekModel(OpenAICompatModel):
    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-flash",
        base_url: str = "https://api.deepseek.com",
        thinking: bool = False,
        timeout: float = 120,
        retries: int = 2,
    ):
        # deepseek-flash 默认先思考再回答，思考内容也占 max_tokens，还会拖慢回复；默认关掉
        super().__init__(
            api_key,
            model,
            base_url,
            extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
            timeout=timeout,
            retries=retries,
        )
