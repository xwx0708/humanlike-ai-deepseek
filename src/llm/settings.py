"""模型配置：先读环境变量，没有的再读项目根目录的 .env，再没有就用默认值。

.env 不进版本库，格式见根目录的 .env.example。
"""

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


PROVIDERS = ("openai_compat", "deepseek", "ollama")


@dataclass(frozen=True)
class Settings:
    # 两个角色各自用哪家、哪个模型，可以随意搭配（决策 D13）
    thinking_provider: str = "deepseek"
    thinking_model: str = "deepseek-flash"
    thinking_context_limit: int = (
        128000  # 思考模型的上下文上限（token）。熟悉期用到七成就启动记忆系统（D24）；换模型时要改
    )
    subconscious_provider: str = "ollama"
    subconscious_model: str = "qwen3.8:27b-q4_K_M"
    # 各家的连接设置，两个角色共用
    # 通用：任何 OpenAI 兼容接口（D28）。填地址、密钥就能用，不限于 DeepSeek
    openai_compat_api_key: str = ""
    openai_compat_base_url: str = ""
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_thinking: bool = False  # 开启后先思考再回答，更慢、更费 token
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_keep_alive: str = "30m"  # 空闲多久后卸载模型；首次加载约 1.5 分钟，所以留久一点
    ollama_num_ctx: int = 16384  # 上下文长度；3090 加载模型后约剩 7GB，太大会爆显存


def read_dotenv(path: Path) -> dict[str, str]:
    values = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if value[:1] in ("'", '"') and value[0] in value[1:]:
            value = value[1 : value.index(value[0], 1)]  # 引号里的内容原样保留，包括 #
        else:
            value = value.split(" #", 1)[0].split("\t#", 1)[0].strip()  # 去掉行尾注释
        values[key.strip()] = value
    return values


def load_settings(dotenv: Path = ROOT / ".env") -> Settings:
    file_values = read_dotenv(dotenv)
    overrides = {}
    for name, default in Settings.__dataclass_fields__.items():
        raw = os.environ.get(name.upper(), file_values.get(name.upper()))
        if raw is None or raw == "":
            continue
        if isinstance(default.default, bool):
            overrides[name] = raw.lower() in ("1", "true", "yes", "on")
        else:
            overrides[name] = type(default.default)(raw)
    return Settings(**overrides)
