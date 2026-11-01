"""模型配置：先读环境变量，没有的再读项目根目录的 .env，再没有就用默认值。

.env 不进版本库，格式见根目录的 .env.example。
"""

import os
from dataclasses import dataclass
from pathlib import Path

from contracts.knobs import knob
from runtime import config

ROOT = Path(__file__).resolve().parents[2]


PROVIDERS = ("openai_compat", "deepseek", "ollama")


@dataclass(frozen=True)
class Settings:
    # 两个角色各自用哪家、哪个模型，可以随意搭配（决策 D13）
    thinking_provider: str = "deepseek"
    thinking_model: str = "deepseek-flash"
    thinking_context_limit: int = knob(
        1000000,
        "思考模型上下文上限",
        "思考模型的最大上下文长度（token）。熟悉期用到这个值的启动比例时，记忆系统启动。"
        "Docker 部署时思考模型在 AstrBot 网页里选，换模型时这里要同步修改。",
        lo=4000,
        hi=4000000,
        step=1000,
        unit="token",
    )  # 熟悉期用到七成就启动记忆系统（D24）；deepseek-flash 约 100 万（D39），换模型时要改
    subconscious_provider: str = knob(
        "ollama",
        "潜意识模型提供方",
        "ollama：本地运行（推荐，记忆整理不出本机，需要显卡）；openai_compat：任意 OpenAI 兼容的云端或本地服务；"
        "deepseek：DeepSeek API。",
        kind="select",
        options=["ollama", "openai_compat", "deepseek"],
    )
    subconscious_model: str = knob(
        "qwen3.8:27b-q4_K_M",
        "潜意识模型",
        "模型名。默认 Qwen 27B Q4 量化约需 17 GB 显存（如 RTX 3090/4090）；显存不足可换 7B～14B 的模型，整理质量会下降。",
    )
    # 各家的连接设置，两个角色共用
    # 通用：任何 OpenAI 兼容接口（D28）。填地址、密钥就能用，不限于 DeepSeek
    openai_compat_api_key: str = knob("", secret=True)  # 密钥只放 .env，不上设置页
    openai_compat_base_url: str = knob(
        "",
        "OpenAI 兼容接口地址",
        "提供方选 openai_compat 时使用，例如 https://api.deepseek.com、http://127.0.0.1:8000/v1。"
        "密钥写在 .env 的 OPENAI_COMPAT_API_KEY。",
    )
    deepseek_api_key: str = knob("", secret=True)
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_thinking: bool = knob(
        False,
        "DeepSeek 思考模式",
        "开启后先推理再回答，更准但更慢、更费 token。只对本项目直接调用的 DeepSeek 生效。",
    )  # 开启后先思考再回答，更慢、更费 token
    ollama_base_url: str = knob(
        "http://127.0.0.1:11434",
        "Ollama 地址",
        "本机 Ollama 服务地址。Docker 部署时由 docker-compose.yml 改写为 host.docker.internal。",
    )
    ollama_keep_alive: str = knob(
        "30m", "Ollama 保持加载", "空闲多久后卸载模型，如 30m、2h。首次加载大模型约需 1～2 分钟。"
    )  # 首次加载约 1.5 分钟，所以留久一点
    ollama_num_ctx: int = knob(
        16384,
        "Ollama 上下文长度",
        "潜意识模型一次能看的长度。越大越占显存；切段的每批长度（整理 → 切段批量）要明显小于它。",
        lo=2048,
        hi=131072,
        step=1024,
        unit="token",
    )  # 3090 加载模型后约剩 7GB，太大会爆显存


config.register(
    Settings(),
    "models",
    "模型",
    "双模型架构：思考模型负责当下的对话，潜意识模型在后台整理记忆。两者都可以是本地模型或云端 API。"
    "API 密钥只写在 .env，不在这里填写。",
)


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
    """优先级：环境变量 > 设置页（settings.json 的 models 分区，D40）> .env > 默认值。密钥只放 .env，不进设置页。"""
    file_values = read_dotenv(dotenv)
    page = config.section("models")
    overrides = {}
    for name, default in Settings.__dataclass_fields__.items():
        raw = os.environ.get(name.upper())
        if raw is None or raw == "":
            raw = page.get(name)
        if raw is None or raw == "":
            raw = file_values.get(name.upper())
        if raw is None or raw == "":
            continue
        if isinstance(default.default, bool):
            overrides[name] = raw if isinstance(raw, bool) else str(raw).lower() in ("1", "true", "yes", "on")
        else:
            overrides[name] = type(default.default)(raw)
    return Settings(**overrides)
