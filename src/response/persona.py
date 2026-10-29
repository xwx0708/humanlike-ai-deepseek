"""人设：由用户输入，长度一般几千字或更短；回应和主动发消息都用（D29）。

暂定放在 data/persona.md（不进版本库），用户直接编辑这个文件；每轮重新读，改了立即生效。
"""

from pathlib import Path

from runtime import config

DEFAULT_PATH = config.data_dir() / "persona.md"


def load_persona(path: Path = DEFAULT_PATH) -> str:
    p = Path(path)
    return p.read_text(encoding="utf-8").strip() if p.is_file() else ""
