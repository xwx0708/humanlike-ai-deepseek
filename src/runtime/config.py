"""用户设置（D40）：设置页写进项目根目录的 settings.json，各模块启动时从这里读；没写的用代码里的默认值。

- 文件位置可以用环境变量 HUMANLIKE_SETTINGS 指定（测试时指向不存在的文件，用纯默认值）。
- 结构：{"分区名": {"参数名": 值, ...}, "features": {"功能名": true/false}, "keywords": {"列表名": {"add": [...], "disabled": [...]}}}
- 改了设置要重启才生效（AstrBot：docker compose restart astrbot）。
- 设置页的说明和范围在 settings_page/schema.py。
"""

import json
import os
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def settings_path() -> Path:
    return Path(os.environ.get("HUMANLIKE_SETTINGS") or ROOT / "settings.json")


def load() -> dict[str, Any]:
    p = settings_path()
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}  # 文件坏了就当没有，用默认值；设置页保存时会重写
    return data if isinstance(data, dict) else {}


def save(data: dict[str, Any]) -> None:
    p = settings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def section(name: str) -> dict[str, Any]:
    v = load().get(name)
    return v if isinstance(v, dict) else {}


def _cast(value: Any, like: Any) -> Any:
    if isinstance(like, bool):
        return bool(value)
    if isinstance(like, int):
        return int(value)
    if isinstance(like, float):
        return float(value)
    return value


def knobs[T](obj: T, name: str) -> T:
    """把设置文件里这个分区的值覆盖到一组参数（frozen dataclass 实例）上。不认识的键忽略。"""
    sec = section(name)
    names = {f.name for f in fields(obj)}
    changes = {k: _cast(v, getattr(obj, k)) for k, v in sec.items() if k in names}
    return replace(obj, **changes) if changes else obj


def value[T](name: str, key: str, default: T) -> T:
    sec = section(name)
    return _cast(sec[key], default) if key in sec else default


def enabled(feature: str, default: bool = True) -> bool:
    """功能开关：设置页可以关掉某个功能。"""
    return bool(section("features").get(feature, default))


def words(name: str, defaults: tuple[str, ...]) -> tuple[str, ...]:
    """可自定义的词表：在默认词里去掉停用的，再加上用户加的。"""
    sec = section("keywords").get(name) or {}
    disabled = set(sec.get("disabled") or [])
    added = [w for w in (sec.get("add") or []) if isinstance(w, str) and w.strip()]
    out = [w for w in defaults if w not in disabled]
    out += [w.strip() for w in added if w.strip() not in out]
    return tuple(out)


def data_dir() -> Path:
    """数据文件夹：记忆库（含档案）、人设、生命周期状态、行动记录、图片都在这里。
    优先级：环境变量 HUMANLIKE_DATA_DIR > 设置页 > 项目里的 data/。相对路径按项目根目录算。"""
    raw = os.environ.get("HUMANLIKE_DATA_DIR") or section("general").get("data_dir") or "data"
    p = Path(raw).expanduser()
    return p if p.is_absolute() else ROOT / p
