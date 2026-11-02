"""用户设置（D40）：设置页写进项目根目录的 settings.json，各模块启动时从这里读；没写的用代码里的默认值。

- 文件位置可以用环境变量 HUMANLIKE_SETTINGS 指定（测试时指向不存在的文件，用纯默认值）。
- 结构：{"分区名": {"参数名": 值, ...}, "features": {"功能名": true/false}, "keywords": {"列表名": {"add": [...], "disabled": [...]}}}
- 改了设置要重启才生效（AstrBot：docker compose restart astrbot）。
- 参数的名称、解释、范围由各模块在加载时登记到 contracts.knobs.REGISTRY（D42）；设置页只遍历注册表。
"""

import json
import os
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

from contracts.knobs import REGISTRY

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


def register[T](obj: T, name: str, title: str = "", help: str = "") -> T:
    """把一组参数（frozen dataclass 实例，字段用 contracts.knobs.knob 写说明）登记到注册表；title 非空时同时登记分区说明。"""
    REGISTRY.section(name, title, help)
    REGISTRY.add_dataclass(obj, name)
    return obj


def section_info(name: str, title: str, help: str = "") -> None:
    """只登记分区的标题和说明（分区里的值用 value() 单独登记时）。"""
    REGISTRY.section(name, title, help)


def knobs[T](obj: T, name: str, title: str = "", help: str = "") -> T:
    """把设置文件里这个分区的值覆盖到一组参数（frozen dataclass 实例）上，并把它登记到注册表。不认识的键忽略。"""
    register(obj, name, title, help)
    sec = section(name)
    names = {f.name for f in fields(obj)}
    changes = {k: _cast(v, getattr(obj, k)) for k, v in sec.items() if k in names}
    return replace(obj, **changes) if changes else obj


def value[T](name: str, key: str, default: T, label: str | None = None, help: str = "", **meta) -> T:
    """读一个单独的值；给了 label 就同时登记到注册表（lo、hi、step、unit、kind、options、option_labels）。"""
    if label:
        REGISTRY.add_value(name, key, default, label, help, **meta)
    sec = section(name)
    return _cast(sec[key], default) if key in sec else default


def feature(key: str, label: str, help: str = "") -> str:
    """登记一个功能开关（在拥有这个功能的模块文件顶层调用一次），返回它的名字。"""
    REGISTRY.add_feature(key, label, help)
    return key


def enabled(feature: str, default: bool = True) -> bool:
    """功能开关：设置页可以关掉某个功能。"""
    return bool(section("features").get(feature, default))


def words(name: str, defaults: tuple[str, ...], label: str | None = None, help: str = "") -> tuple[str, ...]:
    """可自定义的词表：在默认词里去掉停用的，再加上用户加的。给了 label 就登记到注册表。"""
    if label:
        REGISTRY.add_words(name, label, help, defaults)
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


MODULES = ("llm", "memory", "response", "action")  # 有参数要登记的模块；新模块在这里加一个名字


def load_modules(names: tuple[str, ...] = MODULES) -> None:
    """把各模块加载一遍，让它们登记自己的参数（设置页启动时调用）。"""
    import importlib

    for name in names:
        importlib.import_module(name)


REGISTRY.section(
    "general", "数据存放", "记忆库（含档案库）、人设、生命周期状态、行动记录、图片都放在数据文件夹里。"
)
REGISTRY.add_value(
    "general",
    "ai_name",
    "大肥鱼",
    "它的名字",
    "你给它起的名字，请和人设里写的名字保持一致。设置页的说明会用这个称呼；聊天里它叫什么由人设决定。",
)
REGISTRY.add_value(
    "general",
    "data_dir",
    "data",
    "数据文件夹",
    "记忆库 memory.db（概率海与档案库）、persona.md、lifecycle.json、action_log.json、images/ 的存放位置。"
    "相对路径按项目根目录计算。Docker 部署时只能用项目内的路径，或在 docker-compose.yml 里把目标文件夹挂载进容器。"
    "更换后旧数据不会自动迁移，需要手动把原文件夹里的文件复制过去。",
    kind="path",
)
