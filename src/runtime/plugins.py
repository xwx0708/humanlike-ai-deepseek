"""用户自定义插件的载入（改进三，D42）：数据文件夹的 plugins/ 下的 Python 文件，启动时按名字载入。

文件里定义一个类，类上写 plugin_kind（"decay" 遗忘曲线 / "surface" 浮现策略，见 contracts.plugins）和 name。
不做配置文件里的 DSL：会用 AI 写代码的用户，Python 文件最自然。
一个文件坏了只跳过它（记在 errors 里），不影响启动。
"""

import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any

from runtime import config

KINDS = ("decay", "surface")
_loaded: dict[str, dict[str, type]] | None = None
errors: list[tuple[str, str]] = []  # (文件, 错误)


def plugin_dir() -> Path:
    return config.data_dir() / "plugins"


def load_plugins(folder: Path | None = None) -> dict[str, dict[str, type]]:
    """载入一次并缓存：{种类: {名字: 类}}。folder 给了就重新扫描（测试用）。"""
    global _loaded
    if _loaded is not None and folder is None:
        return _loaded
    found: dict[str, dict[str, type]] = {k: {} for k in KINDS}
    errors.clear()
    folder = folder or plugin_dir()
    for path in sorted(folder.glob("*.py")) if folder.is_dir() else []:
        try:
            spec = importlib.util.spec_from_file_location(f"humanlike_plugin_{path.stem}", path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
        except Exception as e:  # noqa: BLE001  用户的文件坏了不能让系统起不来
            errors.append((path.name, f"{type(e).__name__}: {e}"))
            continue
        for _, cls in inspect.getmembers(module, inspect.isclass):
            kind, name = getattr(cls, "plugin_kind", None), getattr(cls, "name", None)
            if kind in found and isinstance(name, str) and name:
                found[kind][name] = cls
    _loaded = found
    return found


def names(kind: str, builtins: dict[str, Any]) -> list[str]:
    """设置页的选项：内置的在前，插件的在后。"""
    return list(builtins) + [n for n in load_plugins()[kind] if n not in builtins]


def choose(kind: str, name: str, builtins: dict[str, Any]) -> Any:
    """按名字取实现：先内置，再插件（实例化）；都没有就用内置的第一个，并记一条错误。"""
    if name in builtins:
        return builtins[name]
    cls = load_plugins()[kind].get(name)
    if cls is not None:
        return cls()
    errors.append(("settings", f"没有叫「{name}」的{kind}实现，改用默认"))
    return next(iter(builtins.values()))
