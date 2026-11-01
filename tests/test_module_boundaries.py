"""模块边界检查（《模块化改进建议》第 0 步；代码规范"模块之间的规矩"第 1 条）。

规则：模块之间只通过 src/contracts/ 往来；一个模块的子文件（memory.store、memory.scanner 这类）
只允许本模块内部导入。导入另一个模块的包本身（from llm import FakeModel）不算越界，
contracts/ 和 runtime/（时钟、配置、调度，各模块共用）不算模块。

新加一处越界导入，这个测试会直接失败。（曾有白名单记录已知违反，2026-11-01 全部消除后删掉了。）
"""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
SHARED = {"contracts", "runtime"}  # 不算模块的共用层


def _modules() -> set[str]:
    return {p.name for p in SRC.iterdir() if p.is_dir() and not p.name.startswith("_")} - SHARED


def _imports(path: Path) -> set[str]:
    """文件里（含函数内部）导入的全部点分名字：import a.b 与 from a.b import c 都记作 a.b。"""
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    return names


def _violations() -> dict[str, set[str]]:
    modules = _modules()
    found: dict[str, set[str]] = {}
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        own = rel.split("/")[0]
        if own in SHARED:
            continue
        for name in _imports(path):
            top = name.split(".")[0]
            if top in modules and top != own and "." in name:  # 别的模块的子文件
                found.setdefault(rel, set()).add(name)
    return found


def test_modules_only_meet_through_contracts():
    found = _violations()
    assert not found, f"越界导入（模块之间只能通过 contracts 往来）：{found}"
