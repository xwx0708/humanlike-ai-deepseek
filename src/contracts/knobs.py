"""参数注册表（《模块化改进建议》改进二，D42）。

每个模块在加载时自己登记"我有哪些参数"：参数类的字段用 knob() 补上名称、解释、范围；单个值、功能开关、
词表用 runtime.config 里的 value() / feature() / words() 顺带登记。设置页只是纯展示层：遍历注册表、
按分区渲染。新增、重命名、换掉一个模块，设置页一行都不用改。

参数的读写仍走 runtime.config（settings.json），这里只管说明。
"""

from dataclasses import dataclass, field, fields
from typing import Any

SECRET = "secret"  # 字段元信息里标 secret=True 的（API 密钥等）不登记、不上设置页


def knob(
    default: Any,
    label: str | None = None,
    help: str = "",
    *,
    lo: float | None = None,
    hi: float | None = None,
    step: float | None = None,
    unit: str = "",
    kind: str | None = None,
    options: list | None = None,
    option_labels: list[str] | None = None,
    secret: bool = False,
) -> Any:
    """参数类里一个字段：默认值照旧写在这里，附上设置页要的说明。kind 默认按默认值的类型推断。"""
    meta = {
        "label": label,
        "help": help,
        "min": lo,
        "max": hi,
        "step": step,
        "unit": unit,
        "kind": kind,
        "options": options,
        "option_labels": option_labels,
        SECRET: secret,
    }
    return field(default=default, metadata={k: v for k, v in meta.items() if v is not None})


@dataclass(frozen=True)
class KnobSpec:
    section: str
    key: str
    label: str
    help: str
    type: str  # int / float / bool / text / select / path
    default: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None
    unit: str = ""
    options: list | None = None
    option_labels: list[str] | None = None
    restart: bool = True  # 改了要重启才生效（目前全部如此）

    def as_dict(self) -> dict[str, Any]:
        d = {
            "key": self.key,
            "label": self.label,
            "help": self.help,
            "type": self.type,
            "default": self.default,
            "min": self.min,
            "max": self.max,
            "step": self.step,
            "unit": self.unit,
        }
        if self.options is not None:
            d["options"] = self.options
            if self.option_labels is not None:
                d["option_labels"] = self.option_labels
        return d


@dataclass
class SectionSpec:
    id: str
    title: str = ""
    help: str = ""
    fields: dict[str, KnobSpec] = field(default_factory=dict)  # 按登记顺序


@dataclass
class FeatureSpec:
    key: str
    label: str
    help: str


@dataclass
class WordListSpec:
    key: str
    label: str
    help: str
    default: list[str]


def default_step(kind: str) -> float | None:
    """数值类参数的默认步长；文字、开关、选项没有步长。"""
    return {"int": 1, "float": 0.01}.get(kind)


def kind_of(default: Any) -> str:
    if isinstance(default, bool):
        return "bool"
    if isinstance(default, int):
        return "int"
    if isinstance(default, float):
        return "float"
    return "text"


class KnobRegistry:
    def __init__(self):
        self.sections: dict[str, SectionSpec] = {}
        self.features: dict[str, FeatureSpec] = {}
        self.word_lists: dict[str, WordListSpec] = {}

    def section(self, id: str, title: str = "", help: str = "") -> SectionSpec:
        sec = self.sections.setdefault(id, SectionSpec(id))
        if title:
            sec.title, sec.help = title, help
        return sec

    def add(self, spec: KnobSpec) -> None:
        self.section(spec.section).fields[spec.key] = spec

    def add_value(self, section: str, key: str, default: Any, label: str, help: str = "", **meta) -> None:
        kind = meta.pop("kind", None) or kind_of(default)
        self.add(
            KnobSpec(
                section,
                key,
                label,
                help,
                kind,
                default,
                meta.pop("lo", None),
                meta.pop("hi", None),
                meta.pop("step", None) or default_step(kind),
                **meta,
            )
        )

    def add_dataclass(self, obj: Any, section: str) -> None:
        """登记一个参数类实例的全部字段：没写 label 的字段不上设置页（密钥等）。"""
        for f in fields(obj):
            m = dict(f.metadata)
            if m.get(SECRET) or not m.get("label"):
                continue
            m.pop(SECRET, None)
            default = getattr(obj, f.name)
            kind = m.get("kind") or kind_of(default)
            self.add(
                KnobSpec(
                    section,
                    f.name,
                    m["label"],
                    m.get("help", ""),
                    kind,
                    default,
                    m.get("min"),
                    m.get("max"),
                    m.get("step") or default_step(kind),
                    m.get("unit", ""),
                    m.get("options"),
                    m.get("option_labels"),
                )
            )

    def add_feature(self, key: str, label: str, help: str) -> None:
        self.features[key] = FeatureSpec(key, label, help)

    def add_words(self, key: str, label: str, help: str, default: tuple[str, ...]) -> None:
        self.word_lists[key] = WordListSpec(key, label, help, list(default))

    def schema(self, order: tuple[str, ...] = ()) -> dict[str, Any]:
        """设置页要的说明表。order 里的分区排前面，其余按登记顺序。"""
        ids = [i for i in order if i in self.sections] + [i for i in self.sections if i not in order]
        return {
            "sections": [
                {
                    "id": i,
                    "title": self.sections[i].title or i,
                    "help": self.sections[i].help,
                    "fields": [f.as_dict() for f in self.sections[i].fields.values()],
                }
                for i in ids
            ],
            "features": [vars(f) for f in self.features.values()],
            "word_lists": [vars(w) for w in self.word_lists.values()],
        }


REGISTRY = KnobRegistry()
