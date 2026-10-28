"""记忆相关的数据结构：一场对话、消息、片段、事件线、人物。记忆模块和情绪模块共用。

三层结构的含义见 00-总纲/术语表.md 和 01-记忆/事件切分.md。
时间统一存 Unix 时间戳（秒）；遗忘曲线按"天"计算，换算用 days()。
"""

from dataclasses import dataclass, field

SECONDS_PER_DAY = 86400.0


def days(ts: float) -> float:
    return ts / SECONDS_PER_DAY


@dataclass
class Conversation:
    """一场对话：一次连续的聊天，以时间间隔为界。"""

    id: int | None
    start_ts: float
    end_ts: float
    summary: str = ""


@dataclass
class ChatTurn:
    """一条消息。原文永久保存，档案就是它。"""

    id: int | None
    conversation_id: int
    role: str  # "user" | "assistant"
    content: str
    ts: float


@dataclass
class EmotionScore:
    """对一个片段的正式情绪评分（峰终定律）。"""

    arousal: float  # 唤醒度 0～1
    valence: float  # 效价 -1～1
    importance: float  # 重要性 0～1
    reason: str = ""


@dataclass
class Segment:
    """片段：对话中的一个子话题，能用一句话概括；是打分和遗忘的基本单位。"""

    id: int | None
    conversation_id: int
    title: str
    start_ts: float
    end_ts: float
    message_ids: list[int] = field(default_factory=list)  # 一条消息可以同时属于两个片段
    person_ids: list[int] = field(default_factory=list)
    emotion: EmotionScore | None = None
    # 遗忘曲线的状态，见 01-记忆/遗忘曲线.md
    half_life: float = 7.0  # 天
    last_recall_ts: float = 0.0  # 最近一次记下或被真正用到的时间
    recall_count: int = 0
    adjustment: float = 1.0  # 潜意识对本条记忆的累积调整倍数
    at_edge: bool = False  # 是否为一场对话的首个或末个片段
    state: str = "active"  # active：在概率海里；forgotten：已遗忘，只剩残影
    trace: str = ""  # 残影：遗忘后留下的极简概括


@dataclass
class Thread:
    """事件线：跨天的同一件事，由多个片段挂在一起。"""

    id: int | None
    title: str
    summary: str
    created_ts: float
    updated_ts: float
    segment_ids: list[int] = field(default_factory=list)
    person_ids: list[int] = field(default_factory=list)
    merged_into: int | None = None  # 夜间合并后指向保留的那条


PROFILE_CATEGORIES = (
    "过敏与禁忌",
    "长期健康",
    "家人与关系",
    "生日与纪念日",
    "职业与学业",
    "居住地",
    "长期偏好",
    "其他",
)


@dataclass
class ProfileFact:
    """用户画像中的一条稳定事实。只能合并、不能改写，也不参与遗忘（设想文档"遗忘之后"一节）。"""

    id: int | None
    person_id: int | None  # 说的是谁；None 表示用户本人
    category: str  # PROFILE_CATEGORIES 之一
    fact: str  # 原样保存，之后不再改写
    first_ts: float
    last_ts: float  # 最近一次被提到
    mentions: int = 1
    segment_ids: list[int] = field(default_factory=list)  # 出处


@dataclass
class Person:
    """人物别名表中的一个人：妈妈 / 我妈 / 老妈 / 母亲 → 同一个 id。"""

    id: int | None
    name: str  # 规范名
    aliases: list[str] = field(default_factory=list)
