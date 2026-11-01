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
    # active：在记忆里；archived：批量遗忘时移进档案，不再出现在记忆系统里；
    # abstract：批量遗忘时被留下，只剩关键词，重新开始一条遗忘曲线（决策 D22）
    state: str = "active"
    trace: str = ""  # state 为 abstract 时的关键词，用"、"隔开
    remember_requested: bool = False  # 用户明确要求记住（所在事件线免竞争常驻，D17）


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
    ongoing: bool = True  # 这件事是否还在进行（潜意识模型更新摘要时判断）；进行中的常驻得分乘系数（D17）
    resident_since: float | None = None  # 进入常驻栏的时间；None 表示在普通记忆池（D17）


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
    archived_ts: float | None = None  # 被新说法覆盖、移进档案的时间（决策 D16）；None 表示仍在画像里
    superseded_by: int | None = None  # 覆盖它的那条事实


@dataclass
class Person:
    """人物别名表中的一个人：妈妈 / 我妈 / 老妈 / 母亲 → 同一个 id。"""

    id: int | None
    name: str  # 规范名
    aliases: list[str] = field(default_factory=list)


# ---------- 记忆服务（《模块化改进建议》改进一）：回应模块只认这个接口 ----------
# 方法按回应模块今天真在用的调用定，不为将来预留；新增方法要在决策日志里记一条。

from typing import Protocol

FORGET_ARCHIVE = "存进档案"  # 用户要求忘掉时的两种方式（D25：由用户选，不替用户默认）
FORGET_ERASE = "彻底删除"


@dataclass
class MemoryContext:
    """每轮交给思考模型的记忆内容，已经排好版；回应模块拿到就能拼提示词，不必知道怎么算出来的。

    熟悉期只有 history（全部旧消息，D24）；运行期只有后面几段文字（D29、D30）。"""

    history: list[ChatTurn] = field(default_factory=list)  # 熟悉期：全部旧消息，不含这一条
    resident: str = ""  # 常驻栏（D17）
    surfaced: str = ""  # 浮现区：自己冒头的记忆 + 被唤起的旧事实（D16、D26）
    fresh: str = ""  # 新鲜记忆：刚说过的话（D30）
    surfaced_ids: list[int] = field(default_factory=list)  # 浮现区里各片段的 id


@dataclass
class ForgetTarget:
    """用户要求忘掉时，AI 理解为哪条事件线；等用户选方式时记着它。"""

    thread_id: int
    title: str


@dataclass
class ArchiveHit:
    """档案搜索的一条结果（只在用户要求翻档案时产生，D26）。"""

    segment_id: int
    title: str
    start_ts: float
    score: float
    excerpt: str


class MemoryService(Protocol):
    """记忆系统对回应模块的门。默认实现是概率海（memory.DefaultMemoryService），
    对照组是只留最近 N 条的 memory.RecentWindowMemory。"""

    def ingest(self, role: str, content: str, ts: float) -> ChatTurn:
        """存入一条消息（对方说的、自己说的、自己做的事），返回存好的消息。"""

    def context_for_turn(self, turn: ChatTurn, ts: float) -> MemoryContext:
        """这一轮交给思考模型的记忆内容。turn 是刚 ingest 的这条消息。"""

    def scan(self, ts: float) -> None:
        """扫描器走一格（运行期每分钟一次，D26）。"""

    def phase(self) -> str:
        """runtime.lifecycle 的 FAMILIAR / RUNNING。"""

    def report_context_used(self, prompt_tokens: int) -> None:
        """回报思考模型这一轮实际用掉的上下文（D24）。"""

    def mark_remember(self, turn_id: int) -> None:
        """用户要求记住这条消息（D17 免竞争）。"""

    def find_forget_target(self, query: str, ts: float) -> ForgetTarget | None:
        """用户要求忘掉：按描述找最相关的事件线，找不到返回 None。"""

    def pending_forget(self) -> ForgetTarget | None:
        """上一轮问过、还等用户选方式的那件事。"""

    def set_pending_forget(self, target: ForgetTarget | None) -> None: ...

    def forget(self, thread_id: int, mode: str, ts: float) -> str:
        """按用户选的方式（FORGET_ARCHIVE / FORGET_ERASE）忘掉，返回给用户看的一句说明。"""

    def search_archive(self, query: str) -> list[ArchiveHit]:
        """只在用户要求时翻档案（D26）。"""

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        """小块键值状态（例如平台会话标识 owner_umo）。"""

    def set_meta(self, key: str, value: str) -> None: ...

    def close(self) -> None: ...
