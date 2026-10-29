"""设置页的参数说明表（D40）：每个参数的名称、悬停解释、类型、范围、默认值。

默认值直接取自各模块代码里的参数类（不受 settings.json 影响），所以这里不重复写数值。
测试会检查：各参数类的每个字段在这里都有条目，保证"所有可修改的参数都能调"。
"""

from dataclasses import fields
from typing import Any

from action import qq_self
from llm.settings import Settings
from memory.boundaries import BoundaryRules
from memory.candidates import CandidateRules
from memory.clock import ClockKnobs
from memory.forgetting import ForgettingKnobs
from memory.fresh import FreshKnobs
from memory.profile import ConflictKnobs
from memory.resident import ResidentKnobs
from memory.runner import RunnerKnobs
from memory.scanner import ScannerKnobs
from memory.sweep import SweepKnobs
from response.requests import DEFAULT_WORDS


def _num(key, label, help, default, *, lo=None, hi=None, step=None, unit="", integer=False):
    return {"key": key, "label": label, "help": help, "type": "int" if integer else "float",
            "default": default, "min": lo, "max": hi, "step": step or (1 if integer else 0.01), "unit": unit}  # fmt: skip


def _from(obj, key: str, label: str, help: str, **kw) -> dict[str, Any]:
    default = getattr(obj, key)
    return _num(
        key, label, help, default, integer=isinstance(default, int) and not isinstance(default, bool), **kw
    )


F, C, S, FR = ForgettingKnobs(), ClockKnobs(), ScannerKnobs(), FreshKnobs()
R, SW, RU, B, P, CA = (
    ResidentKnobs(),
    SweepKnobs(),
    RunnerKnobs(),
    BoundaryRules(),
    ConflictKnobs(),
    CandidateRules(),
)
M = Settings()

SECTIONS: list[dict[str, Any]] = [
    {
        "id": "general",
        "title": "数据存放",
        "help": "记忆库（含档案库）、人设、生命周期状态、行动记录、图片都放在数据文件夹里。",
        "fields": [
            {
                "key": "data_dir",
                "label": "数据文件夹",
                "type": "path",
                "default": "data",
                "help": "记忆库 memory.db（概率海与档案库）、persona.md、lifecycle.json、action_log.json、images/ 的存放位置。"
                "相对路径按项目根目录计算。Docker 部署时只能用项目内的路径，或在 docker-compose.yml 里把目标文件夹挂载进容器。"
                "更换后旧数据不会自动迁移，需要手动把原文件夹里的文件复制过去。",
            },
        ],
    },
    {
        "id": "models",
        "title": "模型",
        "help": "双模型架构：思考模型负责当下的对话，潜意识模型在后台整理记忆。两者都可以是本地模型或云端 API。API 密钥只写在 .env，不在这里填写。",
        "fields": [
            {
                "key": "thinking_context_limit",
                "label": "思考模型上下文上限",
                "type": "int",
                "default": M.thinking_context_limit,
                "min": 4000,
                "max": 4000000,
                "step": 1000,
                "unit": "token",
                "help": "思考模型的最大上下文长度（token）。熟悉期用到这个值的启动比例时，记忆系统启动。Docker 部署时思考模型在 AstrBot 网页里选，换模型时这里要同步修改。",
            },
            {
                "key": "subconscious_provider",
                "label": "潜意识模型提供方",
                "type": "select",
                "default": M.subconscious_provider,
                "options": ["ollama", "openai_compat", "deepseek"],
                "help": "ollama：本地运行（推荐，记忆整理不出本机，需要显卡）；openai_compat：任意 OpenAI 兼容的云端或本地服务；deepseek：DeepSeek API。",
            },
            {
                "key": "subconscious_model",
                "label": "潜意识模型",
                "type": "text",
                "default": M.subconscious_model,
                "help": "模型名。默认 Qwen 27B Q4 量化约需 17 GB 显存（如 RTX 3090/4090）；显存不足可换 7B～14B 的模型，整理质量会下降。",
            },
            {
                "key": "ollama_base_url",
                "label": "Ollama 地址",
                "type": "text",
                "default": M.ollama_base_url,
                "help": "本机 Ollama 服务地址。Docker 部署时由 docker-compose.yml 改写为 host.docker.internal。",
            },
            {
                "key": "ollama_num_ctx",
                "label": "Ollama 上下文长度",
                "type": "int",
                "default": M.ollama_num_ctx,
                "min": 2048,
                "max": 131072,
                "step": 1024,
                "unit": "token",
                "help": "潜意识模型一次能看的长度。越大越占显存；切段的每批长度（整理 → 切段批量）要明显小于它。",
            },
            {
                "key": "ollama_keep_alive",
                "label": "Ollama 保持加载",
                "type": "text",
                "default": M.ollama_keep_alive,
                "help": "空闲多久后卸载模型，如 30m、2h。首次加载大模型约需 1～2 分钟。",
            },
            {
                "key": "openai_compat_base_url",
                "label": "OpenAI 兼容接口地址",
                "type": "text",
                "default": M.openai_compat_base_url,
                "help": "提供方选 openai_compat 时使用，例如 https://api.deepseek.com、http://127.0.0.1:8000/v1。密钥写在 .env 的 OPENAI_COMPAT_API_KEY。",
            },
            {
                "key": "deepseek_thinking",
                "label": "DeepSeek 思考模式",
                "type": "bool",
                "default": M.deepseek_thinking,
                "help": "开启后先推理再回答，更准但更慢、更费 token。只对本项目直接调用的 DeepSeek 生效。",
            },
        ],
    },
    {
        "id": "lifecycle",
        "title": "启动机制",
        "help": "熟悉期 → 启动 → 运行期。熟悉期按普通方式带全部聊天记录；用量达到启动比例时，一次性整理全部对话、建立概率海。",
        "fields": [
            _num(
                "start_ratio",
                "启动比例",
                "上下文用量占上限的比例达到此值时启动记忆系统（D24）。启动只发生一次。",
                0.7,
                lo=0.1,
                hi=0.95,
                step=0.05,
            ),
        ],
    },
    {
        "id": "forgetting",
        "title": "遗忘曲线",
        "help": "记忆强度 S(t) = 0.5^(Δt / H)。半衰期 H = 基础半衰期 × 情绪倍数^a × 重要性倍数^i × 首尾加成，a 为相对唤醒度、i 为重要性（0～1）。",
        "fields": [
            _from(
                F,
                "base_half_life",
                "基础半衰期",
                "平淡小事（a = i = 0）的半衰期。",
                lo=0.5,
                hi=60,
                step=0.5,
                unit="天",
            ),
            _from(
                F,
                "emotion_max",
                "情绪倍数",
                "相对唤醒度为 1 时半衰期的放大倍数，按指数插值。",
                lo=1,
                hi=50,
                step=0.5,
            ),
            _from(
                F,
                "importance_max",
                "重要性倍数",
                "重要性为 1 时半衰期的放大倍数，按指数插值。",
                lo=1,
                hi=50,
                step=0.5,
            ),
            _from(
                F,
                "position_bonus",
                "首尾加成",
                "一场对话首个、末个片段的半衰期倍数（首因、近因效应）。",
                lo=1,
                hi=3,
                step=0.1,
            ),
            _from(
                F,
                "reinforce",
                "巩固强度",
                "提取巩固：被真正用到时 S 回满，H × (1 + 巩固强度 × (1 − S))。越接近遗忘时被想起，巩固越多。",
                lo=0,
                hi=5,
                step=0.1,
            ),
            _from(
                F,
                "forget_line",
                "遗忘阈值",
                "S 低于此值即视为遗忘：退出扫描采样，进入待清理队列。",
                lo=0.01,
                hi=0.5,
                step=0.01,
            ),
            _from(
                F,
                "adjust_step_max",
                "单次调参上限",
                "潜意识模型对一条记忆单次调整半衰期的最大倍数，下限为其倒数（护栏）。",
                lo=1,
                hi=10,
                step=0.5,
                unit="倍",
            ),
            _from(
                F,
                "adjust_total_max",
                "累积调参上限",
                "一条记忆累积调整的最大倍数，下限为其倒数（护栏）。",
                lo=1,
                hi=20,
                step=0.5,
                unit="倍",
            ),
        ],
    },
    {
        "id": "clock",
        "title": "记忆时钟",
        "help": "衰减按对话时间计，不按日历时间：对话时流速加快，停顿后减速，再久则停止。长时间不使用不会导致遗忘一切（D21）。",
        "fields": [
            _from(
                C,
                "speed",
                "对话流速",
                "对话期间记忆时间相对现实时间的倍数。",
                lo=1,
                hi=100,
                step=1,
                unit="倍",
            ),
            _from(
                C,
                "idle_after",
                "减速延迟",
                "最后一条消息后多久开始减速。",
                lo=60,
                hi=7200,
                step=60,
                unit="秒",
            ),
            _from(C, "idle_rate", "减速后流速", "减速后相对对话流速的比例。", lo=0, hi=1, step=0.01),
            _from(
                C,
                "freeze_after",
                "停止延迟",
                "减速多久后记忆时间完全停止。",
                lo=600,
                hi=30 * 86400,
                step=600,
                unit="秒",
            ),
        ],
    },
    {
        "id": "scanner",
        "title": "扫描与浮现",
        "help": "扫描器按月份分区轮询，每分钟一个分区。对每个片段做伯努利采样：P = 1 − e^(−p·Δt)，p = 速率上限 · σ((S + 启动加成 − 阈值) / T)。命中者进入浮现区。",
        "fields": [
            _from(
                S,
                "buffer_size",
                "浮现区容量",
                "浮现区可同时容纳的片段数，FIFO 置换。浮现本身不触发巩固。",
                lo=1,
                hi=12,
            ),
            _from(
                S,
                "threshold",
                "浮现阈值",
                "sigmoid 的中心：S + 启动加成等于它时，p 为速率上限的一半。",
                lo=0,
                hi=1.5,
                step=0.05,
            ),
            _from(
                S,
                "rate",
                "速率上限",
                "每分钟浮现强度的上限。0.02 时最强、最相关的记忆约每 50 分钟浮现一次。",
                lo=0.001,
                hi=0.5,
                step=0.001,
            ),
            _from(
                S,
                "temp_focus",
                "专注温度 T",
                "专注于同一话题时的温度：低温使采样更确定，强记忆更集中地浮现。",
                lo=0.01,
                hi=1,
                step=0.01,
            ),
            _from(
                S,
                "temp_diffuse",
                "发散温度 T",
                "闲聊时的温度：高温使采样更随机，弱记忆也有机会浮现。",
                lo=0.01,
                hi=1,
                step=0.01,
            ),
            _from(
                S,
                "focus_turns",
                "专注判定",
                "同一话题连续多少条用户消息算完全专注（温度降到专注温度）。",
                lo=1,
                hi=30,
            ),
            _from(
                S,
                "priming_max",
                "启动加成上限",
                "启动效应（priming）：与当前话题关键词相关的片段获得的最大加成。",
                lo=0,
                hi=1.5,
                step=0.05,
            ),
            _from(
                S,
                "person_bonus",
                "人物加成",
                "片段涉及的人物与当前话题重合时的加成。",
                lo=0,
                hi=1.5,
                step=0.05,
            ),
            _from(
                S,
                "negative_damp",
                "负面抑制",
                "负面片段在话题无关时，浮现概率最多下降的比例。",
                lo=0,
                hi=1,
                step=0.05,
            ),
        ],
    },
    {
        "id": "fresh",
        "title": "新鲜记忆",
        "help": "运行期不另带聊天记录：刚说过的话作为几乎未衰减的记忆在场（D30、D31）。",
        "fields": [
            _from(
                FR,
                "floor",
                "在场下限",
                "强度不低于此值的消息稳定在场，不经采样。",
                lo=0.5,
                hi=0.99,
                step=0.01,
            ),
            _from(
                FR,
                "half_life",
                "新鲜记忆半衰期",
                "尚未打分的消息按此半衰期衰减。",
                lo=0.5,
                hi=60,
                step=0.5,
                unit="天",
            ),
            _from(
                FR,
                "split_at",
                "中途整理触发",
                "最早一条未整理消息的强度降到此值时，对话中途先整理一批。须高于在场下限。",
                lo=0.5,
                hi=0.999,
                step=0.005,
            ),
            _from(
                FR,
                "split_upto",
                "中途整理范围",
                "这一批整理到强度低于此值的消息为止，更新的留到下一批。须高于触发值。",
                lo=0.5,
                hi=0.999,
                step=0.005,
            ),
        ],
    },
    {
        "id": "resident",
        "title": "常驻栏",
        "help": "免竞争项（画像出处、用户要求记住、多次提及）之外，其余事件线按加权得分竞争席位。常驻期间不衰减、不参与扫描。",
        "fields": [
            _from(R, "seats", "竞争席位", "参与竞争的常驻席位数。", lo=0, hi=100),
            _from(
                R,
                "w_importance",
                "权重：重要性",
                "得分 = Σ 权重 × 分项。四项权重之和建议为 1。",
                lo=0,
                hi=1,
                step=0.05,
            ),
            _from(R, "w_emotion", "权重：情绪强度", "同上。", lo=0, hi=1, step=0.05),
            _from(
                R,
                "w_activity",
                "权重：活跃度",
                "同上。活跃度按最近一次提及的时间衰减。",
                lo=0,
                hi=1,
                step=0.05,
            ),
            _from(
                R,
                "w_recall",
                "权重：回忆频率",
                "同上。回忆频率 = 次数 / (次数 + 饱和常数)。",
                lo=0,
                hi=1,
                step=0.05,
            ),
            _from(
                R, "ongoing_factor", "进行中系数", "尚未完结的事件线得分乘以此系数。", lo=1, hi=5, step=0.1
            ),
            _from(
                R,
                "activity_half_life",
                "活跃度半衰期",
                "多久未被提及，活跃度减半。",
                lo=1,
                hi=180,
                step=1,
                unit="天",
            ),
            _from(
                R,
                "recall_saturation",
                "回忆饱和常数",
                "回忆频率 = 次数 / (次数 + 此值)。",
                lo=0.5,
                hi=20,
                step=0.5,
            ),
            _from(
                R,
                "repeated_conversations",
                "多次提及判定",
                '在至少多少场不同对话中被提及，算免竞争的"多次提及"。',
                lo=2,
                hi=20,
            ),
        ],
    },
    {
        "id": "sweep",
        "title": "批量遗忘与档案库",
        "help": "跌破遗忘阈值的片段进入待清理队列，累计满一批处理：大部分移入档案库（原文保留，可检索），按比例随机保留少数做关键词抽象，重新开始衰减。",
        "fields": [
            _from(SW, "batch_size", "批大小", "累计多少个待清理片段触发一次批处理。", lo=1, hi=200),
            _from(
                SW,
                "keep_ratio",
                "保留比例",
                "每批随机保留做关键词抽象的比例（至少 1 个）。",
                lo=0,
                hi=0.5,
                step=0.01,
            ),
            _from(
                SW,
                "excerpt_chars",
                "抽象原文节选",
                "做关键词抽象时给潜意识模型看的原文长度。",
                lo=50,
                hi=2000,
                step=50,
                unit="字",
            ),
        ],
    },
    {
        "id": "archive",
        "title": "档案库检索",
        "help": "档案库检索：jieba 分词 + BM25 排序，在已归档片段的标题和原文中检索。只在用户要求时执行（D26）。",
        "fields": [
            _num("limit", "返回条数", "一次检索返回的最多结果数。", 5, lo=1, hi=50, integer=True),
            _num(
                "excerpt_chars",
                "结果节选长度",
                "每条结果给思考模型看的原文长度。",
                200,
                lo=50,
                hi=5000,
                step=50,
                unit="字",
                integer=True,
            ),
        ],
    },
    {
        "id": "runner",
        "title": "潜意识整理",
        "help": "对话停顿后由潜意识模型整理：切段、打分、挂事件线、调参；夜间合并重复的事件线。",
        "fields": [
            _from(RU, "pause", "停顿判定", "对话停顿多久后开始整理。", lo=60, hi=7200, step=60, unit="秒"),
            _from(RU, "night_hour", "夜间合并时刻", "每天几点后执行一次事件线合并。", lo=0, hi=23),
            _num(
                "segment_budget",
                "切段批量",
                "切段时每批消息的最大 token 数，须明显小于潜意识模型的上下文长度。",
                6000,
                lo=500,
                hi=100000,
                step=500,
                unit="token",
                integer=True,
            ),
            _num(
                "thread_confidence_min",
                "挂线置信阈值",
                "潜意识模型判断片段属于某事件线的把握低于此值时，新建事件线。",
                0.6,
                lo=0,
                hi=1,
                step=0.05,
            ),
        ],
    },
    {
        "id": "merge",
        "title": "事件线合并",
        "help": "夜间检查近期新建的事件线，与已有事件线重复时合并。",
        "fields": [
            _num(
                "recent_days",
                "近期范围",
                "多少天内新建的事件线参与合并检查。",
                7.0,
                lo=1,
                hi=60,
                step=1,
                unit="天",
            ),
            _num(
                "confidence_min",
                "合并置信阈值",
                "把握不低于此值才合并。宜高于挂线阈值：错误合并比错误新建更难发现。",
                0.8,
                lo=0,
                hi=1,
                step=0.05,
            ),
        ],
    },
    {
        "id": "candidates",
        "title": "事件线候选",
        "help": "给潜意识模型判断归属时提供的候选事件线：最近活跃 ∪ 关键词重合 ∪ 人物重合等来源的并集。",
        "fields": [
            _from(CA, "max_total", "候选总数", "一次最多提供的候选事件线数。", lo=1, hi=100),
            _from(CA, "recent", "最近活跃", "从最近活跃的事件线中取几条。", lo=0, hi=50),
            _from(CA, "keyword", "关键词重合", "按 BM25 关键词重合取几条。", lo=0, hi=50),
            _from(
                CA,
                "min_alias_len",
                "人名最短长度",
                "按人名匹配时，称呼至少几个字（过短易误匹配）。",
                lo=1,
                hi=5,
            ),
        ],
    },
    {
        "id": "boundaries",
        "title": "切段边界",
        "help": "切段第一层规则：按时间间隔划分对话，并在过长处强制标记候选边界。",
        "fields": [
            _from(
                B,
                "conversation_gap_minutes",
                "对话间隔",
                "超过这么久没有消息，视为新的一场对话。",
                lo=1,
                hi=1440,
                step=1,
                unit="分钟",
            ),
            _from(B, "max_turns", "最长条数", "距上一个候选边界满这么多条消息，强制标记一次。", lo=2, hi=500),
            _from(
                B,
                "max_minutes",
                "最长时长",
                "距上一个候选边界满这么久，强制标记一次。",
                lo=1,
                hi=1440,
                step=1,
                unit="分钟",
            ),
        ],
    },
    {
        "id": "profile",
        "title": "画像冲突",
        "help": "新事实与画像中的旧事实矛盾时，按旧事实的牢固度掷骰子：唤起 / 覆盖 / 都不（D16）。",
        "fields": [
            _from(
                P,
                "firm_half_life",
                "牢固度半衰期",
                "旧事实多久未被提及，牢固度减半。",
                lo=1,
                hi=3650,
                step=1,
                unit="天",
            ),
            _from(P, "evoke_min", "最低唤起概率", "最不牢固的旧事实被唤起的概率。", lo=0, hi=1, step=0.05),
            _from(P, "evoke_max", "最高唤起概率", "最牢固的旧事实被唤起的概率。", lo=0, hi=1, step=0.05),
            _from(
                P,
                "overwrite_max",
                "最高覆盖概率",
                "最不牢固的旧事实被覆盖（移入档案）的概率。",
                lo=0,
                hi=1,
                step=0.05,
            ),
        ],
    },
    {
        "id": "attachments",
        "title": "文件与图片",
        "help": "图片交给思考模型直接识别；文件提取文字放入当轮提示词，记忆中只保留摘要（D38）。",
        "fields": [
            _num("max_mb", "单文件上限", "超过此大小的文件不读取。", 20.0, lo=1, hi=200, step=1, unit="MB"),
            _num(
                "prompt_chars",
                "提示词字数上限",
                "一个文件最多放进提示词的字数，超出截断。",
                20000,
                lo=1000,
                hi=500000,
                step=1000,
                unit="字",
                integer=True,
            ),
            _num(
                "memory_chars",
                "记忆摘录长度",
                "记入记忆的文件开头长度。",
                2000,
                lo=0,
                hi=20000,
                step=100,
                unit="字",
                integer=True,
            ),
        ],
    },
    {
        "id": "action",
        "title": "自主行为频率",
        "help": "小d 在严格限频下自主执行的 QQ 操作（D35）。上限 = 在给定天数内最多执行的次数。",
        "fields": [
            *[
                f
                for kind, label, lim in [
                    ("avatar", "换头像", qq_self.LIMITS[qq_self.AVATAR]),
                    ("nickname", "改昵称", qq_self.LIMITS[qq_self.NICKNAME]),
                    ("signature", "改签名", qq_self.LIMITS[qq_self.SIGNATURE]),
                    ("post", "发说说", qq_self.LIMITS[qq_self.POST]),
                    ("delete", "删说说", qq_self.LIMITS[qq_self.DELETE]),
                ]
                for f in (
                    _num(
                        f"{kind}_times",
                        f"{label}：次数",
                        f"{label}在窗口期内最多执行的次数。",
                        1 if kind != "delete" else 2,
                        lo=0,
                        hi=50,
                        integer=True,
                    ),
                    _num(
                        f"{kind}_days",
                        f"{label}：窗口期",
                        f"{label}的频率窗口长度。",
                        {"avatar": 7.0, "nickname": 7.0, "signature": 3.0}.get(kind, 1.0),
                        lo=0.1,
                        hi=90,
                        step=0.5,
                        unit="天",
                    ),
                )
            ],
            {
                "key": "post_visibility",
                "label": "说说可见范围",
                "type": "select",
                "default": 4,
                "options": [1, 4, 64],
                "option_labels": ["所有人", "好友", "仅自己"],
                "help": "自主发布的 QQ 空间说说的可见范围。",
            },
        ],
    },
]

FEATURES: list[dict[str, Any]] = [
    {
        "key": "keyword_requests",
        "label": "关键词识别用户要求",
        "help": '识别"记住、忘掉、翻档案"等要求（D29）。关闭后这些话按普通对话处理。',
    },
    {
        "key": "persona_reminder",
        "label": "人设提醒",
        "help": "每轮附加一句人设提醒，避免口吻被历史记录带偏。",
    },
    {
        "key": "mid_chat_split",
        "label": "对话中途分批整理",
        "help": "长时间不停顿的对话中，在新鲜记忆失效前先整理一批（D31）。",
    },
    {"key": "nightly_merge", "label": "夜间合并事件线", "help": "每天夜间合并重复的事件线。"},
    {
        "key": "batch_sweep",
        "label": "批量遗忘",
        "help": "关闭后跌破遗忘阈值的片段只停止参与采样，不移入档案库。",
    },
    {
        "key": "read_images",
        "label": "识别图片",
        "help": "把用户发来的图片交给思考模型（模型需支持图像输入）。",
    },
    {
        "key": "read_files",
        "label": "读取文件",
        "help": "提取用户发来的文件内容（PDF、Word、Excel、PPT、文本、代码等）。",
    },
    {
        "key": "web_search",
        "label": "联网搜索",
        "help": "使用 AstrBot 自带的网页搜索（D36），来源和密钥在 AstrBot 中配置。",
    },
    {
        "key": "qq_set_avatar",
        "label": "自主换头像",
        "help": "允许小d 自主更换 QQ 头像（图片来自数据文件夹的 images/）。",
    },
    {"key": "qq_set_nickname", "label": "自主改昵称", "help": "允许小d 自主修改 QQ 昵称。"},
    {"key": "qq_set_signature", "label": "自主改签名", "help": "允许小d 自主修改 QQ 个性签名。"},
    {"key": "qq_post_moment", "label": "自主发说说", "help": "允许小d 自主发布 QQ 空间说说。"},
    {"key": "qq_delete_moment", "label": "自主删说说", "help": "允许小d 删除自己发布的说说。"},
]

WORD_LISTS: list[dict[str, Any]] = [
    {
        "key": "remember",
        "label": "要求记住",
        "help": '用户消息包含其中任一词时，所在片段标记为"用户要求记住"（常驻免竞争）。',
    },
    {
        "key": "forget",
        "label": "要求忘掉",
        "help": "触发遗忘流程：先确认是哪件事，再由用户选择彻底删除或存入档案库（D25）。",
    },
    {"key": "archive", "label": "要求翻档案", "help": "触发档案库检索（D26：仅在用户要求时检索）。"},
    {"key": "choice_erase", "label": "选择：彻底删除", "help": "遗忘流程中，表示选择彻底删除的说法。"},
    {"key": "choice_archive", "label": "选择：存入档案", "help": "遗忘流程中，表示选择存入档案库的说法。"},
    {"key": "choice_cancel", "label": "选择：取消", "help": "遗忘流程中，表示取消的说法。"},
]
for w in WORD_LISTS:
    w["default"] = list(DEFAULT_WORDS[w["key"]])

# 数值类参数类里每个字段都要在上面出现（测试检查）
KNOB_CLASSES = {
    "forgetting": ForgettingKnobs, "clock": ClockKnobs, "scanner": ScannerKnobs, "fresh": FreshKnobs,
    "resident": ResidentKnobs, "sweep": SweepKnobs, "runner": RunnerKnobs, "boundaries": BoundaryRules,
    "profile": ConflictKnobs, "candidates": CandidateRules,
}  # fmt: skip


def schema() -> dict[str, Any]:
    return {"sections": SECTIONS, "features": FEATURES, "word_lists": WORD_LISTS}


def class_fields(cls) -> list[str]:
    return [f.name for f in fields(cls)]
