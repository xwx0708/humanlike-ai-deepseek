"""长期对话对照评估的剧本（项目总评第三步，用户 2026-11-08 同意：3 周、约 20 个问题）。

- 要记住的事（FACTS）和探针问题（PROBES）都是手写的，评分标准一目了然。
- 每天围绕要埋的事、用 DeepSeek 生成一场自然的闲聊（双方的话都生成好），存进 eval/data/script.json，
  之后每次评估都用同一份剧本，结果可比。重新生成：python eval/scenario.py --regen
- 模拟用户：林舟，26 岁，在上海做程序员。
"""

import argparse
import json
import sys
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "eval" / "data" / "script.json"
DAYS = 21
USER = "林舟，26 岁，在上海做后端程序员，性格有点宅，说话随意、爱用口语。"


@dataclass(frozen=True)
class Fact:
    day: int  # 第几天聊到（从 1 开始）
    text: str  # 用户这天要自然地说出来的事
    weight: str  # 平淡 / 重要 / 动情：看动情、重要的是不是记得更久


FACTS = [
    Fact(1, "养了一只橘猫，叫馒头", "平淡"),
    Fact(1, "对花生过敏，吃到会起疹子、喘不上气", "重要"),
    Fact(2, "妈妈因为肺炎住院了，很担心，晚上睡不着", "动情"),
    Fact(3, "每天早上都要喝一杯冰美式", "平淡"),
    Fact(4, "下周三要去字节跳动面试，有点紧张", "重要"),
    Fact(5, "有个妹妹叫小雨，在读大三，学设计", "平淡"),
    Fact(6, "最近开始学吉他，手指按弦很疼", "平淡"),
    Fact(8, "妈妈出院了，松了一口气", "动情"),
    Fact(9, "今天去字节面试了，感觉发挥一般", "重要"),
    Fact(10, "从徐汇搬家到了浦东，通勤变远了", "重要"),
    Fact(11, "特别讨厌香菜，点外卖都要备注不要香菜", "平淡"),
    Fact(12, "字节的面试通过了，拿到了 offer，下个月入职", "动情"),
    Fact(14, "和最好的朋友老周因为借钱的事吵架了，很难受", "动情"),
    Fact(15, "馒头这两天不吃东西，带它去了宠物医院，说是肠胃炎", "重要"),
    Fact(16, "为了睡眠，决定把咖啡戒了，改喝茶", "平淡"),
    Fact(18, "和老周和好了，他主动道歉了", "动情"),
    Fact(19, "生日是 12 月 3 日，快到了", "重要"),
]


@dataclass(frozen=True)
class Probe:
    day: int  # 在这天的对话最后问
    question: str
    expect: str  # 给评分用：正确答案要点；never 题写"从没说过"
    kind: str  # 回忆 / 改口（要答最新的）/ 从没说过（不该瞎编）
    about: list[int] = field(default_factory=list)  # 对应 FACTS 的下标，算"隔了几天"


PROBES = [
    Probe(7, "诶我家猫叫啥来着，你还记得不", "馒头", "回忆", [0]),
    Probe(7, "我对什么东西过敏你记得吧？", "花生", "回忆", [1]),
    Probe(13, "我妈现在身体咋样了，你知道吗", "已经出院了（之前因肺炎住院）", "改口", [2, 7]),
    Probe(13, "我上次面试的是哪家公司来着？", "字节跳动", "回忆", [4, 8]),
    Probe(13, "我去年去日本玩的那次，你还记得吗？", "从没说过去日本", "从没说过"),
    Probe(14, "我搬家之前住哪个区？", "徐汇", "回忆", [9]),
    Probe(17, "你还记得我妹叫啥、在干嘛吗", "小雨，读大三，学设计", "回忆", [5]),
    Probe(17, "我现在住哪个区？", "浦东（之前在徐汇）", "改口", [9]),
    Probe(17, "我最近在学什么乐器？", "吉他", "回忆", [6]),
    Probe(20, "我平时早上喝什么？", "戒咖啡了，改喝茶（以前喝冰美式）", "改口", [3, 14]),
    Probe(20, "我找工作那事最后怎么样了", "字节面试通过，拿到 offer，下个月入职", "改口", [4, 11]),
    Probe(20, "我家猫最近怎么了？", "馒头肠胃炎，去了宠物医院", "回忆", [13]),
    Probe(20, "我和老周现在关系怎么样", "和好了，老周主动道歉（之前因借钱吵架）", "改口", [12, 15]),
    Probe(21, "我不吃什么菜来着", "香菜", "回忆", [10]),
    Probe(21, "我生日是哪天你记得吗", "12 月 3 日", "回忆", [16]),
    Probe(21, "我妈之前住院是什么病？", "肺炎", "回忆", [2]),
    Probe(21, "我之前说要考研那事，你觉得我该不该去？", "从没说过要考研", "从没说过"),
    Probe(21, "我养的那条狗叫什么？", "没有狗，养的是橘猫馒头", "从没说过", [0]),
    Probe(21, "馒头是什么颜色的猫？", "橘色", "回忆", [0]),
    Probe(21, "我对什么过敏来着？", "花生", "回忆", [1]),
]

GEN_PROMPT = """你在为一个聊天 AI 的长期记忆测试写剧本。剧本是一个用户和他的 AI 朋友在 QQ 上的私聊。
用户：{user}
这是第 {day} 天的聊天（共 {days} 天）。之前几天聊过的要点：{before}
今天用户要在聊天中自然地提到这些事（不要一上来就全说完，夹在闲聊里）：{facts}
要求：
- 一共 {n} 轮（用户说一句、AI 回一句算一轮），每句 10～60 个字，口语化。
- 其余内容是日常闲聊：工作、吃饭、天气、游戏、刷到的视频等，和之前几天不要重复。
- AI 是朋友的口吻，简短自然；AI 不要主动提起之前几天的事（那是测试要考的）。
- 只说今天提到的事，不要编出今天没列出的、关于用户的新的重要事实（家人、宠物、住址、工作变动、健康、过敏、生日等）。
只输出 JSON：{{"turns": [{{"role": "user", "text": "..."}}, {{"role": "assistant", "text": "..."}}, ...]}}"""


def _deepseek(messages: list[dict], max_tokens: int = 4000, json_mode: bool = False) -> str:
    sys.path.insert(0, str(ROOT / "src"))
    from llm.settings import load_settings

    s = load_settings()
    body = {"model": s.thinking_model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.9}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(
        s.deepseek_base_url.rstrip("/") + "/chat/completions",
        json.dumps(body).encode(),
        {"Authorization": "Bearer " + s.deepseek_api_key, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def generate(days: int = DAYS, turns_per_day: int = 20) -> dict:
    out = []
    for day in range(1, days + 1):
        today = [f.text for f in FACTS if f.day == day]
        before = "；".join(f.text for f in FACTS if f.day < day) or "（第一天）"
        prompt = GEN_PROMPT.format(
            user=USER,
            day=day,
            days=days,
            before=before,
            facts="；".join(today) or "（今天没有特别的事，纯闲聊）",
            n=turns_per_day,
        )
        data = json.loads(_deepseek([{"role": "user", "content": prompt}], json_mode=True))
        turns = [t for t in data["turns"] if t.get("role") in ("user", "assistant") and t.get("text")]
        out.append({"day": day, "turns": turns})
        print(f"第 {day} 天：{len(turns)} 句", flush=True)
    return {
        "user": USER,
        "facts": [asdict(f) for f in FACTS],
        "probes": [asdict(p) for p in PROBES],
        "days": out,
    }


def load() -> dict:
    return json.loads(SCRIPT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="生成评估剧本")
    ap.add_argument("--regen", action="store_true", help="已有剧本也重新生成")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if SCRIPT.exists() and not args.regen:
        sys.exit(f"已有剧本 {SCRIPT}，要重新生成加 --regen")
    SCRIPT.parent.mkdir(parents=True, exist_ok=True)
    SCRIPT.write_text(json.dumps(generate(), ensure_ascii=False, indent=1), encoding="utf-8")
    print("写好了", SCRIPT)
