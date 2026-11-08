"""长期对话对照评估（项目总评第三步）：同一份剧本灌进两组，在同样的时间点问同样的问题，比谁答得对。

- 记忆组：DefaultMemoryService（概率海），后台按真实规则运行：模拟时间里每分钟扫描一格、每分钟 tick 一次
  （停顿后整理、对话中途分批、夜间合并，用真实的潜意识模型），上下文用到 start_tokens 就启动。
  回答问题时可以用 recall_memory 工具想一想；回复里的〔用到：记N〕照常加固。
- 对照组：RecentWindowMemory，只带最近的聊天记录，长度上限 = 记忆组的上下文预算（常驻 + 浮现 + 新鲜，D45）。
- 两组用同一个思考模型、同一个人设、同样的提示词组织（Responder）；剧本里的话双方都事先写好，
  只有问题由思考模型现场回答。回答记进各自的记忆，像真的聊过一样。
- 评分：思考模型按要点判对、部分对、不记得、错、瞎编（从没说过的事说得像真的）。

用法（在项目根目录）：
  python eval/run.py                       # 全量：21 天、20 个问题，约 1 小时
  python eval/run.py --days 3 --start-tokens 2000   # 小规模试跑
结果写在 eval/results/<时间>/：report.md（报告）、answers.json（全部问答和评分）、memory.db（记忆组的库）。
"""

import argparse
import json
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAMP = time.strftime("%Y%m%d-%H%M%S")
OUT = ROOT / "eval" / "results" / STAMP
# 剧本 21 天共约 1.9 万 token，比部署时几个月的聊天短得多。按部署预算（12500）对照组几乎能看到全部记录，
# 比不出差别；所以两组的长度上限按比例一起缩小（常驻 : 浮现 : 新鲜仍是 3000 : 1500 : 8000），启动线同样缩小。
# 偏离记录 #22。
EVAL_BUDGET = {"resident": 1000, "surfaced": 500, "fresh": 2500}
START_TOKENS = 5000
# 必须在导入项目代码之前：其余用默认参数（部署的 settings.json 也是空的），数据放在这次评估自己的文件夹，不碰 data/
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "settings.json").write_text(json.dumps({"budget": EVAL_BUDGET}), encoding="utf-8")
os.environ["HUMANLIKE_SETTINGS"] = str(OUT / "settings.json")
os.environ["HUMANLIKE_DATA_DIR"] = str(OUT / "data")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

import scenario

from llm import load_settings, subconscious_model
from llm.settings import Settings
from memory import DefaultMemoryService, RecentWindowMemory
from memory.budget import DEFAULT_KNOBS as BUDGET
from memory.lifecycle import register
from memory.runner import tick
from memory.scanner import Scanner
from memory.store import MemoryStore
from response.responder import Responder
from runtime.lifecycle import RUNNING, Lifecycle

PERSONA = "你是林舟在网上认识的朋友，平时在 QQ 上聊天。说话简短自然，像朋友一样，不用敬语。"
FIRST_DAY = time.mktime((2026, 10, 1, 21, 0, 0, 0, 0, -1))  # 每天晚上 9 点开始聊（本机时区）
DAY = 86400.0
STEP = 90.0  # 每句话隔一分半
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "recall_memory",
            "description": "在自己的记忆里想一想。想不起来、但觉得以前聊过的事时用。只能想起还记得的，已经忘掉的想不起来。",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "想回忆什么，用几个关键词描述"}},
                "required": ["query"],
            },
        },
    }
]
JUDGE = """你在给一个聊天 AI 的记忆测试评分。用户问了 AI 一个问题，请根据"正确要点"判断 AI 的回答。
问题：{q}
正确要点：{expect}
题型：{kind}（"改口"题要答出最新的情况；"从没说过"题里用户问的事从没发生过，AI 应该说不记得、没听说过，或纠正用户）
AI 的回答：{a}

只输出 JSON：{{"verdict": "对" | "部分对" | "不记得" | "错" | "瞎编", "reason": "一句话"}}
- 对：答出了要点（改口题要答最新的）；从没说过题：明确说没听说过、不记得，或纠正了用户。
- 部分对：答出一部分要点，或新旧情况都提了但没说清哪个是现在。
- 不记得：说想不起来、反问用户，没有给出错误信息。
- 错：给出了和要点矛盾的信息（改口题只答了旧情况也算错）。
- 瞎编：从没说过题里，把没发生的事当真、编出细节。"""


class DeepSeek:
    """思考模型和评分都用它：.env 里的 THINKING_MODEL（和 AstrBot 里配的同一个）。"""

    def __init__(self, s: Settings):
        self.s = s
        self.tokens = 0

    def call(self, messages: list[dict], tools: list | None = None, json_mode: bool = False) -> dict:
        body: dict = {"model": self.s.thinking_model, "messages": messages, "max_tokens": 800}
        if tools:
            body["tools"] = tools
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        req = urllib.request.Request(
            self.s.deepseek_base_url.rstrip("/") + "/chat/completions",
            json.dumps(body).encode(),
            {"Authorization": "Bearer " + self.s.deepseek_api_key, "Content-Type": "application/json"},
        )
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    data = json.load(r)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(5 * (attempt + 1))
        self.tokens += data.get("usage", {}).get("total_tokens", 0)
        return data


class Arm:
    """一组：Responder + 记忆。记忆组另有后台（扫描、整理）。"""

    def __init__(self, name: str, responder: Responder, background=None, tools: bool = False):
        self.name, self.r, self.background, self.tools = name, responder, background, tools
        self.prompt_tokens: list[int] = []
        self.recalls: list[str] = []

    def chat(self, text: str, reply: str, ts: float) -> None:
        """剧本里写好的一来一回。"""
        self.r.on_user_message(text, ts)
        self.r.on_reply(reply, ts + STEP / 2)

    def ask(self, ds: DeepSeek, q: str, ts: float) -> dict:
        p = self.r.on_user_message(q, ts)
        msgs = [{"role": "system", "content": p.system}, *p.contexts, {"role": "user", "content": p.prompt}]
        recalls = []
        for _ in range(4):  # 最多想三次再回答
            data = ds.call(msgs, TOOLS if self.tools and self.r.running else None)
            msg = data["choices"][0]["message"]
            used = data.get("usage", {}).get("prompt_tokens")
            if not msg.get("tool_calls"):
                break
            msgs.append(
                {"role": "assistant", "content": msg.get("content") or "", "tool_calls": msg["tool_calls"]}
            )
            for call in msg["tool_calls"]:
                query = json.loads(call["function"]["arguments"] or "{}").get("query", "")
                recalls.append(query)
                msgs.append({"role": "tool", "tool_call_id": call["id"], "content": self.r.recall(query, ts)})
        raw = msg.get("content") or ""
        if used:
            self.prompt_tokens.append(used)
        offered = sorted(self.r.offered)
        said = self.r.on_reply(raw, ts + STEP / 2, used)
        self.recalls += recalls
        return {"answer": said, "raw": raw, "prompt_tokens": used, "recalls": recalls, "offered": offered}


def run(days: int, start_tokens: int, out: Path) -> dict:
    script = scenario.load()
    settings = load_settings()
    ds, qwen = DeepSeek(settings), subconscious_model()
    budget = BUDGET.resident + BUDGET.surfaced + BUDGET.fresh

    store = MemoryStore(out / "memory.db")
    life = Lifecycle(path=out / "lifecycle.json", start_tokens=start_tokens)
    register(life, store, qwen, random.Random(7))
    svc = DefaultMemoryService(store, Scanner(store, rng=random.Random(7)), phase=lambda: life.phase)
    reports: list[str] = []

    def background(now: float) -> None:
        svc.scan(now)
        r = tick(store, qwen, now, life, settings.thinking_context_limit, rng=random.Random(int(now)))
        if r.started or r.consolidated or r.merged:
            t = time.strftime("%m-%d %H:%M", time.localtime(now))
            merged = getattr(r.merged, "merged", None)
            line = f"{t} {'启动 ' if r.started else ''}整理{list(r.consolidated)}{f' 合并{merged}' if merged else ''}"
            reports.append(line)
            print("  [后台]", line, flush=True)

    memory = Arm("记忆系统", Responder(svc, lambda: PERSONA), background, tools=True)
    window = Arm(
        "最近记录", Responder(RecentWindowMemory(window=10**6, token_budget=budget), lambda: PERSONA)
    )
    arms = [memory, window]
    probes = [p for p in script["probes"] if p["day"] <= days]
    results = []
    last = FIRST_DAY - DAY

    def advance(to: float) -> None:
        """记忆组的后台：对话期间每分钟一次；停顿时只在关键时刻（停顿后、夜里）跑，记忆时钟本来也几乎停了。"""
        nonlocal last
        if to - last > 3600:
            for t in (last + 30 * 60, to - 60):  # 停顿 30 分钟后整理一次；下一场开始前再跑一次
                background(t)
        else:
            t = last + 60
            while t <= to:
                background(t)
                t += 60
        last = to

    for d in script["days"][:days]:
        day_start = FIRST_DAY + (d["day"] - 1) * DAY
        print(f"第 {d['day']} 天（{life.phase}）", flush=True)
        ts = day_start
        turns = d["turns"]
        pairs = [
            (turns[i]["text"], turns[i + 1]["text"])
            for i in range(0, len(turns) - 1, 2)
            if turns[i]["role"] == "user" and turns[i + 1]["role"] == "assistant"
        ]
        for text, reply in pairs:
            advance(ts)
            for a in arms:
                a.chat(text, reply, ts)
            ts += STEP
        for p in [p for p in probes if p["day"] == d["day"]]:
            advance(ts)
            row = {**p, "phase": life.phase, "ts": ts}
            for a in arms:
                row[a.name] = a.ask(ds, p["question"], ts)
                print(f"  问：{p['question']}\n    {a.name}：{row[a.name]['answer']}", flush=True)
            results.append(row)
            ts += STEP
        last = max(last, ts)
    for row in results:
        for a in arms:
            v = json.loads(
                ds.call(
                    [
                        {
                            "role": "user",
                            "content": JUDGE.format(
                                q=row["question"],
                                expect=row["expect"],
                                kind=row["kind"],
                                a=row[a.name]["answer"],
                            ),
                        }
                    ],
                    json_mode=True,
                )["choices"][0]["message"]["content"]
            )
            row[a.name] |= {"verdict": v.get("verdict", "?"), "reason": v.get("reason", "")}
    stats = {
        "片段": len(store.segments()),
        "常驻事件线": len(store.resident_threads()),
        "画像事实": len(store.profile(everyone=True)),
        "启动时间": time.strftime("%m-%d %H:%M", time.localtime(life.started_ts))
        if life.started_ts
        else "没启动",
    }
    store.close()
    return {
        "results": results,
        "arms": {a.name: {"prompt_tokens": a.prompt_tokens, "recalls": a.recalls} for a in arms},
        "budget": budget,
        "deepseek_tokens": ds.tokens,
        "memory_stats": stats,
        "background": reports,
        "days": days,
        "start_tokens": start_tokens,
    }


SCORE = {"对": 1.0, "部分对": 0.5}


def report(data: dict) -> str:
    names = list(data["arms"])
    rows = data["results"]
    facts = data.get("facts") or scenario.load()["facts"]

    def score(rs, name):
        return sum(SCORE.get(r[name]["verdict"], 0) for r in rs), len(rs)

    def age(r):
        return r["day"] - min(facts[i]["day"] for i in r["about"]) if r["about"] else None

    out = [f"# 长期对话对照评估（{STAMP}）", ""]
    out.append(
        f"剧本 {data['days']} 天、{len(rows)} 个问题；记忆组启动线 {data['start_tokens']} token，"
        f"对照组聊天记录上限 {data['budget']} token（= 记忆组上下文预算）。DeepSeek 共用 {data['deepseek_tokens']} token。"
    )
    out += [
        "",
        "## 总分（对 = 1，部分对 = 0.5）",
        "",
        "| 分组 | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
    ]

    def line(label, rs):
        cells = []
        for n in names:
            s, k = score(rs, n)
            cells.append(f"{s:g} / {k}" if k else "—")
        return f"| {label} | " + " | ".join(cells) + " |"

    out.append(line("全部", rows))
    for kind in ("回忆", "改口", "从没说过"):
        out.append(line(kind, [r for r in rows if r["kind"] == kind]))
    for label, lo, hi in (("隔 0～6 天", 0, 6), ("隔 7～13 天", 7, 13), ("隔 14 天以上", 14, 99)):
        out.append(line(label, [r for r in rows if age(r) is not None and lo <= age(r) <= hi]))
    out.append(line("记忆组已启动后问的", [r for r in rows if r["phase"] == RUNNING]))
    out += ["", "## 各项计数", ""]
    for n in names:
        verdicts = [r[n]["verdict"] for r in rows]
        pt = data["arms"][n]["prompt_tokens"]
        out.append(
            f"- {n}："
            + "、".join(f"{v} {verdicts.count(v)}" for v in ("对", "部分对", "不记得", "错", "瞎编"))
            + f"；平均输入 {sum(pt) // max(len(pt), 1)} token；主动回忆 {len(data['arms'][n]['recalls'])} 次"
        )
    out.append(f"- 记忆组的库：{data['memory_stats']}")
    out += ["", "## 逐题", ""]
    for r in rows:
        a = f"隔 {age(r)} 天" if age(r) is not None else ""
        out.append(
            f"**第 {r['day']} 天 · {r['kind']} {a} · 记忆组{r['phase']}**：{r['question']}（要点：{r['expect']}）"
        )
        for n in names:
            extra = f"（想了：{'、'.join(r[n]['recalls'])}）" if r[n]["recalls"] else ""
            out.append(f"- {n} **{r[n]['verdict']}**：{r[n]['answer']}{extra}")
        out.append("")
    out += ["## 记忆组后台记录", "", *[f"- {b}" for b in data["background"]]]
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="长期对话对照评估")
    ap.add_argument("--days", type=int, default=scenario.DAYS)
    ap.add_argument(
        "--start-tokens", type=int, default=START_TOKENS, help="记忆组的启动线（评估里按比例调低）"
    )
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    data = run(args.days, args.start_tokens, OUT)
    (OUT / "answers.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "report.md").write_text(report(data), encoding="utf-8")
    print("报告：", OUT / "report.md")
