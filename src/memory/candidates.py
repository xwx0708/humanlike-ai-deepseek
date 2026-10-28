"""事件线候选（用户方案，见 01-记忆/事件切分.md 第三层）。

常驻事件线单独成栏，整栏交给潜意识模型，不占名额（决策 D17，见 memory.resident）。
其余四种来源取并集，总数上限 20 条，且不含常驻的线：
1. 最近活跃的事件线
2. 人名或称谓重合的事件线（经人物别名表归一）
3. 关键词重合的事件线（jieba 分词 + BM25 打分）
4. 档案中人名强匹配的事件线（在消息原文里找人物的叫法，包括已经遗忘的片段）
"""

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass

import jieba

from contracts.memory import Segment
from memory.store import MemoryStore

jieba.setLogLevel(logging.WARNING)

SOURCES = ("最近活跃", "人物重合", "关键词重合", "档案人名")


@dataclass(frozen=True)
class CandidateRules:
    """数值都是暂定值，待用真实数据校准。"""

    max_total: int = 20  # 用户方案：总数上限 20
    recent: int = 8  # 最近活跃取几条
    keyword: int = 8  # 关键词重合取几条
    min_alias_len: int = 2  # 档案人名匹配时，叫法至少几个字（太短的"妈""哥"会误匹配）


DEFAULT_RULES = CandidateRules()

_STOP = {
    "的",
    "了",
    "是",
    "在",
    "和",
    "就",
    "都",
    "也",
    "还",
    "这",
    "那",
    "有",
    "没",
    "不",
    "吗",
    "呢",
    "吧",
    "啊",
    "呀",
    "嗯",
    "哦",
    "用户",
    "他们",
    "我们",
    "你们",
    "一个",
    "什么",
    "怎么",
    "自己",
    "因为",
    "所以",
    "但是",
}
_WORD = re.compile(r"\w")


def tokenize(text: str) -> list[str]:
    """jieba 分词，去掉标点、停用词和单字。"""
    return [w for w in jieba.lcut(text) if len(w) >= 2 and _WORD.search(w) and w not in _STOP]


def bm25(
    query: list[str], docs: dict[int, list[str]], k1: float = 1.5, b: float = 0.75
) -> list[tuple[int, float]]:
    """标准 BM25。返回 (文档 id, 得分)，只含得分大于 0 的，按得分从高到低。"""
    if not docs or not query:
        return []
    n = len(docs)
    avg_len = sum(len(d) for d in docs.values()) / n or 1.0
    df = Counter(w for d in docs.values() for w in set(d))
    scores = []
    for doc_id, words in docs.items():
        tf = Counter(words)
        s = 0.0
        for w in set(query):
            if w in tf:
                idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(words) / avg_len))
        if s > 0:
            scores.append((doc_id, s))
    return sorted(scores, key=lambda x: -x[1])


def select_candidates(
    store: MemoryStore, segment: Segment, rules: CandidateRules = DEFAULT_RULES
) -> dict[int, list[str]]:
    """返回 {事件线 id: 命中的来源列表}，最多 rules.max_total 条，不含常驻栏里的线。"""
    hits: dict[int, list[str]] = {}
    resident = set(store.resident_threads())

    def add(tids, source):
        for t in tids:
            if t in resident:
                continue
            hits.setdefault(t, [])
            if source not in hits[t]:
                hits[t].append(source)

    live = store.live_threads()  # 最近活跃的在前
    add(live[: rules.recent], "最近活跃")
    add(store.threads_with_people(segment.person_ids), "人物重合")

    turn_text = " ".join(
        t.content for t in store.turns(segment.conversation_id) if t.id in segment.message_ids
    )
    docs = {tid: tokenize(store.thread_text(tid)) for tid in live}
    add(
        [tid for tid, _ in bm25(tokenize(segment.title + " " + turn_text), docs)[: rules.keyword]],
        "关键词重合",
    )

    for pid in segment.person_ids:
        for alias in store.person(pid).aliases:
            if len(alias) >= rules.min_alias_len:
                add(store.threads_mentioning(alias, exclude_turns=segment.message_ids), "档案人名")

    if len(hits) > rules.max_total:
        # 暂行：命中来源多的优先，其次最近活跃的优先（见 00-总纲/偏离记录.md #9）
        recency = {tid: i for i, tid in enumerate(live)}
        keep = sorted(hits, key=lambda t: (-len(hits[t]), recency.get(t, len(live))))[: rules.max_total]
        hits = {t: hits[t] for t in keep}
    return hits
