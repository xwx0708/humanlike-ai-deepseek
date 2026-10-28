"""用关键词认出用户对记忆的要求（D29：用户定用关键词识别）。

能认出的：要求记住、要求忘掉、要求翻档案；以及要求忘掉之后的选择（彻底删除 / 存进档案 / 算了，D25）。
"要求修改"还没做：识别出来以后改什么、怎么改还没定。
关键词表是暂定的，按实际聊天再补。
"""

REMEMBER = "记住"
FORGET = "忘掉"
ARCHIVE = "翻档案"

# 先检查的在前：说"不用记住"时不该当成"记住"
KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    (
        FORGET,
        (
            "忘掉",
            "忘了吧",
            "忘记这",
            "别记着",
            "不要记着",
            "别记住",
            "不用记住",
            "不要记住",
            "删掉这",
            "当我没说",
        ),
    ),
    (ARCHIVE, ("翻档案", "翻翻档案", "查档案", "档案里", "以前的记录", "翻一下以前", "找找以前")),
    (REMEMBER, ("记住", "别忘了", "不要忘了", "帮我记", "记一下", "你要记得")),
]

# 要求忘掉之后，用户的选择（D25：由用户选，不替用户默认）
CHOICE_ERASE = ("彻底删除", "彻底删", "删干净", "彻底忘掉", "永久删除")
CHOICE_ARCHIVE = ("存进档案", "存档案", "放进档案", "放档案", "存档")
CHOICE_CANCEL = ("算了", "不用了", "取消", "不删了", "不忘了")


def detect(text: str) -> str | None:
    """这条消息是不是对记忆的要求；是的话返回 REMEMBER / FORGET / ARCHIVE。"""
    for kind, words in KEYWORDS:
        if any(w in text for w in words):
            return kind
    return None


def detect_choice(text: str) -> str | None:
    """要求忘掉之后的回答：返回 memory.archive 的方式（"彻底删除" / "存进档案"），"取消"，或 None（没选）。"""
    from memory.archive import ERASE, TO_ARCHIVE

    if any(w in text for w in CHOICE_CANCEL):
        return "取消"
    if any(w in text for w in CHOICE_ERASE):
        return ERASE
    if any(w in text for w in CHOICE_ARCHIVE):
        return TO_ARCHIVE
    return None


def strip_keywords(text: str) -> str:
    """去掉关键词，剩下的用来找"说的是哪件事"。"""
    for _, words in KEYWORDS:
        for w in words:
            text = text.replace(w, " ")
    return text.strip()
