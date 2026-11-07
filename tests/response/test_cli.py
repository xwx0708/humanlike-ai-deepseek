"""命令行平台（D44）：不经 AstrBot 也能聊；思考模型走 contracts.platform.ThinkingBackend。"""

from llm import FakeModel
from memory import RecentWindowMemory
from response.cli import ModelBackend, run


class Echo:
    def __init__(self):
        self.calls = []

    def reply(self, system, contexts, prompt, image_paths=()):
        self.calls.append((system, [c["content"] for c in contexts], prompt))
        return f"你说：{prompt}", 42


def test_run_feeds_memory_and_backend_until_quit():
    memory, backend, said = RecentWindowMemory(), Echo(), []
    run(
        memory,
        backend,
        ["你好", "", "今天好累", "/quit", "不该到这"],
        persona=lambda: "人设",
        out=said.append,
        clock=lambda: 1.0,
    )
    assert said == ["你说：你好", "你说：今天好累"]
    system, history, prompt = backend.calls[1]
    assert (system.split("\n\n")[0], history[:2], prompt) == (
        "人设",
        ["你好", "你说：你好"],
        "今天好累",
    )  # 熟悉期：带历史
    assert len(history) == 3  # 末尾是人设提醒
    assert [t.content for t in memory.turns] == ["你好", "你说：你好", "今天好累", "你说：今天好累"]


def test_model_backend_builds_messages_and_reports_usage():
    model = FakeModel(["好的"])
    text, used = ModelBackend(model).reply("系统", [{"role": "user", "content": "早"}], "嗯")
    assert (text, used) == ("好的", None)  # 假模型不报用量
    assert [m.content for m in model.calls[0]["messages"]] == ["系统", "早", "嗯"]
