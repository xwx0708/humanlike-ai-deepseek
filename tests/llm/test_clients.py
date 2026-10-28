"""DeepSeek、Ollama 客户端的请求格式、解析、重试和报错。不联网。"""

import pytest

from contracts.llm import LLMError, system, user
from llm import DeepSeekModel, FakeModel, OllamaModel
from llm._http import post_json
from llm.settings import load_settings

SCHEMA = {"type": "object", "properties": {"changed": {"type": "boolean"}}, "required": ["changed"]}


def deepseek_ok(content: str) -> dict:
    return {
        "model": "deepseek-flash",
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3},
    }


def test_deepseek_plain_chat(server):
    server.responses.append((200, deepseek_ok("你好")))
    m = DeepSeekModel("sk-test", base_url=server.url)
    r = m.chat([system("你是助手"), user("hi")], temperature=0.3, max_tokens=20)
    req = server.requests[0]
    assert req["path"] == "/chat/completions"
    assert req["headers"]["Authorization"] == "Bearer sk-test"
    assert req["body"]["model"] == "deepseek-flash"
    assert req["body"]["temperature"] == 0.3 and req["body"]["max_tokens"] == 20
    assert "response_format" not in req["body"]
    assert (r.text, r.prompt_tokens, r.completion_tokens, r.data) == ("你好", 10, 3, None)


def test_deepseek_json_mode_puts_schema_in_prompt(server):
    server.responses.append((200, deepseek_ok('{"changed": true}')))
    r = DeepSeekModel("k", base_url=server.url).chat([user("话题变了吗")], schema=SCHEMA)
    body = server.requests[0]["body"]
    assert body["response_format"] == {"type": "json_object"}
    assert body["messages"][0]["role"] == "system" and "changed" in body["messages"][0]["content"]
    assert r.data == {"changed": True}


def test_ollama_request_and_think_stripped(server):
    server.responses.append(
        (
            200,
            {
                "model": "qwen",
                "message": {"content": '<think>想想</think>\n{"changed": false}'},
                "prompt_eval_count": 7,
                "eval_count": 4,
            },
        )
    )
    m = OllamaModel("qwen", base_url=server.url, keep_alive="10m", num_ctx=8192)
    r = m.chat([user("x")], schema=SCHEMA, temperature=0.1, max_tokens=50)
    body = server.requests[0]["body"]
    assert server.requests[0]["path"] == "/api/chat"
    assert body["think"] is False and body["stream"] is False and body["keep_alive"] == "10m"
    assert body["format"] == SCHEMA
    assert body["options"] == {"num_ctx": 8192, "temperature": 0.1, "num_predict": 50}
    assert r.text == '{"changed": false}' and r.data == {"changed": False}
    assert (r.prompt_tokens, r.completion_tokens) == (7, 4)


def test_deepseek_thinking_off_by_default_and_on_when_asked(server):
    server.responses.append((200, deepseek_ok("好")))
    DeepSeekModel("k", base_url=server.url).chat([user("x")])
    assert server.requests[0]["body"]["thinking"] == {"type": "disabled"}

    reply = deepseek_ok("好")
    reply["choices"][0]["message"]["reasoning_content"] = "先想一想"
    server.responses.append((200, reply))
    r = DeepSeekModel("k", base_url=server.url, thinking=True).chat([user("x")])
    assert server.requests[1]["body"]["thinking"] == {"type": "enabled"}
    assert r.reasoning == "先想一想"


def test_truncated_json_gives_clear_error(server):
    reply = deepseek_ok('{"')
    reply["choices"][0]["finish_reason"] = "length"
    server.responses.append((200, reply))
    with pytest.raises(LLMError, match="截断"):
        DeepSeekModel("k", base_url=server.url).chat([user("x")], schema=SCHEMA, max_tokens=60)


def test_bad_json_raises(server):
    server.responses.append((200, {"message": {"content": "不是 JSON"}}))
    with pytest.raises(LLMError, match="不是合法 JSON"):
        OllamaModel(base_url=server.url).chat([user("x")], schema=SCHEMA)


def test_missing_required_field_raises(server):
    server.responses.append((200, {"message": {"content": '{"other": 1}'}}))
    with pytest.raises(LLMError, match="缺少字段"):
        OllamaModel(base_url=server.url).chat([user("x")], schema=SCHEMA)


def test_code_fence_is_tolerated(server):
    server.responses.append((200, deepseek_ok('```json\n{"changed": true}\n```')))
    assert DeepSeekModel("k", base_url=server.url).chat([user("x")], schema=SCHEMA).data == {"changed": True}


def test_retries_server_errors(server):
    server.responses += [(503, {"error": "busy"}), (429, {"error": "slow down"}), (200, deepseek_ok("好"))]
    waits = []
    out = post_json(server.url + "/chat/completions", {}, retries=2, sleep=waits.append)
    assert out["choices"][0]["message"]["content"] == "好"
    assert waits == [1, 2]


def test_auth_error_is_not_retried(server):
    server.responses += [(401, {"error": "bad key"}), (200, deepseek_ok("不该用到"))]
    with pytest.raises(LLMError, match="HTTP 401"):
        DeepSeekModel("bad", base_url=server.url, retries=3).chat([user("x")])
    assert len(server.requests) == 1


def test_connection_refused_raises():
    with pytest.raises(LLMError, match="连接失败"):
        post_json("http://127.0.0.1:9/none", {}, retries=1, sleep=lambda s: None)


def test_fake_model_records_calls_and_checks_schema():
    fake = FakeModel(["随便聊聊", {"changed": True}])
    assert fake.chat([user("a")]).text == "随便聊聊"
    assert fake.chat([user("b")], schema=SCHEMA).data == {"changed": True}
    assert [c["messages"][0].content for c in fake.calls] == ["a", "b"]
    with pytest.raises(LLMError, match="用完"):
        fake.chat([user("c")])


def test_settings_env_beats_dotenv(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "OLLAMA_MODEL=from-file\nOLLAMA_NUM_CTX=4096\n# 注释\nDEEPSEEK_MODEL='deepseek-v4-pro'\n", "utf-8"
    )
    monkeypatch.setenv("OLLAMA_MODEL", "from-env")
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    s = load_settings(env)
    assert s.ollama_model == "from-env"
    assert s.ollama_num_ctx == 4096
    assert s.deepseek_model == "deepseek-v4-pro"


def test_env_example_parses_cleanly(monkeypatch):
    """项目根目录的 .env.example 原样复制成 .env 也能用：行尾注释不会混进值里。"""
    from llm.settings import ROOT

    for k in ("DEEPSEEK_MODEL", "DEEPSEEK_THINKING", "OLLAMA_NUM_CTX", "OLLAMA_KEEP_ALIVE"):
        monkeypatch.delenv(k, raising=False)
    s = load_settings(ROOT / ".env.example")
    assert (s.deepseek_model, s.deepseek_thinking) == ("deepseek-flash", False)
    assert (s.ollama_num_ctx, s.ollama_keep_alive) == (16384, "30m")


def test_dotenv_quoted_value_keeps_hash(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text('DEEPSEEK_API_KEY="sk-a#b"  # 注释\n', "utf-8")
    assert load_settings(env).deepseek_api_key == "sk-a#b"


def test_settings_bool(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_THINKING", "true")
    assert load_settings(tmp_path / "none.env").deepseek_thinking is True
    monkeypatch.setenv("DEEPSEEK_THINKING", "0")
    assert load_settings(tmp_path / "none.env").deepseek_thinking is False
