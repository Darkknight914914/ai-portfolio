import asyncio
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from agent import Agent, Tool, find_unsupported_numbers
from run import scripted_model, TASK
from wiring import SYSTEM, build_tools

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from claude import ClaudeError, call_claude  # noqa: E402


def use(name, payload=None, id="c1"):
    return {"content": [{"type": "tool_use", "id": id, "name": name, "input": payload or {}}]}


def say(text):
    return {"content": [{"type": "text", "text": text}]}


def script(*responses):
    queue = list(responses)
    seen = []

    def model(messages, system, tools):
        seen.append(json.loads(json.dumps(messages)))
        return queue.pop(0)

    model.seen = seen
    return model


async def ok_tool(args):
    return {"count": 123}


async def bad_tool(args):
    raise ValueError("boom")


async def slow_tool(args):
    await asyncio.sleep(5)


OK = Tool("ok", "", {"type": "object"}, ok_tool)
BAD = Tool("bad", "", {"type": "object"}, bad_tool)
SLOW = Tool("slow", "", {"type": "object"}, slow_tool)


def run(agent, message="go"):
    return asyncio.run(agent.run(message))


def test_answers_without_tools():
    result = run(Agent(script(say("hello")), []))
    assert (result.status, result.answer, result.steps) == ("completed", "hello", 1)


def test_tool_result_goes_back_to_the_model():
    model = script(use("ok"), say("done, 123"))
    result = run(Agent(model, [OK]))
    assert result.status == "completed"
    block = model.seen[1][-1]["content"][0]
    assert block["tool_use_id"] == "c1" and "123" in block["content"]
    assert result.unsupported_numbers == []


def test_tool_failure_reaches_the_model_and_the_run_continues():
    model = script(use("bad"), say("the tool failed"))
    result = run(Agent(model, [BAD]))
    block = model.seen[1][-1]["content"][0]
    assert block["is_error"] and "boom" in block["content"]
    assert result.status == "completed"


def test_unknown_tool_is_reported():
    model = script(use("ghost"), say("sorry"))
    run(Agent(model, [OK]))
    assert "Unknown tool" in model.seen[1][-1]["content"][0]["content"]


def test_slow_tool_times_out():
    model = script(use("slow"), say("timed out"))
    run(Agent(model, [SLOW], tool_timeout_seconds=0.05))
    assert "timed out" in model.seen[1][-1]["content"][0]["content"]


def test_stops_at_step_limit():
    def forever(messages, system, tools):
        return use("ok", {"n": len(messages)})

    result = run(Agent(forever, [OK], max_steps=4))
    assert result.status == "step_limit" and result.steps == 4


def test_third_identical_call_is_refused():
    model = script(use("ok"), use("ok"), use("ok"), say("stop"))
    result = run(Agent(model, [OK]))
    assert "twice already" in model.seen[3][-1]["content"][0]["content"]
    assert [t["is_error"] for t in result.trace if t["type"] == "tool"] == [False, False, True]


def test_parallel_tool_calls_each_get_a_result():
    both = {"content": use("ok", id="a")["content"] + use("bad", id="b")["content"]}
    model = script(both, say("done"))
    run(Agent(model, [OK, BAD]))
    blocks = model.seen[1][-1]["content"]
    assert [b["tool_use_id"] for b in blocks] == ["a", "b"]


def test_model_error_ends_the_run_cleanly():
    def broken(messages, system, tools):
        raise RuntimeError("api down")

    result = run(Agent(broken, []))
    assert result.status == "model_error" and "api down" in result.answer


def test_long_output_is_cut():
    async def big(args):
        return "x" * 500

    model = script(use("big"), say("ok"))
    run(Agent(model, [Tool("big", "", {}, big)], max_tool_output_chars=100))
    assert "output cut" in model.seen[1][-1]["content"][0]["content"]


def test_invented_number_is_flagged():
    assert find_unsupported_numbers("Found 123 rows and 456 errors.", ['{"count": 123}']) == ["456"]
    assert find_unsupported_numbers("Step 1 found 1,000 rows.", ['{"rows": 1000}']) == []


def test_real_tools_over_mcp_with_scripted_decisions():
    async def main():
        async with build_tools() as tools:
            names = {t.name for t in tools}
            assert {"search_policy", "check_rules", "profile_file"} <= names
            return await Agent(scripted_model(), tools, SYSTEM).run(TASK)

    result = asyncio.run(main())
    assert result.status == "completed"
    assert "500 rows" in result.answer and "Do not send" in result.answer
    assert result.unsupported_numbers == []
    policy = next(t for t in result.trace if t["tool" if t["type"] == "tool" else "type"] == "search_policy")
    assert "deposit" in policy["output"].lower() or "blocking" in policy["output"].lower()


def test_bad_path_from_the_model_comes_back_as_tool_error():
    async def main():
        async with build_tools() as tools:
            model = script(use("profile_file", {"path": "../server.py"}), say("not allowed"))
            await Agent(model, tools).run("x")
            return model.seen[1][-1]["content"][0]

    block = asyncio.run(main())
    assert block.get("is_error") and "outside" in block["content"]


class FakeClaude(BaseHTTPRequestHandler):
    """Local stand-in for the API, to test the request we send. Not a test of Claude itself."""

    received = []
    failures_left = 0

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        FakeClaude.received.append((dict(self.headers), body))
        if FakeClaude.failures_left:
            FakeClaude.failures_left -= 1
            self.send_response(529)
            self.end_headers()
            self.wfile.write(b"busy")
            return
        out = json.dumps(say("hi")).encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *args):
        pass


def with_fake_api(fn):
    server = HTTPServer(("127.0.0.1", 0), FakeClaude)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    os.environ["ANTHROPIC_API_URL"] = f"http://127.0.0.1:{server.server_port}/"
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    FakeClaude.received.clear()
    try:
        return fn()
    finally:
        server.shutdown()
        os.environ.pop("ANTHROPIC_API_URL")
        os.environ.pop("ANTHROPIC_API_KEY")


def test_request_format_and_retry(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    FakeClaude.failures_left = 1
    response = with_fake_api(lambda: call_claude([{"role": "user", "content": "x"}], system="s", tools=[{"name": "t"}]))
    assert response["content"][0]["text"] == "hi"
    assert len(FakeClaude.received) == 2
    headers, body = FakeClaude.received[-1]
    lowered = {k.lower(): v for k, v in headers.items()}
    assert lowered["x-api-key"] == "test-key" and lowered["anthropic-version"]
    assert body["system"] == "s" and body["tools"] == [{"name": "t"}] and body["max_tokens"] > 0


def test_missing_key_gives_a_clear_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    try:
        call_claude([])
    except ClaudeError as exc:
        assert "ANTHROPIC_API_KEY" in str(exc)
    else:
        raise AssertionError("expected ClaudeError")
