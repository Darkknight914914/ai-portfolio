"""An AI agent written without a framework, so every part of the loop is visible.

The loop: ask the model, run any tools it requests, feed the results back, and
repeat until it answers or a limit stops it. The failure handling matters more
than the happy path: tool errors go back to the model, repeated calls are
called out, and the run always ends.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

Model = Callable[[list[dict], str, list[dict]], dict]
ToolRunner = Callable[[dict], Awaitable[Any]]


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    run: ToolRunner

    def spec(self) -> dict:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}


@dataclass
class AgentResult:
    status: str  # completed, step_limit, model_error
    answer: str
    steps: int
    trace: list[dict] = field(default_factory=list)
    unsupported_numbers: list[str] = field(default_factory=list)


NUMBER = re.compile(r"(?<![\w.,-])\d+(?:[.,]\d+)*")


def find_unsupported_numbers(answer: str, tool_outputs: list[str], user_message: str = "") -> list[str]:
    """Numbers in the answer that no tool returned and the user never gave.
    A cheap check that catches a model making up a count. Numbers under 10 are
    skipped because they appear naturally in prose, such as list numbering."""
    source = " ".join(tool_outputs) + " " + user_message
    known = set(NUMBER.findall(source))
    missing = []
    for raw in NUMBER.findall(answer):
        cleaned = raw.rstrip(".,")
        if cleaned in known or cleaned.replace(",", "") in known:
            continue
        try:
            if float(cleaned.replace(",", "")) < 10:
                continue
        except ValueError:
            pass
        if cleaned not in missing:
            missing.append(cleaned)
    return missing


class Agent:
    def __init__(
        self,
        model: Model,
        tools: list[Tool],
        system: str = "",
        max_steps: int = 12,
        max_tool_output_chars: int = 20000,
        tool_timeout_seconds: float = 60,
    ):
        self.model = model
        self.tools = {t.name: t for t in tools}
        self.specs = [t.spec() for t in tools]
        self.system = system
        self.max_steps = max_steps
        self.max_tool_output_chars = max_tool_output_chars
        self.tool_timeout_seconds = tool_timeout_seconds

    async def _run_tool(self, call: dict, seen: dict[str, int]) -> tuple[str, bool]:
        """Return (output, is_error). Never raises: a failed tool is information
        the model can react to, not a reason to crash the run."""
        tool = self.tools.get(call["name"])
        if tool is None:
            return f"Unknown tool '{call['name']}'. Available: {', '.join(self.tools)}.", True
        key = call["name"] + json.dumps(call.get("input", {}), sort_keys=True)
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > 2:
            return "You have made this exact call twice already with the same input. Change the input or finish.", True
        try:
            output = await asyncio.wait_for(tool.run(call.get("input", {})), self.tool_timeout_seconds)
        except TimeoutError:
            return f"Tool '{call['name']}' timed out.", True
        except Exception as exc:  # the model decides what to do next
            return f"Tool '{call['name']}' failed: {exc}", True
        text = output if isinstance(output, str) else json.dumps(output, default=str)
        if len(text) > self.max_tool_output_chars:
            text = text[: self.max_tool_output_chars] + "\n[output cut: too long]"
        return text, False

    async def run(self, user_message: str) -> AgentResult:
        messages: list[dict] = [{"role": "user", "content": user_message}]
        trace: list[dict] = []
        outputs: list[str] = []
        seen: dict[str, int] = {}
        for step in range(1, self.max_steps + 1):
            try:
                response = await asyncio.to_thread(self.model, messages, self.system, self.specs)
            except Exception as exc:
                trace.append({"step": step, "type": "model_error", "error": str(exc)})
                return AgentResult("model_error", f"The model call failed: {exc}", step, trace)
            content = response.get("content", [])
            messages.append({"role": "assistant", "content": content})
            calls = [b for b in content if b.get("type") == "tool_use"]
            text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
            if not calls:
                trace.append({"step": step, "type": "answer", "text": text})
                unsupported = find_unsupported_numbers(text, outputs, user_message)
                return AgentResult("completed", text, step, trace, unsupported)
            results = []
            for call in calls:
                output, is_error = await self._run_tool(call, seen)
                if not is_error:
                    outputs.append(output)
                trace.append(
                    {"step": step, "type": "tool", "tool": call["name"], "input": call.get("input", {}),
                     "is_error": is_error, "output": output}
                )
                result = {"type": "tool_result", "tool_use_id": call["id"], "content": output}
                if is_error:
                    result["is_error"] = True
                results.append(result)
            messages.append({"role": "user", "content": results})
        return AgentResult("step_limit", f"Stopped after {self.max_steps} steps without a final answer.", self.max_steps, trace)
