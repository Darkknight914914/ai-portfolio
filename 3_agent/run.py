"""Run the agent.

  python run.py                 uses Claude (needs ANTHROPIC_API_KEY)
  python run.py --scripted      no key: a fixed script drives the real tools

The scripted run shows the loop and tools working. It does not show an AI making
decisions, because the script already knows each step.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from agent import Agent
from wiring import SYSTEM, build_tools, claude_model

HERE = Path(__file__).parent
TASK = "Review deposit_report.csv against our controls. Can it be sent to the partner bank?"
RULES = json.loads((HERE.parent / "1_mcp_server" / "rules_example.json").read_text())


def scripted_model():
    """Plays the model's part. The final answer is built from the real tool result."""
    step = {"n": 0}

    def use(name, payload):
        step["n"] += 1
        return {"content": [{"type": "tool_use", "id": f"call_{step['n']}", "name": name, "input": payload}]}

    def model(messages, system, tools):
        if step["n"] == 0:
            return use("search_policy", {"query": "controls for deposit report identifiers iban amounts dates interest"})
        if step["n"] == 1:
            return use("search_policy", {"query": "blocking issues escalation what a review must contain"})
        if step["n"] == 2:
            return use("check_rules", {"path": "deposit_report.csv", "rules": RULES})
        report = json.loads(messages[-1]["content"][0]["content"])
        lines = [f"Reviewed deposit_report.csv: {report['rows_checked']} rows."]
        for r in report["results"]:
            if r["status"] == "fail":
                lines.append(f"- {r['rule']['rule']}: {r['violations']} violations, e.g. rows {r['sample_rows'][:3]}")
        lines.append("Do not send: blocking issues found. Report owner fixes the source; raise to the Operations Lead today.")
        return {"content": [{"type": "text", "text": "\n".join(lines)}]}

    return model


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scripted", action="store_true")
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--trace", default=None, help="Save the step-by-step trace to this JSON file")
    args = parser.parse_args()
    async with build_tools() as tools:
        agent = Agent(scripted_model() if args.scripted else claude_model, tools, SYSTEM)
        result = await agent.run(args.task)
    for item in result.trace:
        if item["type"] == "tool":
            print(f"[step {item['step']}] {item['tool']} {'FAILED' if item['is_error'] else 'ok'}")
    print(f"\nStatus: {result.status}, steps: {result.steps}\n\n{result.answer}")
    if result.unsupported_numbers:
        print("\nWARNING, numbers no tool returned:", result.unsupported_numbers)
    if args.trace:
        Path(args.trace).write_text(json.dumps(result.trace, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
