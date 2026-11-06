"""Connect the agent to the real MCP server from project 1 and the RAG from project 2."""
from __future__ import annotations

import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from mcp import Client, StdioServerParameters

from agent import Tool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "2_rag"))
sys.path.insert(0, str(ROOT / "common"))
from rag import Retriever, load_folder  # noqa: E402

SYSTEM = """You review data-quality for deposit reports.
Find the controls with search_policy before you decide which rules to check. Check the file with check_rules.
Report only what the tools returned, including their exact numbers. If a tool fails, say so rather than guessing.
Row numbers count the header as row 1. Finish with the review the escalation policy asks for."""


def _text(result) -> str:
    return "".join(getattr(block, "text", "") for block in result.content)


@asynccontextmanager
async def build_tools(data_dir: Path | None = None):
    """Yield the MCP tools, discovered from the server, plus search_policy."""
    data_dir = data_dir or ROOT / "1_mcp_server" / "sample"
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(ROOT / "1_mcp_server" / "server.py")],
        env={**os.environ, "PLAUSI_DATA_DIR": str(data_dir)},
    )
    retriever = Retriever(load_folder(Path(__file__).parent / "policy", size=60, overlap=10), dims=5)

    async def search_policy(args: dict) -> str:
        hits = retriever.search(args["query"], int(args.get("k", 4)))
        return json.dumps([{"source": c.ref, "text": c.text} for c, _ in hits])

    tools = [
        Tool(
            "search_policy",
            "Search the data-quality policy documents. Returns passages with source ids.",
            {"type": "object", "properties": {"query": {"type": "string"}, "k": {"type": "integer"}}, "required": ["query"]},
            search_policy,
        )
    ]
    async with Client(params) as client:
        for spec in (await client.list_tools()).tools:

            def make_runner(name: str):
                async def run(args: dict) -> str:
                    result = await client.call_tool(name, args)
                    if result.is_error:
                        raise RuntimeError(_text(result))
                    return _text(result)

                return run

            tools.append(Tool(spec.name, spec.description or "", spec.input_schema, make_runner(spec.name)))
        yield tools


def claude_model(messages, system, tools):
    from claude import call_claude

    return call_claude(messages, system=system, tools=tools, max_tokens=3000)
