"""MCP server that lets an AI client check spreadsheet data quality.

Run it with:  python server.py
It speaks MCP over stdin/stdout. Every tool is read-only, and files can only
be read from the folder named in the PLAUSI_DATA_DIR environment variable.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

import plausi_engine as engine

DATA_DIR = Path(os.environ.get("PLAUSI_DATA_DIR", Path(__file__).parent / "sample"))

mcp = MCPServer(
    "plausi",
    instructions=(
        "Checks CSV and Excel files against business rules. Typical order: list_data_files, "
        "profile_file, suggest_rules, then check_rules. Report row numbers as given: "
        "row 1 is the header."
    ),
)


def _load(path: str):
    try:
        return engine.load_table(engine.resolve_path(DATA_DIR, path))
    except engine.PlausiError as exc:
        # ToolError passes the reason to the client. Other exceptions are masked.
        raise ToolError(str(exc)) from exc


@mcp.tool()
def list_data_files() -> list[str]:
    """List the CSV and Excel files that are available to check."""
    return engine.list_files(DATA_DIR)


@mcp.tool()
def profile_file(path: str) -> dict[str, Any]:
    """Describe a file: row count, and each column's type, empty cells, distinct values and examples."""
    return engine.profile(_load(path))


@mcp.tool()
def suggest_rules(path: str) -> list[dict[str, Any]]:
    """Propose rules from the data's shape. Confirm them with a person before relying on them."""
    return engine.suggest_rules(_load(path))


@mcp.tool()
def check_rules(path: str, rules: list[dict[str, Any]]) -> dict[str, Any]:
    """Check a file against rules and return violations with spreadsheet row numbers.

    Each rule is an object with a "rule" key:
    - not_null: column
    - unique: columns (a list; together they must be unique)
    - in_range: column, min and/or max
    - allowed_values: column, values
    - regex: column, pattern (must match the whole value)
    - valid_date: column (YYYY-MM-DD or DD.MM.YYYY)
    - date_order: earlier, later (column names)
    - arithmetic: target, left, op (+ - * /), right, scale (default 1), tolerance (default 0.01)
    - iban_checksum: column
    """
    return engine.check(_load(path), rules)


if __name__ == "__main__":
    mcp.run()
