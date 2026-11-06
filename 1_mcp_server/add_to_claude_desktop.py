"""Add the plausi MCP server to Claude Desktop's config, keeping everything else.

    python add_to_claude_desktop.py            uses the most recently changed config
    python add_to_claude_desktop.py <path>     uses that file

Quit Claude Desktop first, so it doesn't overwrite the change when it closes.
A backup is saved next to the file before anything is changed.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def find_configs() -> list[Path]:
    base = Path(os.environ.get("LOCALAPPDATA", "")), Path(os.environ.get("APPDATA", ""))
    found = []
    for folder in base:
        if folder.is_dir():
            found.extend(folder.rglob("claude_desktop_config.json"))
    return sorted(set(found), key=lambda p: p.stat().st_mtime, reverse=True)


def main() -> None:
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
    else:
        configs = find_configs()
        if not configs:
            sys.exit("No claude_desktop_config.json found. Pass the path as an argument.")
        for p in configs:
            print("Found:", p)
        target = configs[0]
    print("Updating:", target)

    data = json.loads(target.read_text(encoding="utf-8-sig"))
    backup = target.with_name(f"claude_desktop_config.backup-{time.strftime('%Y%m%d-%H%M%S')}.json")
    shutil.copy2(target, backup)
    print("Backup saved:", backup)

    python = ROOT / ".venv" / "Scripts" / "python.exe"
    if not python.exists():
        python = Path(sys.executable)
    data.setdefault("mcpServers", {})["plausi"] = {
        "command": str(python),
        "args": [str(HERE / "server.py")],
        "env": {"PLAUSI_DATA_DIR": str(HERE / "sample")},
    }
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    json.loads(target.read_text(encoding="utf-8"))  # make sure it still reads
    print("Added 'plausi'. Servers now:", ", ".join(data["mcpServers"]))
    print("Now reopen Claude Desktop.")


if __name__ == "__main__":
    main()
