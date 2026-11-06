"""Small client for the Claude Messages API, using only the standard library.

Needs an ANTHROPIC_API_KEY environment variable. ANTHROPIC_API_URL can point it
somewhere else, which is how the tests use a local fake server.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")
RETRYABLE = {429, 500, 502, 503, 529}


class ClaudeError(RuntimeError):
    pass


def call_claude(
    messages: list[dict[str, Any]],
    system: str = "",
    tools: list[dict[str, Any]] | None = None,
    model: str | None = None,
    max_tokens: int = 2000,
    retries: int = 3,
) -> dict[str, Any]:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ClaudeError("Set ANTHROPIC_API_KEY first.")
    url = os.environ.get("ANTHROPIC_API_URL", "https://api.anthropic.com/v1/messages")
    body: dict[str, Any] = {"model": model or DEFAULT_MODEL, "max_tokens": max_tokens, "messages": messages}
    if system:
        body["system"] = system
    if tools:
        body["tools"] = tools
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json", "x-api-key": key, "anthropic-version": "2023-06-01"},
        method="POST",
    )
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            if exc.code not in RETRYABLE or attempt == retries:
                raise ClaudeError(f"Claude API error {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            if attempt == retries:
                raise ClaudeError(f"Could not reach the Claude API: {exc.reason}") from exc
        time.sleep(2**attempt)
    raise ClaudeError("Unreachable")


def text_of(response: dict[str, Any]) -> str:
    return "".join(b.get("text", "") for b in response.get("content", []) if b.get("type") == "text")
