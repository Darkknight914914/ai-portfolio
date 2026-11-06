# 3. AI agent, without a framework

Reviews a data report against policy. It searches the policy (project 2), discovers and calls the data-quality tools over MCP (project 1), and writes a review that says whether the report can be sent.

```
python -m pytest
python run.py --scripted          # no key needed
python run.py                     # uses Claude, needs ANTHROPIC_API_KEY
```

On Windows, set the key with `$env:ANTHROPIC_API_KEY="..."` in PowerShell.

## Live result with Claude
I ran `python run.py --trace live_trace.json` with Claude Sonnet on 6 November 2026. In 3 rounds, Claude:
1. searched the policy twice, listed the files and profiled the report (four tool calls at once)
2. wrote 8 rules from the policy and ran them, including interest = amount × rate ÷ 100 using the `scale` option
3. wrote the review

It found all six planted error types, with counts and row numbers matching `../1_mcp_server/answer_key.json`. It said the report must not be sent, and it applied the policy's 1% threshold correctly: 10 of 500 rows (2%) with interest errors makes that a blocking issue, not a warning. No number in its answer was flagged as unsupported. The full step-by-step record is in `live_trace.json`.

**This is one run on one synthetic file.** It shows that the agent works. It does not measure how reliably it works. Running the same task several times, and on files with different errors, would be the next test.

`--scripted` still runs without a key. It follows a fixed script, so no AI is deciding anything in that mode.

## Failure handling (each has a test)
| Failure | What happens |
|---|---|
| Tool raises an error | The reason goes back to the model, which can react |
| Model asks for a tool that doesn't exist | It's told which tools exist |
| Tool hangs | Timeout, reported to the model |
| Same call repeated | The third identical call is refused |
| Model never finishes | Stops at the step limit |
| Tool output is huge | Cut, with a note saying so |
| API is down | Retries on temporary errors, then ends cleanly |
| Model states a number no tool returned | Flagged in `unsupported_numbers` |

## Files
`agent.py` is the loop, `wiring.py` connects the real tools, and `policy/` holds a short fictional policy.

The number check is deliberately simple. It skips numbers under 10 and can't tell whether a correct number is used in the wrong way.
