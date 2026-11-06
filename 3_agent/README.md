# 3. AI agent, without a framework

Reviews a data report against policy. It searches the policy (project 2), discovers and calls the data-quality tools over MCP (project 1), and writes a review that says whether the report can be sent.

```
python -m pytest
python run.py --scripted          # no key needed
python run.py                     # uses Claude, needs ANTHROPIC_API_KEY
```

On Windows, set the key with `$env:ANTHROPIC_API_KEY="..."` in PowerShell.

## What is and isn't proven
- **Proven by tests:** the loop, the failure handling, the real MCP and RAG connections, and the format of API requests (against a local fake server).
- **Not proven yet:** that Claude makes good decisions here. `--scripted` follows a fixed script, so no AI is deciding anything. **The live mode has not been run**, because no API key was available where this was built. Run it and read the output before describing this as a working AI agent.

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
