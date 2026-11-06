# AI engineering portfolio

Five connected projects: the data-quality checks become MCP tools, the agent uses those tools plus RAG, and the evals test a real product. Each project has its own README with results and limits.

| # | Project | What it shows | Measured result |
|---|---|---|---|
| 1 | [MCP server](1_mcp_server) | Exposing business logic to AI clients | Found all planted errors, on exactly the right rows, in a synthetic 500-row report |
| 2 | [RAG](2_rag) | Retrieval built from parts, then measured | 94% recall@5 on 18 test questions (interval 83–100%) |
| 3 | [Agent](3_agent) | Tool loop and failure handling | With Claude, found all six error types and applied the policy correctly (one live run) |
| 4 | [Evals](4_evals) | Testing an AI product against its promise | Live: found unsupported claims in 6 of 24 CVs; a revised prompt cut that to 0 of 24, with 2 CVs now underselling a skill. The checker caught 53 of 56 planted faults, against 8 for the current check |
| 5 | [GPT](5_gpt) | Transformer and backprop internals | Beat a 5-gram baseline. Gradients verified numerically |

## Setup
```
pip install -r requirements.txt
```
Run each project's tests from its own folder with `python -m pytest`. Run `python make_sample.py` in `1_mcp_server` first if the sample is missing.

## Still to do
These steps need an `ANTHROPIC_API_KEY`:
1. `2_rag`: `python rag.py ../3_agent/policy "your question" --claude`.
