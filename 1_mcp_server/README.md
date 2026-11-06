# 1. MCP server: data-quality checks for any AI client

Lets Claude (or any MCP client) check a CSV or Excel file against business rules, and returns violations with spreadsheet row numbers. All tools are read-only and can only read one chosen folder.

| Tool | What it does |
|---|---|
| `list_data_files` | Lists files that can be checked |
| `profile_file` | Row count, column types, empty cells, examples |
| `suggest_rules` | Proposes rules from the data for a person to confirm |
| `check_rules` | Runs nine rule types, including IBAN checksum, date order and arithmetic such as interest = amount × rate |

## Results
On a synthetic 500-row deposit report with 60 planted errors, each of the six error types was found on exactly the planted rows. The test suite compares against `answer_key.json`, and also runs the tools through a real MCP client over stdio.

## Run it
```
pip install -r ../requirements.txt
python make_sample.py
python -m pytest
```

To use it in Claude Desktop, add this to `%APPDATA%\Claude\claude_desktop_config.json`, changing the paths to yours:
```json
{
  "mcpServers": {
    "plausi": {
      "command": "python",
      "args": ["C:\\path\\to\\ai-portfolio\\1_mcp_server\\server.py"],
      "env": { "PLAUSI_DATA_DIR": "C:\\path\\to\\your\\data" }
    }
  }
}
```
Then restart Claude and ask: "Check deposit_report.csv for data-quality problems."

## Design choices worth explaining in an interview
- **The checks are separate from MCP.** `plausi_engine.py` knows nothing about MCP, so a website or add-on could reuse it.
- **Every cell is read as text.** Pandas would otherwise guess types and hide problems such as `1.234,50` or a stray letter in an amount.
- **Errors are written for the model.** A broken rule is reported and the others still run. Expected errors use `ToolError`, because the SDK hides the reason for any other exception. A test exposed this.
- **No `eval`.** Arithmetic rules take fixed operators, so a rule can't run arbitrary code.
- **`suggest_rules` only suggests.** A pattern in today's file isn't proof of a business rule.

## Limits
- Tested on Linux with mcp 2.3. I haven't run the Claude Desktop setup on Windows.
- `check_rules` returns row numbers, not cell values. `profile_file` does return example values, which an AI client will see, so don't point it at real customer data without approval.
- Not connected to the Plausi app code.
