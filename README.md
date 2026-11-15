# AI Engineering Portfolio

Five connected projects that show how AI products are built, connected to business data, and tested with real numbers.

**Highlight:** I evaluated the AI CV-tailoring feature of my own product, [Factfit](https://factfit.de). Every bullet point was supported by facts, but 6 of 24 test CVs claimed unproven skills in the summary or keyword list. A revised prompt that requires evidence for every claim brought this to **0 of 24**. [Read the evaluation →](4_evals)

## Projects

| # | Project | What it does | Result |
|---|---|---|---|
| 1 | [Data-Quality MCP Server](1_mcp_server) | Lets Claude check CSV and Excel files against business rules (IBAN checksums, duplicates, date order, interest calculations) | Found all 60 errors planted in a 500-row deposit report, on exactly the right rows |
| 2 | [RAG Pipeline](2_rag) | Document search built from its parts: keyword search, semantic search and a combination of both | 94% of test questions found a correct source in the top 5 results (18 questions, so a wide margin) |
| 3 | [Policy Review AI Agent](3_agent) | Searches policy documents, chooses which data checks to run and decides whether a report can be sent | In a live run with Claude, found all six error types and applied the escalation rules correctly |
| 4 | [LLM Evaluation for Factfit](4_evals) | Tests whether an AI CV-tailoring product invents experience, with automatic checks and hand review | Unsupported claims: 6 of 24 CVs before the fix, 0 of 24 after |
| 5 | [GPT from Scratch](5_gpt) | A small transformer language model in NumPy, including its own training engine | Beat a statistical text baseline; gradients verified numerically by tests |

## How the projects connect

```
 Policy documents ──► RAG (2) ──┐
                                ├──► AI Agent (3) ──► review: can the report be sent?
 Spreadsheet data ──► MCP (1) ──┘

 Factfit (live product) ──► Evals (4) ──► find problem ──► fix prompt ──► measure again
```

The data-quality checks are exposed as MCP tools, the agent combines them with document search, and the evaluation applies the same "measure, don't guess" approach to a real product.

## Tech stack
Python · Claude API (Anthropic) · Model Context Protocol (MCP) · NumPy · pandas · scikit-learn · pytest

## Run it yourself

Requires Python 3.10 or newer.

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt     (Windows)
.venv/bin/python -m pip install -r requirements.txt         (Mac or Linux)
```

Then run the tests in any project folder with `python -m pytest`. Every project has offline tests, and none of them need an API key. The live steps (agent, evaluation, RAG answers) need an `ANTHROPIC_API_KEY`. Each project's README explains how to run it.

## Honest notes
- **All data is made up.** The deposit report, policies and CV candidates are fictional. No real customer or personal data is used.
- **Small tests.** The results come from small test sets (24 CVs, 18 questions, one agent run). They show that each method works, not how it performs at scale. Each README lists its limits.
- **Built with AI assistance.** I developed this with Claude as a coding partner: Claude wrote most of the code and helped review the results, and I chose the projects and ran the tests and live experiments on my own machine.

## Next steps
- Test the improved Factfit prompt on new candidates it hasn't seen, and reduce its tendency to undersell skills.
- Bring the improved prompt into the live Factfit product.
- Generate RAG answers with Claude and measure answer quality, not only search quality.

## Author
**Akshay Jadhav**, Business Analyst, Berlin · [LinkedIn](https://www.linkedin.com/in/akshay-jadhav-49a697104)
