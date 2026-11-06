"""Measure the real thing: send Factfit's tailoring prompt to Claude and score what comes back.

    python live_eval.py --runs 3          needs ANTHROPIC_API_KEY, and costs API credit

This is what produces a number like "invented claims per 100 bullets". The prompt
below is copied from tailorCv() in Factfit's lib/ai.ts. If that prompt changes,
change it here too. Flags are candidates for review: open live_outputs.jsonl and
confirm them by hand before quoting a rate.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from checks import confirmed_facts, evaluate, factfit_current_guard
from fixtures import DEV, HOLDOUT

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "common"))
SYSTEM = "You are the engine of Factfit, a CV service with one hard rule: never invent experience. Always reply with a single JSON value and nothing else."


def tailor_prompt(case: dict) -> str:
    parsed = case["parsed"]
    facts = confirmed_facts(parsed, case["answers"])
    fact_list = "\n".join(f"[{i}] (role {f['role']}: {parsed['roles'][f['role']]['title']}) {f['text']}" for i, f in facts.items())
    role_list = "\n".join(f"role {i}: {r['title']}, {r['org']}, {r['dates']}" for i, r in enumerate(parsed["roles"]))
    return f"""Tailor a CV to a job ad using ONLY verified information. Reply with only JSON:
{{"jobTitle": string, "summary": string, "matched": [string], "missing": [string], "roles": [{{"role": number, "bullets": [{{"text": string, "facts": [string]}}]}}], "skills": [string]}}

Hard rules:
- Every bullet must be a rephrasing of one or more CONFIRMED FACTS and list their ids in "facts". Never add tools, numbers, employers, responsibilities or claims that are not in the confirmed facts.
- You may reorder, shorten and use the job ad's terminology only where the fact truly supports it.
- Order bullets within each role by relevance to the job ad. At most 5 bullets per role.
- "jobTitle": the job title from the ad, short.
- "summary": 2 sentences, built only from confirmed facts and the original CV.
- "matched": short keywords (1-3 words) from the job ad that the facts or the original CV clearly support.
- "missing": short keywords from the job ad's requirements that nothing supports. Be strict.
- "skills": skills from the original CV and confirmed facts only, most relevant to the ad first.
- Write all CV text in the same language as the job ad. Keywords stay in the job ad's wording.

ROLES:
{role_list}

CONFIRMED FACTS:
{fact_list}

ORIGINAL CV (true, may be used for summary and skills):
{case["cv_text"]}

JOB AD:
{case["job_ad"]}"""


def parse_json(text: str) -> dict:
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    raw = fenced.group(1) if fenced else text
    return json.loads(raw[raw.index("{") : raw.rindex("}") + 1])


def summarise(rows: list[dict]) -> dict:
    bullets = sum(r["bullets"] for r in rows)
    claim_checks = {"unsupported_number", "unsupported_term", "inflated_seniority", "unsupported_skill", "unsupported_match"}
    flags = sum(1 for r in rows for f in r["findings"] if f["check"] in claim_checks)
    return {
        "runs": len(rows),
        "bullets": bullets,
        "runs_with_a_flag": sum(1 for r in rows if r["findings"]),
        "claim_flags": flags,
        "claim_flags_per_100_bullets": round(100 * flags / bullets, 1) if bullets else None,
        "note": "Flags are unconfirmed until a person reviews them.",
    }


def main(ask=None, runs: int = 3, out: Path = HERE / "live_outputs.jsonl") -> dict:
    if ask is None:
        from claude import call_claude, text_of

        def ask(prompt):
            return text_of(call_claude([{"role": "user", "content": prompt}], system=SYSTEM, max_tokens=6000))

    rows = []
    with out.open("w", encoding="utf-8") as handle:
        for case in DEV + HOLDOUT:
            for run in range(runs):
                try:
                    output = factfit_current_guard(case, parse_json(ask(tailor_prompt(case))))
                except (ValueError, KeyError) as exc:
                    row = {"persona": case["id"], "run": run, "error": str(exc), "bullets": 0, "findings": []}
                else:
                    found = [vars(f) for f in evaluate(case, output)]
                    row = {"persona": case["id"], "run": run, "output": output,
                           "bullets": sum(len(b["bullets"]) for b in output["roles"]), "findings": found}
                rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return summarise(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3, help="Runs per fictional candidate")
    print(json.dumps(main(runs=parser.parse_args().runs), indent=2))
