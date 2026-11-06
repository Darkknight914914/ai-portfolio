# 4. Evals for Factfit's tailored CVs

Factfit's rule is "never invent experience". These checks test that rule on real Factfit output (the `TailorResult` shape from `lib/types.ts`).

```
python -m pytest
python run_eval.py holdout
python live_eval.py --runs 3     # needs ANTHROPIC_API_KEY and costs API credit
```

## The gap this found
`tailorCv()` in `lib/ai.ts` drops a bullet only when it has no valid fact ID. It never compares the wording with the fact. So a bullet can cite a real fact and still add a number or a tool. The summary, skills and keyword lists aren't checked at all.

## Results: 14 fault types planted in 4 fictional candidates (56 faults)
Scored once on a holdout set that wasn't used to build the checks.

| | Faults caught |
|---|---|
| Factfit's current check | 8 of 56 (14%) |
| New checks | 53 of 56 (95%) |
| False alarms on clean CVs | 1 across 52 clean items |

The false alarm is a real weakness: "month-end" in a keyword list wasn't matched to "month end" in the facts. I left it unfixed, so the holdout score stays honest.

## Live result: Factfit's real prompt with Claude Sonnet (6 November 2026)
I ran `live_eval.py --runs 3`: 8 fictional candidates, 3 runs each, so 24 tailored CVs. Every CV was then reviewed by hand, not only the flagged ones. The outputs are in `live_run_2026-11-06.jsonl`.

| | Result |
|---|---|
| Bullets | **99 of 99 faithful** to their cited facts |
| CVs with an unsupported claim | **6 of 24**, all in the summary or keyword list |
| Candidates affected | 2 of 8, **in every run** |

The problems:
- **Overclaimed keyword (3 of 3 runs):** "matched bank statements at month end" was listed as matching **"month-end close"**.
- **Fact put under the wrong job (3 of 3 runs):** the summary placed pytest testing in the candidate's *previous* job.
- **Borderline (3 of 3 runs):** "built REST APIs" was matched to **"API design"**. I didn't count it, and it doesn't change the 6 of 24.
- **Style, not a factual error (2 of 3 runs):** the summary switched to first person ("where I prepared…").

**What this shows:** the prompt keeps bullets honest, and Factfit's guard is aimed at bullets. The risk is in the summary and keyword lists, which nothing checks. The errors were the **same every run** for the same candidate, so regenerating a CV wouldn't fix them, but an automatic check would.

**How the checker did on real output:** of 10 flags, 3 were real, 3 borderline and 4 false alarms. The false alarms were the pronoun "I", and "4 years" worked out from dates 2021–2025. It **missed** the wrong-job error in all 3 runs, because it doesn't check which job a fact in the summary belongs to.

**Limits:** 8 fictional candidates, one model, one prompt version. "6 of 24" really means 2 of 8 people, so it describes these test candidates and isn't a rate for real users.

## After the fix: prompt v2 (same 24 CVs, same day)
v2 (in `live_eval.py`) makes every matched keyword cite the facts that prove it, drops keywords without valid evidence in code, and tells the summary to keep each fact under its own job and to avoid "I". Run with `python live_eval.py --runs 3 --prompt v2`. Outputs: `live_run_v2_2026-11-06.jsonl`. All 24 were reviewed by hand.

| | v1 (Factfit today) | v2 |
|---|---|---|
| Bullets faithful | 99 of 99 | 99 of 99 |
| CVs with an unsupported claim | **6 of 24** | **0 of 24** |
| "month-end close" overclaim | 3 of 3 runs | 0 (listed as missing) |
| pytest under the wrong job | 3 of 3 runs | 0 |
| Summary in first person | 2 of 3 runs | 0 |
| A skill the person has, wrongly listed as missing | 0 of 24 | **2 of 24** |

**The trade-off:** v2 is stricter, and twice it put "Google Analytics" under missing for a candidate who pulled reports from it. That doesn't invent anything, but it undersells the candidate. Some v2 summaries also slipped into passive voice ("campaigns were run"), because of the no-pronoun rule.

**The biggest caveat:** v2 was written after seeing these exact failures, and tested on the same 8 candidates. Some of the improvement may not carry over to new people. The fair next test is new fictional candidates that neither prompt has seen.

## How to read the planted-fault results
- **They don't say how often Factfit invents things.** They measure the checker on planted faults, not the AI. The live result above is the measure of the AI.
- **The test was fairly easy for the checks.** I wrote the checks, candidates and faults, and most faults are obvious on purpose. Expect a lower score on real output.
- **Plain-language exaggeration gets through.** Claims like "saving the business a great deal of money" were caught 1 time in 4, and that one was luck (a word that happened to be in the job ad). Catching these needs an AI judge, which is the next step.
- **English only.** German CVs still get the citation and number checks, but the word-level checks aren't reliable for German.
- **Flags aren't proof.** Review `live_outputs.jsonl` by hand before quoting a rate.

## Getting the number for your CV
1. Run `python live_eval.py --runs 3` (8 candidates × 3 runs = 24 API calls).
2. Open `live_outputs.jsonl` and mark which flags are real.
3. Quote it as: "Confirmed invented claims per 100 bullets on 8 fictional test profiles."
4. Add the checks to Factfit, change the prompt or guard, run again, and quote the before and after.

The prompt in `live_eval.py` is copied from `lib/ai.ts`. Update it if that prompt changes.
