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

## How to read these honestly
- **This does not say how often Factfit invents things.** It measures the checker on planted faults, not the AI. `live_eval.py` produces the real rate, and **it hasn't been run yet**, because it needs an API key.
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
