"""Score the checks against planted faults, next to what Factfit's current guard catches.

    python run_eval.py dev
    python run_eval.py holdout

These numbers describe the checker on synthetic CVs. They do NOT say how often
Factfit invents things in real use: live_eval.py measures that.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

from checks import evaluate, factfit_current_guard
from faults import FAULTS
from fixtures import DEV, HOLDOUT

HERE = Path(__file__).parent


def guard_catches(case, faulted, where):
    """Today's guard 'catches' a fault only if it removes the faulty bullet."""
    if not where.startswith("role0.bullet"):
        return False
    return not factfit_current_guard(case, faulted)["roles"][0]["bullets"] or len(
        factfit_current_guard(case, faulted)["roles"][0]["bullets"]
    ) < len(faulted["roles"][0]["bullets"])


def score(personas):
    per_fault = {}
    for fault in FAULTS:
        caught = guard = 0
        missed = []
        for case in personas:
            faulted = copy.deepcopy(case["clean"])
            where = fault(case, faulted)
            before = set(evaluate(case, case["clean"]))
            new = [f for f in evaluate(case, faulted) if f not in before and f.where == where]
            caught += bool(new)
            if not new:
                missed.append(case["id"])
            guard += guard_catches(case, faulted, where)
        per_fault[fault.__name__] = {"checks_caught": caught, "current_guard_caught": guard, "cases": len(personas), "missed": missed}

    total = sum(v["cases"] for v in per_fault.values())
    clean_items = false_alarms = 0
    alarms = []
    for case in personas:
        output = case["clean"]
        clean_items += sum(len(b["bullets"]) for b in output["roles"]) + 1 + len(output["skills"]) + len(output["matched"]) + len(output["missing"])
        found = evaluate(case, output)
        false_alarms += len(found)
        alarms += [f"{case['id']}: {f.check} at {f.where}: {f.detail}" for f in found]
    return {
        "faults_planted": total,
        "checks_caught": sum(v["checks_caught"] for v in per_fault.values()),
        "current_guard_caught": sum(v["current_guard_caught"] for v in per_fault.values()),
        "clean_items_checked": clean_items,
        "false_alarms_on_clean": false_alarms,
        "false_alarm_details": alarms,
        "per_fault": per_fault,
    }


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "dev"
    result = score(DEV if name == "dev" else HOLDOUT)
    (HERE / f"results_{name}.json").write_text(json.dumps(result, indent=2))
    print(f"{name}: planted {result['faults_planted']}, checks caught {result['checks_caught']}, "
          f"current guard caught {result['current_guard_caught']}, "
          f"false alarms {result['false_alarms_on_clean']} on {result['clean_items_checked']} clean items")
    for key, value in result["per_fault"].items():
        print(f"  {key:26} checks {value['checks_caught']}/{value['cases']}  guard {value['current_guard_caught']}/{value['cases']}  missed {value['missed']}")
    for line in result["false_alarm_details"]:
        print("  false alarm:", line)
