import copy
import json

import pytest

import live_eval
from checks import confirmed_facts, evaluate, factfit_current_guard, stem
from faults import FAULTS
from fixtures import DEV, HOLDOUT


def checks_of(findings):
    return {f.check for f in findings}


CASE = DEV[0]


def test_stem_links_word_forms():
    assert stem("Reconciled") == stem("reconciliation")
    assert stem("dashboards") == stem("dashboard")


def test_only_confirmed_answers_count():
    parsed = {"roles": [{"questions": ["a", "b", "c"]}]}
    answers = {"r0q0": {"text": "yes", "confirmed": True}, "r0q1": {"text": "no", "confirmed": False}, "r0q2": {"text": "  ", "confirmed": True}}
    assert list(confirmed_facts(parsed, answers)) == ["r0q0"]


def test_current_guard_matches_factfit_behaviour():
    output = copy.deepcopy(CASE["clean"])
    output["roles"][0]["bullets"][0]["facts"] = ["r9q9"]
    output["roles"][0]["bullets"][1]["text"] += " and invented 99%"
    guarded = factfit_current_guard(CASE, output)["roles"][0]["bullets"]
    assert len(guarded) == 2
    assert "invented 99%" in guarded[0]["text"]


@pytest.mark.parametrize(
    "text,check",
    [
        ("Reconciled partner bank files with 99% accuracy", "unsupported_number"),
        ("Reconciled partner bank files in Tableau", "unsupported_term"),
        ("Reconciled partner bank files every morning using Snowflake", "unsupported_term"),
        ("Led reconciliation of partner bank files", "inflated_seniority"),
    ],
)
def test_bullet_checks(text, check):
    output = copy.deepcopy(CASE["clean"])
    output["roles"][0]["bullets"][0]["text"] = text
    assert check in checks_of(evaluate(CASE, output))


def test_numbers_must_come_from_the_cited_fact():
    output = copy.deepcopy(CASE["clean"])
    output["roles"][0]["bullets"][0]["text"] = "Reconciled files, reducing errors by 15%"
    assert "unsupported_number" in checks_of(evaluate(CASE, output))


def test_every_clean_dev_cv_passes():
    for case in DEV:
        assert evaluate(case, case["clean"]) == [], case["id"]


def test_fixtures_only_cite_real_facts_from_the_right_role():
    for case in DEV + HOLDOUT:
        facts = confirmed_facts(case["parsed"], case["answers"])
        for block in case["clean"]["roles"]:
            for bullet in block["bullets"]:
                assert all(facts[f]["role"] == block["role"] for f in bullet["facts"])


def test_faults_do_not_change_the_clean_fixture():
    for fault in FAULTS:
        before = json.dumps(CASE["clean"])
        fault(CASE, copy.deepcopy(CASE["clean"]))
        assert json.dumps(CASE["clean"]) == before


def test_prompt_contains_rules_and_inputs():
    prompt = live_eval.tailor_prompt(CASE)
    assert "Never add tools, numbers" in prompt
    assert "[r0q1]" in prompt and CASE["job_ad"] in prompt


def test_parse_json_handles_a_fenced_reply():
    assert live_eval.parse_json('Here:\n```json\n{"a": 1}\n```') == {"a": 1}


def test_live_flow_with_a_fake_model(tmp_path):
    good = json.dumps(CASE["clean"])

    def fake(prompt):
        if CASE["job_ad"] in prompt:
            return good
        return "not json"

    summary = live_eval.main(ask=fake, runs=1, out=tmp_path / "o.jsonl")
    assert summary["runs"] == len(DEV) + len(HOLDOUT)
    assert summary["bullets"] == 4
    assert summary["claim_flags"] == 0
    assert len((tmp_path / "o.jsonl").read_text().splitlines()) == summary["runs"]
