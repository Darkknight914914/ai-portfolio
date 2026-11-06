"""Plant one known fault at a time in a clean CV, then see whether the checks catch it.
Same idea as mutation testing: the planted fault is the answer key."""
from __future__ import annotations

import copy


def _first_bullet(out):
    return out["roles"][0]["bullets"][0]


def _lower_first(text):
    return text[0].lower() + text[1:]


def fake_citation(c, o):
    _first_bullet(o)["facts"] = ["r9q9"]
    return "role0.bullet0"


def no_citation(c, o):
    _first_bullet(o)["facts"] = []
    return "role0.bullet0"


def wrong_role(c, o):
    other = o["roles"][1]["bullets"][0]["facts"][0]
    _first_bullet(o)["facts"] = [other]
    return "role0.bullet0"


def invented_number(c, o):
    _first_bullet(o)["text"] += ", cutting processing time by 37%"
    return "role0.bullet0"


def invented_tool_from_ad(c, o):
    _first_bullet(o)["text"] += f" and {c['traps']['ad_tool']}"
    return "role0.bullet0"


def invented_tool_not_in_ad(c, o):
    _first_bullet(o)["text"] += f" using {c['traps']['other_tool']}"
    return "role0.bullet0"


def inflated_seniority(c, o):
    b = _first_bullet(o)
    b["text"] = f"{c['traps']['inflate']} the team that {_lower_first(b['text'])}"
    return "role0.bullet0"


def summary_invented_number(c, o):
    o["summary"] += " Delivered 42% cost savings."
    return "summary"


def summary_invented_tool(c, o):
    o["summary"] += f" Experienced with {c['traps']['ad_tool']}."
    return "summary"


def invented_skill(c, o):
    o["skills"].append(c["traps"]["ad_skill"])
    return "skills"


def false_match(c, o):
    o["matched"].append(c["traps"]["ad_skill"])
    return "matched"


def false_missing(c, o):
    o["missing"].append(c["traps"]["supported"])
    return "missing"


def too_many_bullets(c, o):
    o["roles"][0]["bullets"] = o["roles"][0]["bullets"] + [copy.deepcopy(_first_bullet(o)) for _ in range(6)]
    return "role0"


def vague_overclaim(c, o):
    """No new number, tool or job-ad word: only a claim in plain language.
    Rule-based checks shouldn't be able to catch this, and that limit is worth measuring."""
    _first_bullet(o)["text"] += c["traps"]["vague"]
    return "role0.bullet0"


FAULTS = [
    fake_citation, no_citation, wrong_role, invented_number, invented_tool_from_ad, invented_tool_not_in_ad,
    inflated_seniority, summary_invented_number, summary_invented_tool, invented_skill, false_match, false_missing,
    too_many_bullets, vague_overclaim,
]
