"""Automatic checks for Factfit's tailored CVs (the TailorResult shape in lib/types.ts).

Factfit's rule is "never invent experience". Each check looks for a specific way
that rule can break, and needs no AI, so it is cheap enough to run on every CV.
It reads English; German CVs still get the citation and number checks, but the
word-level checks won't be reliable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

STOP = frozenset(
    """a about across after all also an and any are as at be been both by can during each for from has have
    in into is it its more of on or our over per that the their this through to up using was were which while
    with within you your i my we us own new key various multiple including based ensuring""".split()
)
# Everyday CV verbs. Rewording a fact with these isn't a new claim.
HARMLESS = frozenset(
    """work worked working support supported supporting help helped use used prepare prepared handle handled
    deliver delivered perform performed carry carried complete completed responsible provide provided
    create created run ran maintain maintained coordinate coordinated produce produced document documented
    review reviewed track tracked experience experienced strong proven""".split()
)
SENIORITY = frozenset(
    "led lead leading headed head managed manager managing directed director owned spearheaded supervised oversaw senior principal".split()
)
TOKEN = re.compile(r"[A-Za-zÄÖÜäöüß][A-Za-zÄÖÜäöüß0-9+#./-]*")
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def stem(word: str) -> str:
    w = word.lower().strip(".-/")
    for suffix in ("iations", "iation", "ations", "ation", "ments", "ment", "ings", "ing", "ers", "er", "ies", "ied", "ed", "es", "s"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 4:
            return w[: -len(suffix)]
    return w


def stems(text: str) -> set[str]:
    return {stem(t) for t in TOKEN.findall(text) if t.lower() not in STOP}


def numbers(text: str) -> set[str]:
    return {n.replace(",", ".").rstrip(".") for n in NUMBER.findall(text)}


@dataclass(frozen=True)
class Finding:
    check: str
    where: str
    detail: str


def confirmed_facts(parsed: dict, answers: dict) -> dict[str, dict]:
    """Same logic as confirmedFacts() in Factfit's lib/types.ts."""
    out = {}
    for ri, role in enumerate(parsed["roles"]):
        for qi, _ in enumerate(role["questions"]):
            key = f"r{ri}q{qi}"
            answer = answers.get(key)
            if answer and answer.get("confirmed") and answer["text"].strip():
                out[key] = {"role": ri, "text": answer["text"].strip()}
    return out


def factfit_current_guard(case: dict, output: dict) -> dict:
    """Port of what tailorCv() in lib/ai.ts does today: drop a citation that doesn't
    exist, then drop a bullet left with no citations. Nothing else is checked, so
    the summary, skills and bullet wording pass through untouched."""
    ids = set(confirmed_facts(case["parsed"], case["answers"]))
    roles = []
    for block in output["roles"]:
        bullets = []
        for bullet in block["bullets"]:
            kept = [f for f in bullet["facts"] if f in ids]
            if bullet["text"] and kept:
                bullets.append({**bullet, "facts": kept})
        roles.append({**block, "bullets": bullets})
    return {**output, "roles": roles}


def _capitalised_terms(text: str) -> list[str]:
    """Tool and brand names: capitalised words that don't start a sentence, or tokens like C++ or S3."""
    found = []
    for match in TOKEN.finditer(text):
        token = match.group()
        before = text[: match.start()].rstrip()
        sentence_start = not before or before[-1] in ".!?:•"
        if (token[0].isupper() and not sentence_start) or re.search(r"[+#]|\d", token) or (len(token) > 1 and token.isupper()):
            found.append(token.strip(".-/"))
    return found


def _unsupported_terms(text: str, source_text: str, ad_text: str) -> list[str]:
    source = stems(source_text)
    ad = stems(ad_text)
    bad = []
    for term in _capitalised_terms(text):
        if stem(term) not in source and term not in bad:
            bad.append(term)
    for token in TOKEN.findall(text):
        low = token.lower()
        if low in STOP or low in HARMLESS or low in SENIORITY:
            continue
        s = stem(token)
        if s in ad and s not in source and token not in bad:
            bad.append(token)
    return bad


def evaluate(case: dict, output: dict) -> list[Finding]:
    facts = confirmed_facts(case["parsed"], case["answers"])
    all_facts = " ".join(f["text"] for f in facts.values())
    profile = all_facts + " " + case["cv_text"]
    ad = case["job_ad"]
    findings: list[Finding] = []

    for block in output["roles"]:
        role = block["role"]
        if len(block["bullets"]) > 5:
            findings.append(Finding("too_many_bullets", f"role{role}", f"{len(block['bullets'])} bullets, limit is 5"))
        for i, bullet in enumerate(block["bullets"]):
            where = f"role{role}.bullet{i}"
            text = bullet["text"]
            known = [f for f in bullet["facts"] if f in facts]
            for fact_id in bullet["facts"]:
                if fact_id not in facts:
                    findings.append(Finding("invalid_citation", where, f"'{fact_id}' isn't a confirmed fact"))
            if not bullet["facts"]:
                findings.append(Finding("invalid_citation", where, "no citation"))
            for fact_id in known:
                if facts[fact_id]["role"] != role:
                    findings.append(Finding("wrong_role", where, f"'{fact_id}' belongs to another role"))
            source = " ".join(facts[f]["text"] for f in known)
            for n in sorted(numbers(text) - numbers(source)):
                findings.append(Finding("unsupported_number", where, n))
            for term in _unsupported_terms(text, source, ad):
                findings.append(Finding("unsupported_term", where, term))
            for token in TOKEN.findall(text):
                if token.lower() in SENIORITY and token.lower() not in {t.lower() for t in TOKEN.findall(source)}:
                    findings.append(Finding("inflated_seniority", where, token))

    for n in sorted(numbers(output["summary"]) - numbers(profile)):
        findings.append(Finding("unsupported_number", "summary", n))
    for term in _unsupported_terms(output["summary"], profile, ad):
        findings.append(Finding("unsupported_term", "summary", term))

    profile_stems = stems(profile)
    for field, label in (("skills", "unsupported_skill"), ("matched", "unsupported_match")):
        for item in output.get(field, []):
            if not stems(item) <= profile_stems:
                findings.append(Finding(label, field, item))
    for item in output.get("missing", []):
        words = stems(item)
        if words and words <= profile_stems:
            findings.append(Finding("false_missing", "missing", item))
    return findings
