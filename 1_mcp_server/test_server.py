"""Tests the checks directly, then again through a real MCP client over stdio."""
import asyncio
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest
from mcp import Client, StdioServerParameters

import plausi_engine as engine

HERE = Path(__file__).parent
SAMPLE = HERE / "sample"
KEY = json.loads((HERE / "answer_key.json").read_text())
RULES = json.loads((HERE / "rules_example.json").read_text())


@pytest.fixture(scope="module")
def df():
    return engine.load_table(SAMPLE / "deposit_report.csv")


def rows_for(df, rule):
    result = engine.check(df, [rule])["results"][0]
    mask = engine.RULES[rule["rule"]](df, rule)
    return sorted(int(i) + 2 for i, flag in enumerate(mask) if flag), result


@pytest.mark.parametrize(
    "rule,key",
    [
        ({"rule": "unique", "columns": ["deposit_id"]}, "duplicate_id"),
        ({"rule": "iban_checksum", "column": "iban"}, "bad_iban"),
        ({"rule": "in_range", "column": "amount", "min": 0}, "negative_amount"),
        ({"rule": "allowed_values", "column": "currency", "values": ["EUR"]}, "bad_currency"),
        ({"rule": "date_order", "earlier": "booking_date", "later": "maturity_date"}, "dates_reversed"),
        ({"rule": "arithmetic", "target": "interest_amount", "left": "amount", "op": "*", "right": "interest_rate_pct", "scale": 0.01}, "bad_interest"),
    ],
)
def test_finds_exactly_the_planted_errors(df, rule, key):
    rows, _ = rows_for(df, rule)
    assert rows == KEY[key]


@pytest.mark.parametrize(
    "text,expected",
    [("1234.5", 1234.5), ("1.234,50", 1234.5), ("1,234.50", 1234.5), ("12,5", 12.5), ("1,234", 1234.0), ("-3", -3.0)],
)
def test_number_formats(text, expected):
    assert engine.to_number(text) == expected


def test_arithmetic_rule():
    df = pd.DataFrame({"a": ["2", "2"], "b": ["3", "3"], "c": ["6", "7"]})
    rows, _ = rows_for(df, {"rule": "arithmetic", "target": "c", "left": "a", "op": "*", "right": "b"})
    assert rows == [3]


def test_iban_check():
    assert engine.iban_is_valid("DE89370400440532013000")
    assert engine.iban_is_valid("DE89 3704 0044 0532 0130 00")
    assert not engine.iban_is_valid("DE89370400440532013001")
    assert not engine.iban_is_valid("not an iban")


def test_bad_rule_is_reported_and_others_still_run(df):
    out = engine.check(df, [{"rule": "nope"}, {"rule": "not_null", "column": "missing"}, {"rule": "not_null", "column": "iban"}])
    assert [r["status"] for r in out["results"]] == ["error", "error", "pass"]


def test_path_cannot_escape_data_folder():
    with pytest.raises(engine.PlausiError):
        engine.resolve_path(SAMPLE, "../server.py")
    with pytest.raises(engine.PlausiError):
        engine.resolve_path(SAMPLE, "/etc/passwd")


def test_suggestions_find_the_main_rules(df):
    kinds = {(s["rule"], s.get("column") or s.get("earlier")) for s in engine.suggest_rules(df)}
    assert ("iban_checksum", "iban") in kinds
    assert ("date_order", "booking_date") in kinds
    assert ("not_null", "amount") in kinds


def text_of(result):
    return "".join(getattr(block, "text", "") for block in result.content)


def test_over_real_mcp_stdio():
    async def main():
        params = StdioServerParameters(
            command=sys.executable,
            args=[str(HERE / "server.py")],
            env={**os.environ, "PLAUSI_DATA_DIR": str(SAMPLE)},
        )
        async with Client(params) as client:
            names = {t.name for t in (await client.list_tools()).tools}
            assert names == {"list_data_files", "profile_file", "suggest_rules", "check_rules"}
            schema = next(t for t in (await client.list_tools()).tools if t.name == "check_rules")
            assert "iban_checksum" in schema.description

            files = text_of(await client.call_tool("list_data_files", {}))
            assert "deposit_report.csv" in files

            report = await client.call_tool("check_rules", {"path": "deposit_report.csv", "rules": RULES})
            payload = json.loads(text_of(report))
            assert payload["rows_checked"] == 500
            assert payload["rules_failed"] == 6

            escaped = await client.call_tool("profile_file", {"path": "../server.py"})
            assert escaped.is_error and "outside" in text_of(escaped)

    asyncio.run(main())
