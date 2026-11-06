"""Business-rule data-quality checks for CSV and Excel files.

No MCP code lives here, so the checks can be tested and reused on their own.
Row numbers in every result match what a person sees in a spreadsheet: the
header is row 1 and the first data row is row 2.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MAX_SAMPLES = 20
ID_NAME = re.compile(r"(^|_)(id|iban|number|no|ref|reference)$", re.I)
IBAN_SHAPE = re.compile(r"^[A-Z]{2}\d{2}[A-Z0-9]{11,30}$")


class PlausiError(ValueError):
    """A problem the caller can fix, such as a bad path or an unknown column."""


def resolve_path(data_dir: str | Path, relative: str) -> Path:
    """Return a file inside data_dir, and refuse anything outside it."""
    root = Path(data_dir).resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise PlausiError(f"'{relative}' is outside the allowed data folder.")
    if not target.is_file():
        raise PlausiError(f"File not found: '{relative}'.")
    if target.suffix.lower() not in {".csv", ".xlsx"}:
        raise PlausiError("Only .csv and .xlsx files are supported.")
    return target


def list_files(data_dir: str | Path) -> list[str]:
    root = Path(data_dir).resolve()
    return sorted(
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".csv", ".xlsx"}
    )


def load_table(path: Path) -> pd.DataFrame:
    """Read every cell as text so the checks see what is really in the file."""
    if path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path, dtype=str, keep_default_na=False)
    else:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return df.apply(lambda col: col.str.strip())


def to_number(value: str) -> float:
    """Parse 1234.5, 1,234.50 and the German 1.234,50. Returns NaN if it can't."""
    if value is None or value == "":
        return float("nan")
    text = str(value).replace(" ", "").replace("€", "").replace("%", "")
    if "," in text and "." in text:
        decimal = "," if text.rfind(",") > text.rfind(".") else "."
        thousands = "." if decimal == "," else ","
        text = text.replace(thousands, "").replace(decimal, ".")
    elif "," in text:
        whole, _, frac = text.rpartition(",")
        text = text.replace(",", "") if len(frac) == 3 and whole else text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def to_numbers(series: pd.Series) -> pd.Series:
    return series.map(to_number).astype(float)


def to_dates(series: pd.Series) -> pd.Series:
    """Parse ISO (2026-03-31) and German (31.03.2026) dates. Anything else is NaT."""
    iso = pd.to_datetime(series, format="%Y-%m-%d", errors="coerce")
    german = pd.to_datetime(series, format="%d.%m.%Y", errors="coerce")
    return iso.fillna(german)


def iban_is_valid(value: str) -> bool:
    """ISO 13616 mod-97 check."""
    text = str(value).replace(" ", "").upper()
    if not IBAN_SHAPE.match(text):
        return False
    rearranged = text[4:] + text[:4]
    digits = "".join(str(int(ch, 36)) for ch in rearranged)
    return int(digits) % 97 == 1


def profile(df: pd.DataFrame) -> dict[str, Any]:
    columns = []
    for name in df.columns:
        col = df[name]
        filled = col[col != ""]
        n = max(len(filled), 1)
        numeric_share = to_numbers(filled).notna().sum() / n
        date_share = to_dates(filled).notna().sum() / n
        kind = "date" if date_share >= 0.9 else "number" if numeric_share >= 0.9 else "text"
        columns.append(
            {
                "name": name,
                "type_guess": kind,
                "empty": int((col == "").sum()),
                "distinct": int(filled.nunique()),
                "examples": filled.drop_duplicates().head(3).tolist(),
            }
        )
    return {"rows": len(df), "columns": columns}


def _require(df: pd.DataFrame, rule: dict, key: str) -> Any:
    if key not in rule:
        raise PlausiError(f"Rule '{rule.get('rule')}' needs '{key}'.")
    return rule[key]


def _column(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        raise PlausiError(f"Unknown column '{name}'. Columns: {', '.join(df.columns)}.")
    return df[name]


def _not_null(df, r):
    return _column(df, _require(df, r, "column")) == ""


def _unique(df, r):
    names = _require(df, r, "columns")
    for name in names:
        _column(df, name)
    filled = (df[names] != "").all(axis=1)
    return df.duplicated(subset=names, keep=False) & filled


def _in_range(df, r):
    col = _column(df, _require(df, r, "column"))
    numbers = to_numbers(col)
    bad = (col != "") & numbers.isna()
    if "min" in r:
        bad |= numbers < float(r["min"])
    if "max" in r:
        bad |= numbers > float(r["max"])
    return bad


def _allowed_values(df, r):
    col = _column(df, _require(df, r, "column"))
    return (col != "") & ~col.isin([str(v) for v in _require(df, r, "values")])


def _regex(df, r):
    col = _column(df, _require(df, r, "column"))
    try:
        pattern = re.compile(_require(df, r, "pattern"))
    except re.error as exc:
        raise PlausiError(f"Invalid pattern: {exc}") from exc
    return (col != "") & ~col.map(lambda v: bool(pattern.fullmatch(v)))


def _valid_date(df, r):
    col = _column(df, _require(df, r, "column"))
    return (col != "") & to_dates(col).isna()


def _date_order(df, r):
    earlier = to_dates(_column(df, _require(df, r, "earlier")))
    later = to_dates(_column(df, _require(df, r, "later")))
    return earlier > later


def _arithmetic(df, r):
    """target must equal (left <op> right) * scale, within tolerance."""
    target = to_numbers(_column(df, _require(df, r, "target")))
    left = to_numbers(_column(df, _require(df, r, "left")))
    right = to_numbers(_column(df, _require(df, r, "right")))
    op = _require(df, r, "op")
    if op == "+":
        expected = left + right
    elif op == "-":
        expected = left - right
    elif op == "*":
        expected = left * right
    elif op == "/":
        expected = left / right.replace(0, np.nan)
    else:
        raise PlausiError("'op' must be one of + - * /.")
    expected = expected * float(r.get("scale", 1))
    tolerance = float(r.get("tolerance", 0.01))
    known = target.notna() & expected.notna()
    return known & ((target - expected).abs() > tolerance)


def _iban(df, r):
    col = _column(df, _require(df, r, "column"))
    return (col != "") & ~col.map(iban_is_valid)


RULES = {
    "not_null": _not_null,
    "unique": _unique,
    "in_range": _in_range,
    "allowed_values": _allowed_values,
    "regex": _regex,
    "valid_date": _valid_date,
    "date_order": _date_order,
    "arithmetic": _arithmetic,
    "iban_checksum": _iban,
}


def check(df: pd.DataFrame, rules: list[dict]) -> dict[str, Any]:
    """Run every rule. A broken rule is reported and doesn't stop the others."""
    results = []
    failing_rows: set[int] = set()
    for rule in rules:
        name = rule.get("rule")
        entry: dict[str, Any] = {"rule": rule}
        if name not in RULES:
            entry.update(status="error", error=f"Unknown rule '{name}'. Options: {', '.join(RULES)}.")
            results.append(entry)
            continue
        try:
            mask = RULES[name](df, rule)
        except PlausiError as exc:
            entry.update(status="error", error=str(exc))
            results.append(entry)
            continue
        rows = [int(i) + 2 for i in np.flatnonzero(mask.to_numpy())]
        failing_rows.update(rows)
        entry.update(
            status="pass" if not rows else "fail",
            violations=len(rows),
            sample_rows=rows[:MAX_SAMPLES],
            truncated=len(rows) > MAX_SAMPLES,
        )
        results.append(entry)
    counts = [r["status"] for r in results]
    return {
        "rows_checked": len(df),
        "rules_passed": counts.count("pass"),
        "rules_failed": counts.count("fail"),
        "rules_errored": counts.count("error"),
        "rows_with_issues": len(failing_rows),
        "results": results,
    }


def suggest_rules(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Propose rules from the data's shape. A person should confirm each one,
    because a pattern in today's file isn't proof of a business rule."""
    out: list[dict[str, Any]] = []
    n = max(len(df), 1)
    for name in df.columns:
        col = df[name]
        filled = col[col != ""]
        if filled.empty:
            continue
        if (col == "").sum() / n <= 0.05:
            out.append({"rule": "not_null", "column": name, "why": "Almost every row has a value."})
        if ID_NAME.search(name) and filled.nunique() / len(filled) >= 0.95:
            out.append({"rule": "unique", "columns": [name], "why": "Looks like an identifier."})
        if "iban" in name.lower():
            out.append({"rule": "iban_checksum", "column": name, "why": "Column name mentions IBAN."})
        date_share = to_dates(filled).notna().mean()
        number_share = to_numbers(filled).notna().mean()
        if date_share >= 0.9:
            out.append({"rule": "valid_date", "column": name, "why": "Nearly all values are dates."})
        elif number_share >= 0.9 and (to_numbers(filled).dropna() >= 0).mean() >= 0.95:
            out.append({"rule": "in_range", "column": name, "min": 0, "why": "Values are almost all non-negative."})
        elif filled.nunique() <= 10 and number_share < 0.9:
            common = filled.value_counts()
            kept = common[common / len(filled) >= 0.01].index.tolist()
            out.append({"rule": "allowed_values", "column": name, "values": kept, "why": "Few distinct values."})
    dates = [c for c in df.columns if (df[c] != "").any() and to_dates(df[c][df[c] != ""]).notna().mean() >= 0.9]
    starts = [c for c in dates if re.search(r"start|open|from|begin|booking", c, re.I)]
    ends = [c for c in dates if re.search(r"end|close|to$|until|maturity|value", c, re.I)]
    for a in starts:
        for b in ends:
            if a != b:
                out.append({"rule": "date_order", "earlier": a, "later": b, "why": "Column names suggest a time order."})
    return out
