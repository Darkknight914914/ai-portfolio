"""Create a synthetic deposit report with known errors, plus an answer key.

Everything here is made up: no real customers, accounts or banks.
"""
import json
import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
ROWS = 500


def make_iban(rng: random.Random) -> str:
    bban = "".join(rng.choice("0123456789") for _ in range(18))
    digits = "".join(str(int(c, 36)) for c in bban + "DE00")
    return f"DE{98 - int(digits) % 97:02d}{bban}"


def build(seed: int = 7):
    rng = random.Random(seed)
    rows = []
    for i in range(ROWS):
        booked = date(2026, 1, 1) + timedelta(days=rng.randrange(0, 300))
        term = rng.choice([90, 180, 365, 730])
        amount = rng.randrange(1000, 100000, 500)
        rate = rng.choice([1.5, 2.0, 2.25, 2.5, 3.0])
        rows.append(
            {
                "deposit_id": f"DEP-{100000 + i}",
                "iban": make_iban(rng),
                "partner_bank": rng.choice(["Bank A", "Bank B", "Bank C"]),
                "currency": "EUR",
                "amount": f"{amount:.2f}",
                "interest_rate_pct": f"{rate:.2f}",
                "interest_amount": f"{amount * rate / 100:.2f}",
                "booking_date": booked.isoformat(),
                "maturity_date": (booked + timedelta(days=term)).isoformat(),
            }
        )
    df = pd.DataFrame(rows)

    picks = rng.sample(range(ROWS), 60)
    groups = {name: picks[i * 10 : (i + 1) * 10] for i, name in enumerate(
        ["duplicate_id", "bad_iban", "negative_amount", "bad_currency", "dates_reversed", "bad_interest"])}
    for i in groups["duplicate_id"]:
        df.loc[i, "deposit_id"] = df.loc[(i + 1) % ROWS, "deposit_id"]
    for i in groups["bad_iban"]:
        value = df.loc[i, "iban"]
        last = "0" if value[-1] != "0" else "1"
        df.loc[i, "iban"] = value[:-1] + last
    for i in groups["negative_amount"]:
        df.loc[i, "amount"] = "-" + df.loc[i, "amount"]
        df.loc[i, "interest_amount"] = "-" + df.loc[i, "interest_amount"]
    for i in groups["bad_currency"]:
        df.loc[i, "currency"] = "USD"
    for i in groups["dates_reversed"]:
        a, b = df.loc[i, "booking_date"], df.loc[i, "maturity_date"]
        df.loc[i, "booking_date"], df.loc[i, "maturity_date"] = b, a
    for i in groups["bad_interest"]:
        df.loc[i, "interest_amount"] = f"{float(df.loc[i, 'interest_amount']) + 12.34:.2f}"

    truth = {name: sorted(v + 2 for v in idx) for name, idx in groups.items()}
    dup_rows = [(i + 1) % ROWS + 2 for i in groups["duplicate_id"]]
    truth["duplicate_id"] = sorted(set(truth["duplicate_id"] + dup_rows))
    return df, truth


if __name__ == "__main__":
    frame, answer_key = build()
    (HERE / "sample").mkdir(exist_ok=True)
    frame.to_csv(HERE / "sample" / "deposit_report.csv", index=False)
    (HERE / "answer_key.json").write_text(json.dumps(answer_key, indent=2))
    print(f"Wrote {len(frame)} rows")
