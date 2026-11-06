# Deposit report controls (fictional example policy)

Every deposit report must pass these controls before it is shared with a partner bank.

## Identifiers
Each deposit_id must appear once only. A duplicate is a blocking issue, because it can cause a deposit to be paid twice.

## Account details
Every IBAN must pass the ISO 13616 checksum. A failed checksum is a blocking issue.

## Amounts and currency
The amount must not be negative. A negative amount is a blocking issue. The currency must be EUR. Any other currency is a blocking issue until Treasury confirms it.

## Dates
The booking_date must not be later than the maturity_date. Reversed dates are a blocking issue. All dates must be valid calendar dates.

## Interest
The interest_amount must equal the amount multiplied by interest_rate_pct divided by 100, within a tolerance of 0.01. A mismatch is a warning when fewer than 1% of rows are affected, and a blocking issue at 1% or more.
