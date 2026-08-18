# Panel provenance

What each column is, where it came from, and why it can be published.

| Column | Source | Nature |
|---|---|---|
| `ticker`, `score_year`, `fiscal_year` | identifiers | not data |
| `f_roa` … `f_dturn` (nine flags) | derived from licensed-terminal fundamentals | **irreversible**: each is one bit, the outcome of a comparison (e.g. ROA > 0). The underlying value cannot be recovered. |
| `fscore` | sum of the nine flags | further aggregation, 0–9 |
| `bm`, `market_cap` | this project's own Yahoo Finance / SEC EDGAR cache | not vendor data |
| `sector` | Yahoo Finance classification | not vendor data |

Deliberately **not** exported: `shares_outstanding` — a per-security raw
value from the source workbook. It is unused downstream; the EQ_OFFER flag
was computed inside the workbook from its own year-on-year share columns, so
removing it changes no result.

## Information content

One firm-year emits nine bits, derived from roughly fifteen continuous line
items across three fiscal years. The mapping is many-to-one: infinitely many
statement sets produce the same nine flags. The panel therefore does not
substitute for, and cannot be used to reconstruct, the source data.

The facts the flags summarise — whether ROA was positive, whether gross
margin improved — are disclosed in public statutory filings (SEC 10-K,
Japanese 有価証券報告書). A reader can recompute every flag from those
filings without any commercial subscription.

## Reproducing the results from this panel

`results/panel/*.csv` plus the price caches rebuilt by
`scripts/fetch_us_japan.py` / `scripts/fetch_us_edgar.py` are enough to
re-run every notebook: the study consumes `fscore` for ranking and
`bm` / `market_cap` / `sector` for the controls and constraints.

## Status of permission

The terminal licence treats genuinely derived data more permissively than raw
redistribution, but prior approval is commonly still required. Confirmation
was requested from the vendor; until it is on file, these CSVs are generated
locally and kept out of the repository (see .gitignore).
