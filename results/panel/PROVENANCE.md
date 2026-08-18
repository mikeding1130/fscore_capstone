# Panel provenance

What each column is, where it came from, and why it can be published.

| Column | Source | Nature |
|---|---|---|
| `ticker`, `score_year`, `fiscal_year` | identifiers | not data |
| `f_roa` … `f_dturn` (nine flags) | derived from licensed-terminal fundamentals | **irreversible**: each is one bit, the outcome of a comparison (e.g. ROA > 0). The underlying value cannot be recovered. |
| `fscore` | sum of the nine flags | further aggregation, 0–9 |
| `sector` | Yahoo Finance classification | not vendor data |

**No continuous per-security value is exported.** Three columns that earlier
drafts carried are excluded:

- `shares_outstanding` — a raw vendor figure. Nothing reads it: the EQ_OFFER
  flag was computed inside the source workbook from its own year-on-year
  share columns, so the flag is unaffected by its removal.
- `bm`, `market_cap` — ours rather than the vendor's, but still per-security
  numbers. They are recomputed at load time from the caches that
  `scripts/fetch_us_edgar.py` and `scripts/fetch_us_japan.py` rebuild from
  public sources, so shipping them would add nothing but exposure.

What remains is identifiers plus nine bits and their sum.

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

`results/panel/*.csv` plus the caches rebuilt by `scripts/fetch_us_edgar.py`
and `scripts/fetch_us_japan.py` are enough to re-run every notebook. The
study consumes `fscore` for ranking and `sector` for the sector constraint;
book-to-market and market capitalisation are joined from the rebuilt caches
when the panel is loaded, so no continuous value has to travel with it.

## Status of permission

The terminal is not accessible to the team, so a written confirmation cannot
be obtained. The panel was therefore reduced to the narrowest form that still
allows the study to be reproduced: identifiers and irreversible one-bit
flags, with every continuous value regenerated from public sources at run
time. The United States panel can additionally be rebuilt end to end from SEC
EDGAR alone (`scripts/fetch_us_edgar.py`), with no vendor input of any kind.
