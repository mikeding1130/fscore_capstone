# Panel provenance

What each column is, where it came from, and why it can be published.

| Column | Source | Nature |
|---|---|---|
| `ticker`, `score_year`, `fiscal_year` | identifiers | not data |
| `f_roa` … `f_dturn` (nine flags) | computed here by `fscore.signal.piotroski` | **irreversible**: each is one bit, the outcome of a comparison (e.g. ROA > 0). The underlying value cannot be recovered. |
| `fscore` | sum of the nine flags | further aggregation, 0–9 |
| `sector` | Yahoo Finance classification | not vendor data |
| `bm_rank` | rank by book-to-market within the score year, 1 = highest | **irreversible**: an ordering, not a level. Book value and market capitalisation cannot be recovered from it. Shipped because the study uses B/M only as an order — the top 40% forms the value universe, the top k forms the value control. |

**Statement sources.** The US flags come from SEC EDGAR XBRL — public filings,
no vendor involved, rebuildable end to end by `scripts/fetch_us_edgar.py`. The
Japanese flags come from the Bloomberg statements under `data/processed/Japan/`
(git-ignored, never redistributed), rebuilt into a canonical frame by
`scripts/build_japan_bbg.py`; the facts underneath are the statutory 有価証券
報告書, which any reader can recompute from independently.

**No continuous per-security value is exported.** `bm` and `market_cap` are
ours rather than the vendor's, but they are still per-security numbers, so
they are recomputed at load time from the caches that
`scripts/fetch_us_edgar.py`, `scripts/fetch_us_japan.py` and
`scripts/build_japan_bbg.py` rebuild; shipping them would add nothing but
exposure. Raw statement values never leave the machine at all.

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

With this panel and freely re-fetchable prices, a reader can rebuild the study
without any licensed input:

1. **Universe** — the panel's rows for a score year *are* that year's index
   constituents, because the panel is built after the membership filter. Median
   dollar volume, the liquidity screen, comes from public prices.
2. **Value subset** — take the top 40% by `bm_rank` within the rebuilt
   universe. `bm_rank` is an ordering, so restricting it to a subset preserves
   the order.
3. **Baskets** — top k by `fscore`, ties broken by the seeded shuffle in
   `fscore.selection.baskets.rank_by_fscore` (seed = 42 + formation year).
4. **Weights and evaluation** — `fscore.construction` and
   `fscore.evaluation` need only prices and the sector labels shipped here.


`results/panel/*.csv` plus the caches rebuilt by `scripts/fetch_us_edgar.py`,
`scripts/fetch_us_japan.py` and `scripts/build_japan_bbg.py` are enough to
re-run every notebook. The
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
