# Data directory (git-ignored)

Cache files are written here by `python scripts/fetch_us_japan.py`:

| file | contents |
|---|---|
| `{market}_fundamentals.csv` | canonical annual statements + fiscal-year-end market cap |
| `{market}_prices.csv.gz` | adjusted daily closes + volume, 2022-01 onward |
| `{market}_sectors.csv` | constituent list with sector labels |
| `{market}_benchmarks.csv.gz` | investable benchmark ETFs (+ `JPY=X` for Japan) |

Canonical schemas are defined in `src/fscore/data/loaders.py`
(fundamentals: one row per ticker-fiscal_year; prices: one row per
ticker-date). All downstream code consumes only these frames, so a better
source swaps in without further changes.

Gating checks before backtesting a market on a *commercial* source
(both still open for US/Japan on the free Yahoo data — see README):
1. Delisted firms present in the universe (survivorship).
2. True report/available dates for fundamentals (point-in-time).

## Vietnam

Vietnam has no vendor API in this study. Its files come from the pipeline
in `src/fscore_vietnam`, which crawls FireAnt, CafeF and TCBS into
`fscore.db`, reconciles the three sources, applies accounting checks,
**scores the nine signals**, and writes a per-firm-year panel. That pipeline
is the source of record for every Vietnamese number here: both studies read
its score panel and neither recomputes it. Two files come from its
`run_grid_export.ipynb`; the rest are built by
`python src/fscore_vietnam/schema_adapter_util.py`, which adds only what that export
leaves out. The pipeline's own panels live in `data/vietnam_pipeline/`.

| file | contents | written by |
|---|---|---|
| `vietnam_prices.csv.gz` | dividend-adjusted daily closes + order-matching volume | `src/fscore_vietnam` |
| `vietnam_scores.csv` | score panel: nine flags, F-Score, B/M, sector | `src/fscore_vietnam` |
| `vietnam_fundamentals.csv` | FY-end book equity and market value — the B/M sort, nothing else | `schema_adapter_util.py` |
| `vietnam_sectors.csv` | ticker → sector | `schema_adapter_util.py` |
| `vietnam_benchmarks.csv.gz` | VN30 and VNINDEX levels, from `fscore.db` | `schema_adapter_util.py` |
| `vietnam_exclusions.csv` | what the source holds vs what is scored, by reason | `schema_adapter_util.py` |

Three things to know before reading any Vietnamese number:

1. **The reporting lag is 6 months**, not 5 or 3. `report_date` is the
   31 December fiscal year end and the panel carries no filing date, so
   +6 months lands on 30 June — the last day before a 1 July formation, and
   the same screening date the pipeline itself screens on.
2. **The benchmarks are capital indices.** VN30 and VNINDEX exclude cash
   dividends; the portfolios are built on dividend-adjusted closes. The gap
   (~1.5–2% a year) flatters every portfolio-vs-index row.
3. **The panel is already liquidity-screened.** A June-turnover tradability
   gate removes 5,296 of 23,493 firm-years before the study code sees them —
   a gate the US and Japan panels do not have. Vietnam scores 40.4% of its
   source rows against 83.0% (US) and 88.7% (Japan).

The two gating checks at the top of this file, applied to Vietnam:
delisted firms **are** partially present (125 of 1,371 tickers stop printing
before 2026, spread across 2012–2025), and report dates are fiscal period
ends rather than true filing dates — hence the conservative 6-month lag.

### A note on the file names

The score panel and the exclusion table are `{market}_scores.csv` and
`{market}_exclusions.csv` in every market. They used to be
`{market}_fsclean_*`, which read as a claim that every market came through an
`FS_clean.xlsx` workbook. Vietnam never did — its panel is exported by the
pipeline in `src/fscore_vietnam` and only *read* by `fs_clean.load_scores`,
which is why a missing Vietnamese cache used to die on a `KeyError` deep in
the workbook lookup. It now raises a message naming the script that rebuilds
it. The old names are still accepted on read, so an existing cache is not
silently invalidated; writes always use the new ones.
