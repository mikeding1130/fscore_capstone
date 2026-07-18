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
