# Data directory (git-ignored)

Place per-market files here; loaders in `src/fscore/data/loaders.py` define
the canonical schemas (fundamentals: one row per ticker-fiscal_year;
prices: one row per ticker-date).

Gating checks before backtesting a market:
1. Delisted firms present in the universe (survivorship).
2. True report/available dates for fundamentals (point-in-time).
