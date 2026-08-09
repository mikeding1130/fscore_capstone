"""Adapter for the team-computed F-Score workbooks (data/processed/).

`USA_Fscores_nonfinancial.xlsx` / `Japan_Fscores_nonfinancial.xlsx` hold one
sheet per score year with the nine Piotroski signals and composite score for
a curated non-financial universe (~70-80 names/year), computed from Bloomberg
fundamentals under the exact Piotroski (2000) conventions:

  * ROA and CFO scaled by *beginning-of-year* total assets;
  * accrual signal: CFO ratio > ROA (i.e. accruals < 0);
  * ΔROA uses t-2 assets; Δleverage uses average assets;
  * equity-issuance from the issuance cash-flow line, share-count fallback;
  * financial firms removed (Yahoo sector classification, audited).

This addresses two peer-review comments directly (signal conventions and the
financial-firm screen). The adapter flattens the year sheets into one tidy
frame keyed by (score_year, ticker) with the Yahoo-resolved symbol, sector
and share count, and joins book-to-market from the existing fundamentals
caches where available (B/M coverage is reported, not assumed).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

WORKBOOKS = {
    "us": "USA_Fscores_nonfinancial.xlsx",
    "japan": "Japan_Fscores_nonfinancial.xlsx",
}
SIGNAL_COLS = ["F_ROA", "F_CFO", "F_DROA", "F_ACCRUAL", "F_DLEVER",
               "F_DLIQUID", "F_EQ_OFFER", "F_DMARGIN", "F_DTURN"]


def load_team_scores(market: str, data_dir: str | Path = "data",
                     years: range = range(2002, 2026),
                     refresh: bool = False) -> pd.DataFrame:
    """Tidy frame: one row per (score_year, ticker) with fscore, signals,
    sector, shares, and B/M joined from the market's fundamentals cache.

    Cached as {market}_team_scores.csv; delete or pass refresh=True after
    the workbook is updated.
    """
    market = market.lower()
    d = Path(data_dir)
    cache = d / f"{market}_team_scores.csv"
    if cache.exists() and not refresh:
        return pd.read_csv(cache)

    xl = pd.ExcelFile(d / "processed" / WORKBOOKS[market])
    rows = []
    for y in years:
        if str(y) not in xl.sheet_names:
            continue
        sheet = xl.parse(str(y))
        ok = sheet[(sheet.Score_Available == True)  # noqa: E712
                   & sheet.Resolved_Yahoo_Symbol.notna()
                   & sheet.F_Score.notna()]
        for _, r in ok.iterrows():
            rows.append({
                "score_year": y,
                "ticker": r.Resolved_Yahoo_Symbol,
                "fscore": float(r.F_Score),
                **{c.lower(): float(r[c]) if pd.notna(r[c]) else np.nan
                   for c in SIGNAL_COLS},
                "fiscal_year": int(r.Fiscal_Year),
                "sector": r.get("Yahoo_Sector", np.nan),
                "shares_outstanding": r.get("Shares_Outstanding", np.nan),
            })
    out = pd.DataFrame(rows).drop_duplicates(["score_year", "ticker"])

    # book-to-market from the existing fundamentals cache (fiscal-year join);
    # coverage is partial for early Japan years and reported downstream
    fund = pd.read_csv(d / f"{market}_fundamentals.csv")
    bm = fund[["ticker", "fiscal_year", "book_value", "market_cap"]].copy()
    bm["bm"] = bm.book_value / bm.market_cap
    bm.loc[bm.market_cap <= 0, "bm"] = np.nan
    out = out.merge(bm[["ticker", "fiscal_year", "bm", "market_cap"]],
                    on=["ticker", "fiscal_year"], how="left")
    out.to_csv(cache, index=False)
    return out


def sectors_from_scores(scores: pd.DataFrame) -> pd.Series:
    """ticker -> sector map (last observed label per ticker)."""
    return (scores.dropna(subset=["sector"])
                  .drop_duplicates("ticker", keep="last")
                  .set_index("ticker")["sector"])
