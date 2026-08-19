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

    **Only complete scores are kept.** A firm-year is dropped unless all nine
    signals are present: a partial score is not a low score, and summing the
    signals that happen to be available would silently push those firms
    towards the bottom of the ranking (and into the short leg). The test is
    made directly on the nine signal columns rather than on the workbook's
    own availability flag, so the rule holds whatever the workbook says.

    Every exclusion is counted and written to {market}_score_exclusions.csv
    (see `exclusion_report`), so the report can state how much data was
    discarded and why.

    Cached as {market}_team_scores.csv; delete or pass refresh=True after
    the workbook is updated.
    """
    market = market.lower()
    d = Path(data_dir)
    cache = d / f"{market}_team_scores.csv"
    if cache.exists() and not refresh:
        return _attach_market_values(pd.read_csv(cache), market, d)

    wb = d / "processed" / WORKBOOKS[market]
    if not wb.exists():
        # No licensed workbook here — fall back to the published panel, which
        # carries only the irreversible 0/1 flags. Everything continuous is
        # recomputed below from the freely rebuildable caches, so this path
        # reproduces the study without redistributing vendor values.
        panel = _shipped_panel(market)
        if panel is None:
            raise FileNotFoundError(
                f"neither {wb} nor a published panel for {market!r} was found; "
                "run scripts/export_panel.py, or place the source workbook")
        return _attach_market_values(panel, market, d)

    xl = pd.ExcelFile(wb)
    rows, dropped = [], []
    for y in years:
        if str(y) not in xl.sheet_names:
            continue
        sheet = xl.parse(str(y))
        n_signals = sheet[SIGNAL_COLS].notna().sum(axis=1)
        incomplete = n_signals < len(SIGNAL_COLS)
        no_symbol = sheet.Resolved_Yahoo_Symbol.isna()
        no_score = sheet.F_Score.isna()
        keep = ~incomplete & ~no_symbol & ~no_score
        dropped.append({
            "score_year": y,
            "rows": len(sheet),
            "dropped_incomplete_signals": int(incomplete.sum()),
            "dropped_no_symbol": int((~incomplete & no_symbol).sum()),
            "dropped_no_score": int((~incomplete & ~no_symbol & no_score).sum()),
            "kept": int(keep.sum()),
            "median_signals_when_incomplete": (float(n_signals[incomplete].median())
                                               if incomplete.any() else float("nan")),
        })
        ok = sheet[keep]
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

    out.to_csv(cache, index=False)
    out = _attach_market_values(out, market, d)
    pd.DataFrame(dropped).to_csv(d / f"{market}_score_exclusions.csv", index=False)
    return out


def exclusion_report(market: str, data_dir: str | Path = "data",
                     by_year: bool = False) -> pd.DataFrame:
    """How much data the completeness rule discarded, for the write-up.

    Per score year: rows in the workbook, rows dropped because fewer than
    nine signals were available, rows dropped for a missing symbol or score,
    and rows kept. `by_year=False` returns the study totals.
    """
    p = Path(data_dir) / f"{market.lower()}_score_exclusions.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing — call load_team_scores(..., refresh=True) first")
    rep = pd.read_csv(p)
    if by_year:
        return rep.set_index("score_year")
    cols = ["rows", "dropped_incomplete_signals", "dropped_no_symbol",
            "dropped_no_score", "kept"]
    total = rep[cols].sum()
    total["pct_dropped_incomplete"] = round(
        100 * total.dropped_incomplete_signals / total.rows, 2)
    total["pct_kept"] = round(100 * total.kept / total.rows, 2)
    return total.to_frame(market.lower()).T


def _shipped_panel(market: str) -> pd.DataFrame | None:
    """The derived panel published with the repository, if present."""
    p = (Path(__file__).resolve().parents[3] / "results" / "panel"
         / f"{market}_fscore_panel.csv")
    return pd.read_csv(p) if p.exists() else None


def _attach_market_values(scores: pd.DataFrame, market: str,
                          data_dir: Path) -> pd.DataFrame:
    """Add book-to-market and market cap from the fundamentals cache.

    These are continuous per-security values, so they are never carried in
    the published panel — they are recomputed here from the caches that
    `scripts/fetch_*.py` rebuild from public sources. Coverage begins where
    the cache does (US FY2009 via EDGAR, Japan later via Yahoo); the gap is
    reported in the per-year diagnostics rather than filled.
    """
    if {"bm", "market_cap"}.issubset(scores.columns):
        return scores
    f = Path(data_dir) / f"{market}_fundamentals.csv"
    if not f.exists():
        return scores.assign(bm=np.nan, market_cap=np.nan)
    fund = pd.read_csv(f)
    bm = fund[["ticker", "fiscal_year", "book_value", "market_cap"]].copy()
    bm["bm"] = bm.book_value / bm.market_cap
    bm.loc[bm.market_cap <= 0, "bm"] = np.nan
    return scores.merge(bm[["ticker", "fiscal_year", "bm", "market_cap"]],
                        on=["ticker", "fiscal_year"], how="left")


def sectors_from_scores(scores: pd.DataFrame) -> pd.Series:
    """ticker -> sector map (last observed label per ticker)."""
    return (scores.dropna(subset=["sector"])
                  .drop_duplicates("ticker", keep="last")
                  .set_index("ticker")["sector"])
