"""Build the grid's score panel from a canonical fundamentals frame.

The grid consumes a tidy panel - one row per (ticker, score_year) with the
nine flags, the composite, sector, book-to-market and market cap - which
`fs_clean.load_scores` used to be the only way to produce. That tied the grid
to the team workbook while the main study moved on: the US runs on SEC EDGAR
and Japan on the Bloomberg statements, so the grid was answering a robustness
question about a different dataset than the one being reported.

This builds the same panel from whatever canonical fundamentals a market
actually uses, so the grid and the main study rest on one source per market.
The signal computation is unchanged - `piotroski_signals`, the same code the
main study calls - and the flags are renamed to the `f_*` names the grid
expects.

`bm` and `market_cap` come from the fundamentals frame itself rather than a
separate cache, so a market whose market cap had to be rebuilt (Japan: raw
close x shares outstanding, since the vendor's column is empty) carries the
same values here as in the main study.
"""
from __future__ import annotations

import pandas as pd

from ..signal.piotroski import piotroski_signals

FLAG_NAMES = {
    "roa_pos": "f_roa", "cfo_pos": "f_cfo", "delta_roa_pos": "f_droa",
    "accruals_ok": "f_accrual", "delta_leverage_down": "f_dlever",
    "delta_liquidity_up": "f_dliquid", "no_issuance": "f_eq_offer",
    "delta_margin_up": "f_dmargin", "delta_turnover_up": "f_dturn",
}

PANEL_COLUMNS = ["ticker", *FLAG_NAMES.values(), "fscore", "score_year",
                 "fiscal_year", "sector", "bm", "market_cap"]


def build_score_panel(fund: pd.DataFrame, score_years,
                      sectors: pd.Series | None = None,
                      membership: dict[int, set[str]] | None = None
                      ) -> pd.DataFrame:
    """Score every firm-year in `score_years` from `fund`.

    `membership`, when given, is keyed by *formation* year and restricts each
    score year to the index constituents as of the formation that will use it
    (score year y-1 is used at formation y). Without it a firm scored in one
    year would remain in the panel for years it had already left the index.

    A firm-year appears only with a complete nine-signal score; the per-year
    counts of what was dropped, and of how EQ_OFFER was measured, are attached
    as `.attrs["per_year"]` for the exclusion report.
    """
    rows, per_year = [], []
    for year in score_years:
        snap = fund[fund.fiscal_year.isin([year, year - 1, year - 2])]
        if snap.empty:
            continue
        if membership is not None:
            members = membership.get(year + 1)      # formation that uses it
            if members is not None:
                snap = snap[snap.ticker.isin(members)]
        if snap.empty:
            continue
        scored = piotroski_signals(snap, year=year)
        if scored.empty:
            continue
        scored = scored.reset_index() if "ticker" not in scored.columns else scored
        scored["score_year"] = year
        scored["fiscal_year"] = year
        rows.append(scored)
        per_year.append({
            "score_year": year,
            "scored": len(scored),
            "dropped_incomplete_signals": scored.attrs.get("dropped_incomplete", 0),
            "eq_offer_from_cashflow": scored.attrs.get("eq_offer_from_cashflow", 0),
            "eq_offer_from_shares": scored.attrs.get("eq_offer_from_shares", 0),
            "no_tm2_assets": scored.attrs.get("no_tm2_assets", 0),
        })

    if not rows:
        return pd.DataFrame(columns=PANEL_COLUMNS)

    panel = pd.concat(rows, ignore_index=True).rename(columns=FLAG_NAMES)
    panel["sector"] = (panel.ticker.map(sectors) if sectors is not None
                       else pd.NA)

    # Book-to-market and market cap from the same frame the study scored, so
    # a rebuilt market cap is not silently replaced by a stale cached one.
    vals = fund[["ticker", "fiscal_year", "book_value", "market_cap"]].copy()
    vals = vals.drop_duplicates(subset=["ticker", "fiscal_year"], keep="first")
    vals["bm"] = vals.book_value / vals.market_cap
    vals.loc[vals.market_cap <= 0, "bm"] = pd.NA
    panel = panel.merge(vals[["ticker", "fiscal_year", "bm", "market_cap"]],
                        on=["ticker", "fiscal_year"], how="left")

    panel.attrs["per_year"] = pd.DataFrame(per_year)
    return panel[[c for c in PANEL_COLUMNS if c in panel.columns]]
