"""Adapter for the team's `*_FS_clean.xlsx` statements (FY2000-2025).

These workbooks carry the raw line items the nine Piotroski signals need, for
both markets, over the whole sample — which the earlier per-market feeds did
not. Wiring them in buys two things:

  * **one scoring implementation.** Until now the grid consumed the flags
    precomputed inside the scores workbook while the main study computed its
    own, so the two disagreed on the equity-issuance convention. Feeding
    these statements through `piotroski_signals` puts every result on the
    same code, and that code is unit-tested.
  * **the full window for Japan's signals.** The Yahoo feed served roughly
    five annual periods; this reaches back to FY2000.

What it does **not** solve is book value. `FS_clean` has no equity line, so
the high-B/M value universe the main study forms still depends on the
fundamentals cache (US from FY2009 via EDGAR, Japan from FY2021 via Yahoo).
That gap is the one remaining reason the Japanese main study is shorter than
the American one; the grid study, which ranks inside the full universe rather
than a value subset, is unaffected.

Two conventions worth stating:

  * `report_date` is set to the fiscal year end (Dec-31 US, Mar-31 Japan),
    since the workbooks carry no filing date. Publication delay is then
    applied by `lag_months` at formation, exactly as elsewhere — 1 month is
    not enough here, so callers should pass the statutory lag (5 for the US
    convention used with period-end dates, 3 for Japan).
  * Bloomberg identifiers are mapped to Yahoo symbols through the scores
    workbook, which carries the resolved pair. Rows whose identifier never
    resolved (mostly Bloomberg internal IDs for renamed or delisted lines)
    are dropped and counted rather than guessed at.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

WORKBOOKS = {"us": "USA_FS_clean.xlsx", "japan": "Japan_FS_clean.xlsx"}
SCORE_WORKBOOKS = {"us": "USA_Fscores_nonfinancial.xlsx",
                   "japan": "Japan_Fscores_nonfinancial.xlsx"}

# FS_clean column -> canonical name
FIELDS = {
    "Net_Income": "net_income",
    "Sales_Revenue": "revenue",
    "Total_Assets": "total_assets",
    "Long_Term_Debt": "long_term_debt",
    "Current_Assets": "current_assets",
    "Current_Liabilities": "current_liabilities",
    "Operating_Cash_Flow": "cfo",
    "Shares_Outstanding": "shares_outstanding",
}
# fiscal year end by market convention (the workbooks carry no dates)
FY_END = {"us": (12, 31), "japan": (3, 31)}
FY_END_OFFSET = {"us": 0, "japan": 1}   # Japan's FY2015 ends 2016-03-31


def _bbg_to_yahoo(market: str, data_dir: Path) -> dict[str, str]:
    """Bloomberg identifier -> Yahoo symbol, taken from the scores workbook."""
    path = data_dir / "processed" / SCORE_WORKBOOKS[market]
    if not path.exists():
        return {}
    xl = pd.ExcelFile(path)
    out: dict[str, str] = {}
    for sheet in (s for s in xl.sheet_names if s.isdigit()):
        d = xl.parse(sheet)[["BBG_Ticker", "Resolved_Yahoo_Symbol"]].dropna()
        out.update(dict(zip(d.BBG_Ticker, d.Resolved_Yahoo_Symbol)))
    return out


def _fallback_symbol(bbg: str, market: str) -> str | None:
    """Mechanical mapping for identifiers the workbook never resolved.

    Only the plain forms are attempted — '7203 JT Equity' -> '7203.T',
    'CRM UN Equity' -> 'CRM'. Bloomberg internal IDs ('1697067D UN Equity')
    carry no exchange ticker and are left unmapped.
    """
    head = str(bbg).split()[0]
    if re.fullmatch(r"\d{6,}[A-Z]?", head):
        return None                      # internal id, not a ticker
    if market == "japan":
        return f"{head}.T" if head.isdigit() else None
    return head.replace("/", "-") if re.fullmatch(r"[A-Z.\-/]+", head) else None


def load_fs_clean(market: str, data_dir: str | Path = "data",
                  prices: pd.DataFrame | None = None) -> pd.DataFrame:
    """Canonical annual fundamentals from `{market}_FS_clean.xlsx`.

    `book_value` is absent from the source and returned as NaN; `market_cap`
    is attached from the price cache when one is supplied.
    """
    market = market.lower()
    d = Path(data_dir)
    xl = pd.ExcelFile(d / "processed" / WORKBOOKS[market])
    mapping = _bbg_to_yahoo(market, d)

    frames, unmapped = [], set()
    for sheet in (s for s in xl.sheet_names if s.isdigit()):
        raw = xl.parse(sheet)
        out = pd.DataFrame({canon: raw[src] for src, canon in FIELDS.items()})
        out["fiscal_year"] = int(sheet)
        # cost of sales is not carried directly; gross profit gives it
        out["cogs"] = raw["Sales_Revenue"] - raw["Gross_Profit"]
        sym = raw.BBG_Ticker.map(mapping)
        need = sym.isna()
        if need.any():
            sym.loc[need] = raw.loc[need, "BBG_Ticker"].map(
                lambda b: _fallback_symbol(b, market))
        unmapped |= set(raw.loc[sym.isna(), "BBG_Ticker"])
        out["ticker"] = sym
        frames.append(out.dropna(subset=["ticker"]))

    f = pd.concat(frames, ignore_index=True)
    month, day = FY_END[market]
    f["report_date"] = pd.to_datetime(
        {"year": f.fiscal_year + FY_END_OFFSET[market],
         "month": month, "day": day})
    f["book_value"] = np.nan          # no equity line in this source
    f["market_cap"] = np.nan
    f = (f.sort_values(["ticker", "fiscal_year"])
           .drop_duplicates(["ticker", "fiscal_year"], keep="last")
           .reset_index(drop=True))

    if prices is not None:
        from .yahoo import attach_market_cap
        f = attach_market_cap(f.drop(columns=["market_cap"]), prices)

    f.attrs["unmapped_identifiers"] = len(unmapped)
    f.attrs["rows"] = len(f)
    return f


def scores_from_fs_clean(market: str, data_dir: str | Path = "data",
                         years: range = range(2002, 2026),
                         prices: pd.DataFrame | None = None,
                         sectors: pd.Series | None = None) -> pd.DataFrame:
    """Tidy score panel computed by *our* signal code from these statements.

    Same shape as `team_scores.load_team_scores`, so the grid study consumes
    it unchanged — but every flag here comes from `piotroski_signals`, which
    prefers the cash-flow issuance measure and drops firm-years whose nine
    signals are not all computable.
    """
    from ..signal.piotroski import SIGNALS, piotroski_signals

    fund = load_fs_clean(market, data_dir, prices=prices)
    rows, dropped = [], []
    for year in years:
        snap = fund[fund.fiscal_year.isin([year, year - 1, year - 2])]
        if snap.empty:
            continue
        scored = piotroski_signals(snap, year=year)
        if scored.empty:
            continue
        scored = scored.rename(columns={s: s for s in SIGNALS})
        scored["score_year"] = year
        scored["fiscal_year"] = year
        rows.append(scored)
        dropped.append({"score_year": year,
                        "scored": len(scored),
                        "dropped_incomplete_signals":
                            scored.attrs.get("dropped_incomplete", 0),
                        "eq_offer_from_cashflow":
                            scored.attrs.get("eq_offer_from_cashflow", 0)})

    panel = pd.concat(rows, ignore_index=True)
    # match the column names the rest of the study expects
    panel = panel.rename(columns={
        "roa_pos": "f_roa", "cfo_pos": "f_cfo", "delta_roa_pos": "f_droa",
        "accruals_ok": "f_accrual", "delta_leverage_down": "f_dlever",
        "delta_liquidity_up": "f_dliquid", "no_issuance": "f_eq_offer",
        "delta_margin_up": "f_dmargin", "delta_turnover_up": "f_dturn"})
    if sectors is not None:
        panel["sector"] = panel.ticker.map(sectors)
    panel.attrs["per_year"] = pd.DataFrame(dropped)
    panel.attrs["unmapped_identifiers"] = fund.attrs["unmapped_identifiers"]
    return panel
