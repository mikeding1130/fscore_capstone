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


def _sector_map(market: str, data_dir: Path) -> pd.Series:
    """Ticker -> sector, from the constituent file first and the scores
    workbook second. Neither covers every name on its own; names still
    missing fall into the optimiser's "Unknown" bucket, which the sector cap
    then treats as one group — reported rather than silently merged."""
    parts = []
    csv = data_dir / f"{market}_sectors.csv"
    if csv.exists():
        d = pd.read_csv(csv).dropna(subset=["sector"])
        parts.append(d.set_index("ticker")["sector"])
    wb = data_dir / "processed" / SCORE_WORKBOOKS[market]
    if wb.exists():
        xl = pd.ExcelFile(wb)
        rows = []
        for sheet in (s for s in xl.sheet_names if s.isdigit()):
            d = xl.parse(sheet)[["Resolved_Yahoo_Symbol", "Yahoo_Sector"]].dropna()
            rows.append(d.rename(columns={"Resolved_Yahoo_Symbol": "ticker",
                                          "Yahoo_Sector": "sector"}))
        if rows:
            d = pd.concat(rows).drop_duplicates("ticker", keep="last")
            parts.append(d.set_index("ticker")["sector"])
    if not parts:
        return pd.Series(dtype=object)
    return pd.concat(parts).groupby(level=0).first()


def load_scores(market: str, data_dir: str | Path = "data",
                refresh: bool = False) -> pd.DataFrame:
    """The study's score panel, computed by our own signal code.

    This is the single entry point the grid and the export script use, so
    both sit on the same implementation instead of one reading precomputed
    flags. Adds the sector label and rejoins book-to-market and market cap
    from the fundamentals cache; results are cached as
    {market}_fsclean_scores.csv.
    """
    market = market.lower()
    d = Path(data_dir)
    cache = d / f"{market}_fsclean_scores.csv"
    if cache.exists() and not refresh:
        return pd.read_csv(cache)

    prices = pd.read_csv(d / f"{market}_prices.csv.gz", parse_dates=["date"])
    panel = scores_from_fs_clean(market, d, prices=prices)
    panel["sector"] = panel.ticker.map(_sector_map(market, d))

    from .team_scores import _attach_market_values
    panel = _attach_market_values(panel, market, d)
    panel.to_csv(cache, index=False)
    _write_exclusions(market, d, panel)
    return panel


def _write_exclusions(market: str, data_dir: Path, panel: pd.DataFrame) -> None:
    """Account for every firm-year the source held but the study does not use.

    Three things remove rows, and they are counted separately because they
    mean different things: an identifier that never resolved to a tradable
    symbol, a year without the t-1 row the deltas need, and a year whose nine
    signals were not all computable. Written to {market}_fsclean_exclusions.csv.
    """
    xl = pd.ExcelFile(data_dir / "processed" / WORKBOOKS[market])
    mapping = _bbg_to_yahoo(market, data_dir)
    scored = panel.groupby("score_year").size()
    per_year = panel.attrs.get("per_year", pd.DataFrame())
    incomplete = (per_year.set_index("score_year")["dropped_incomplete_signals"]
                  if len(per_year) else pd.Series(dtype=int))

    rows = []
    for sheet in sorted(s for s in xl.sheet_names if s.isdigit()):
        year, raw = int(sheet), xl.parse(sheet)
        sym = raw.BBG_Ticker.map(mapping)
        need = sym.isna()
        if need.any():
            sym.loc[need] = raw.loc[need, "BBG_Ticker"].map(
                lambda b: _fallback_symbol(b, market))
        rows.append({
            "score_year": year,
            "rows_in_source": len(raw),
            "dropped_unresolved_identifier": int(sym.isna().sum()),
            "dropped_incomplete_signals": int(incomplete.get(year, 0)),
            "scored": int(scored.get(year, 0)),
        })
    d = pd.DataFrame(rows)
    # whatever is left over had no prior-year row to difference against
    d["dropped_no_prior_year"] = (d.rows_in_source
                                  - d.dropped_unresolved_identifier
                                  - d.dropped_incomplete_signals
                                  - d.scored).clip(lower=0)
    d.to_csv(data_dir / f"{market}_fsclean_exclusions.csv", index=False)


def exclusion_report(market: str, data_dir: str | Path = "data",
                     by_year: bool = False) -> pd.DataFrame:
    """How much of the source the study discards, and for which reason."""
    p = Path(data_dir) / f"{market.lower()}_fsclean_exclusions.csv"
    if not p.exists():
        load_scores(market, data_dir, refresh=True)
    rep = pd.read_csv(p)
    if by_year:
        return rep.set_index("score_year")
    total = rep.drop(columns=["score_year"]).sum()
    total["pct_scored"] = round(100 * total.scored / total.rows_in_source, 2)
    return total.to_frame(market.lower()).T
