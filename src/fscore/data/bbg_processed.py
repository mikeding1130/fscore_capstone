"""Adapter for the Bloomberg-sourced `data/processed/{USA,Japan}/` tree.

This supersedes `fs_clean.py`. The tree is organised by *universe year* rather
than by firm: `financials/{MKT}_financials_{t,t_minus_1,t_minus_2}.xlsx` each
hold one sheet per year 2000-2025, and sheet `2012` of the `t_minus_1` file is
fiscal 2011 **for the firms that were in the index in 2012**. That layout is a
point-in-time universe by construction - the constituent list is whatever the
sheet contains - which is what `index/historical_constituents/{year}.xlsx`
repeats in longer form.

Three columns are present but empty in every sheet of both markets, so the
loader does not pretend otherwise (see `field_coverage`):

  * `Book_Value` - book equity comes from `Common_Shareholders_Equity`
    instead, which is populated ~99% of the time.
  * `Historical_Market_Cap` - market capitalisation has to be built from a
    price series and `Shares_Outstanding`; nothing here supplies it.
  * `Proceeds_Issuance_Common_Stock` - the equity-issuance cash-flow line this
    dataset was expected to provide. EQ_OFFER therefore still falls back to
    the year-on-year share count, and `piotroski_signals` records that it did.

`Gross_Profit` is populated for roughly 80% (US) and 85% (Japan) of firm-years,
and COGS is derived from it, so about one firm-year in six cannot produce a
complete nine-signal score. Those are dropped, not partially scored, and
counted in the exclusion report.

Prices are **not** read from this tree: `prices/{MKT}_adjusted_close_prices.xlsx`
contains no data at all (every date is the 1970 epoch and every cell is empty),
as its own `diagnostics/price_coverage.xlsx` records with
`Has_Any_Adjusted_Price = False` for all names. `load_prices_cached` maps the
Bloomberg tickers onto the existing Yahoo cache so the pipeline can still run,
and reports the coverage that mapping achieves.
"""
from __future__ import annotations

import pathlib
import re

import numpy as np
import pandas as pd

# market key -> directory name under data/processed
MARKET_DIR = {"us": "USA", "japan": "Japan"}
OFFSETS = {"t": 0, "t_minus_1": 1, "t_minus_2": 2}

# Bloomberg column -> canonical name used by the signal layer
COLUMNS = {
    "Net_Income": "net_income",
    "Sales_Revenue": "revenue",
    "Total_Assets": "total_assets",
    "Long_Term_Debt": "long_term_debt",
    "Current_Assets": "current_assets",
    "Current_Liabilities": "current_liabilities",
    "Operating_Cash_Flow": "cfo",
    "Shares_Outstanding": "shares_outstanding",
    "Common_Shareholders_Equity": "book_value",
    "Proceeds_Issuance_Common_Stock": "equity_issued",
}
# Columns the vendor supplies as headers but never fills, in either market.
KNOWN_EMPTY = ["Book_Value", "Historical_Market_Cap",
               "Proceeds_Issuance_Common_Stock"]

_PLACEHOLDER = re.compile(r"^[0-9]{7}[A-Z]$")


def _root(bbg: str) -> str | None:
    """`AAPL UW Equity` -> `AAPL`; `7203 JT Equity` -> `7203`.

    Bloomberg placeholder codes (`0203524D UN Equity`) identify a security that
    the index carried but that never resolved to a listing; they cannot be
    mapped to any price source and are returned as None so they are counted
    rather than silently carried.
    """
    s = str(bbg).replace(" Equity", "").strip()
    parts = s.split()
    if len(parts) < 2:
        return None
    root = parts[0]
    return None if _PLACEHOLDER.match(root) else root


def to_yahoo(bbg: str, market: str) -> str | None:
    """Bloomberg ticker -> the symbol used by the cached Yahoo prices."""
    root = _root(bbg)
    if root is None:
        return None
    return f"{root}.T" if market == "japan" else root


def _sheet(root: pathlib.Path, market: str, offset: str,
           year: int) -> pd.DataFrame:
    mkt = MARKET_DIR[market]
    path = root / "processed" / mkt / "financials" / f"{mkt}_financials_{offset}.xlsx"
    return pd.read_excel(path, sheet_name=str(year))


def load_fundamentals(market: str, data_dir: pathlib.Path,
                      years: list[int] | range) -> pd.DataFrame:
    """Canonical fundamentals for the formation `years`.

    The vendor aligns offset `t` to the *universe* year, so sheet 2012 holds
    fiscal 2012 — which is not public at a July 2012 formation. Scoring
    formation year y therefore needs sheet y-1, whose three offsets are fiscal
    y-1, y-2 and y-3: exactly the scoring year, the year to difference against,
    and the beginning-of-year assets behind it. One extra leading sheet is
    read for that reason.

    Because the universe is rebuilt every year, a firm appears once per year it
    was a constituent; identical (ticker, fiscal_year) rows arriving from two
    different formation years are de-duplicated, keeping the first.
    """
    yrs = list(years)
    frames = []
    for year in range(min(yrs) - 1, max(yrs) + 1):
        for offset in OFFSETS:
            d = _sheet(data_dir, market, offset, year)
            keep = {k: v for k, v in COLUMNS.items() if k in d.columns}
            out = d[list(keep)].rename(columns=keep).copy()
            out["ticker"] = [to_yahoo(t, market) for t in d.BBG_Ticker]
            out["bbg_ticker"] = d.BBG_Ticker.values
            out["fiscal_year"] = d.Financial_Year.values
            out["universe_year"] = year
            # Gross margin needs COGS; the vendor gives gross profit instead.
            out["cogs"] = (d.Sales_Revenue - d.Gross_Profit
                           if "Gross_Profit" in d.columns else np.nan)
            frames.append(out)

    f = pd.concat(frames, ignore_index=True)
    f = f[f.ticker.notna()]
    f = f.drop_duplicates(subset=["ticker", "fiscal_year"], keep="first")
    # `report_date` drives the point-in-time lag downstream. Bloomberg gives a
    # fiscal period end here, not a filing date, so the reporting lag is
    # applied on top of it exactly as it was for the Yahoo-sourced Japan data.
    f["report_date"] = pd.to_datetime(
        f.fiscal_year.astype(int).astype(str) + "-12-31")
    return f.reset_index(drop=True)


def constituents(market: str, data_dir: pathlib.Path,
                 years: list[int] | range) -> dict[int, set[str]]:
    """Index membership as of each formation year, from the financials sheets
    themselves (the `historical_constituents/` files repeat the same list).

    A year with no sheet is omitted rather than raised on: callers that probe
    for the widest feasible span legitimately ask about years beyond the
    workbook, and a missing year is an answer ("no membership known"), not an
    error. Requesting a specific year and getting nothing back is still
    visible — the key is simply absent.
    """
    out = {}
    for year in years:
        try:
            d = _sheet(data_dir, market, "t", year)
        except ValueError:                     # no worksheet for that year
            continue
        out[year] = {t for t in (to_yahoo(b, market) for b in d.BBG_Ticker)
                     if t is not None}
    return out


def field_coverage(market: str, data_dir: pathlib.Path,
                   years: list[int] | range) -> pd.DataFrame:
    """Per-year non-null share of every field the study needs.

    Reported rather than assumed: three of these columns are empty in every
    sheet, and one (`Gross_Profit`) is patchy enough to decide whether a firm
    can be scored at all.
    """
    rows = []
    for year in years:
        d = _sheet(data_dir, market, "t", year)
        r = {"universe_year": year, "names": len(d)}
        for col in list(COLUMNS) + ["Gross_Profit", "Book_Value",
                                    "Historical_Market_Cap"]:
            if col in d.columns:
                r[col] = round(float(d[col].notna().mean()), 3)
        rows.append(r)
    return pd.DataFrame(rows).set_index("universe_year")


def load_prices_cached(market: str, data_dir: pathlib.Path,
                       universe: set[str] | None = None
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prices for the Bloomberg universe, taken from the Yahoo cache.

    The vendor price workbook in this tree is empty, so there is nothing to
    read from it. Falling back to the cache is a documented compromise, not a
    silent substitution: the cache only holds symbols that are *still listed*,
    so the names it cannot supply are disproportionately the delisted ones —
    precisely the survivorship problem this dataset was meant to remove. The
    second returned frame reports, per formation year, how much of the
    universe was actually covered, so the gap stays visible.
    """
    px = pd.read_csv(data_dir / f"{'us' if market == 'us' else 'japan'}"
                     f"_prices.csv.gz", parse_dates=["date"])
    if universe is not None:
        px = px[px.ticker.isin(universe)]
    have = set(px.ticker.unique())
    cov = pd.DataFrame([{"ticker": t, "has_price": t in have}
                        for t in sorted(universe or have)])
    return px, cov


def vendor_prices_are_empty(market: str, data_dir: pathlib.Path) -> bool:
    """True when the vendor price workbook holds no data.

    Cheap structural check on the diagnostics file the vendor ships beside it,
    so callers can fail loudly instead of running on an empty panel.
    """
    mkt = MARKET_DIR[market]
    p = (data_dir / "processed" / mkt / "diagnostics" / "price_coverage.xlsx")
    if not p.exists():
        return False
    d = pd.read_excel(p)
    return bool(d.get("Has_Any_Adjusted_Price", pd.Series([True])).sum() == 0)
