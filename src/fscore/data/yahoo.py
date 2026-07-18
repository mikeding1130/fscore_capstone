"""Yahoo Finance adapter: real prices and annual fundamentals -> canonical frames.

Yahoo serves ~5 annual statement periods per name, which under the 5-month
reporting lag supports July-1 portfolio formations in 2023, 2024 and 2025 for
both the US (mostly Dec FYE) and Japan (mostly Mar FYE). Longer history (the
proposal's 2000-2025 sample) needs a commercial point-in-time source; it plugs
into the same canonical schemas without downstream changes.

Everything is cached as flat files under data/ (git-ignored) so notebooks and
scripts never re-hit the network once `scripts/fetch_us_japan.py` has run.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from .universe import constituents

PRICE_START = "2022-01-01"  # 1y of returns needed before the first (Jul-2023) formation

# canonical field -> ordered fallback keys in the yfinance statement frames
BS_FIELDS = {
    "total_assets": ["Total Assets"],
    "long_term_debt": ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"],
    "current_assets": ["Current Assets"],
    "current_liabilities": ["Current Liabilities"],
    "shares_outstanding": ["Ordinary Shares Number", "Share Issued"],
    "book_value": ["Stockholders Equity", "Common Stock Equity"],
}
INC_FIELDS = {
    "net_income": ["Net Income", "Net Income Common Stockholders"],
    "revenue": ["Total Revenue", "Operating Revenue"],
    "cogs": ["Cost Of Revenue"],
}
CF_FIELDS = {
    "cfo": ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"],
}

BENCHMARKS = {
    # market index ETF, value-factor ETF (+ FX series for USD conversion where needed)
    "us": ["SPY", "VTV"],
    "japan": ["1306.T", "EWJV", "JPY=X"],
}


def _pick(frame: pd.DataFrame, keys: list[str], col) -> float:
    for k in keys:
        if k in frame.index:
            v = frame.at[k, col]
            if pd.notna(v):
                return float(v)
    return np.nan


def _fiscal_year(end: pd.Timestamp) -> int:
    """Label a fiscal period by its dominant calendar year:
    Dec-2023 -> FY2023 (US style), Mar-2024 -> FY2023 (Japan style)."""
    return end.year if end.month >= 6 else end.year - 1


def fetch_fundamentals(tickers: list[str], pause: float = 0.25,
                       progress_every: int = 25) -> pd.DataFrame:
    """Pull annual statements per ticker into the canonical fundamentals frame
    (market_cap attached later from prices). Failures are skipped, not fatal."""
    rows: list[dict] = []
    for i, tk in enumerate(tickers):
        if progress_every and i % progress_every == 0:
            print(f"  fundamentals {i}/{len(tickers)}", flush=True)
        try:
            t = yf.Ticker(tk)
            bs, inc, cf = t.balance_sheet, t.income_stmt, t.cash_flow
            if bs is None or bs.empty or inc is None or inc.empty:
                continue
            for col in bs.columns:
                if not isinstance(col, pd.Timestamp):
                    continue
                row = {"ticker": tk, "report_date": col,
                       "fiscal_year": _fiscal_year(col)}
                for f, keys in BS_FIELDS.items():
                    row[f] = _pick(bs, keys, col)
                for f, keys in INC_FIELDS.items():
                    row[f] = _pick(inc, keys, col) if col in inc.columns else np.nan
                if np.isnan(row["cogs"]) and col in inc.columns:
                    gp = _pick(inc, ["Gross Profit"], col)
                    if not np.isnan(gp) and not np.isnan(row["revenue"]):
                        row["cogs"] = row["revenue"] - gp
                for f, keys in CF_FIELDS.items():
                    row[f] = _pick(cf, keys, col) if (cf is not None and col in cf.columns) else np.nan
                if np.isnan(row["long_term_debt"]):
                    row["long_term_debt"] = 0.0  # genuinely no LT debt reported
                rows.append(row)
        except Exception as e:  # network hiccups, delistings, odd statements
            print(f"  ! {tk}: {e}", flush=True)
        time.sleep(pause)
    out = pd.DataFrame(rows)
    if not out.empty:
        # one row per (ticker, fiscal_year): keep the latest-reported period
        out = (out.sort_values(["ticker", "fiscal_year", "report_date"])
                  .drop_duplicates(["ticker", "fiscal_year"], keep="last")
                  .reset_index(drop=True))
    return out


def fetch_prices(tickers: list[str], start: str = PRICE_START,
                 end: str | None = None, chunk: int = 80) -> pd.DataFrame:
    """Daily prices in long canonical form.

    Keeps both the dividend/split-adjusted close (`adj_close`, for returns)
    and the raw close (`close_raw`): market cap must be raw close x
    as-reported shares — adjusted prices are rescaled by later splits while
    filed share counts are not.
    """
    frames: list[pd.DataFrame] = []
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i + chunk]
        raw = yf.download(part, start=start, end=end, auto_adjust=False,
                          progress=False, group_by="column", threads=True)
        if raw.empty:
            continue
        if not isinstance(raw.columns, pd.MultiIndex):
            raw.columns = pd.MultiIndex.from_product([raw.columns, part])

        def melt(field, name):
            return (raw[field].rename_axis("date").reset_index()
                    .melt(id_vars="date", var_name="ticker", value_name=name))

        adj = melt("Adj Close", "adj_close")
        cls = melt("Close", "close_raw")
        vol = melt("Volume", "volume")
        frames.append(adj.merge(cls, on=["date", "ticker"], how="left")
                         .merge(vol, on=["date", "ticker"], how="left"))
        time.sleep(0.5)
    px = pd.concat(frames, ignore_index=True).dropna(subset=["adj_close"])
    px["date"] = pd.to_datetime(px["date"]).dt.tz_localize(None)
    return px.sort_values(["ticker", "date"]).reset_index(drop=True)


def attach_market_cap(fundamentals: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """market_cap = shares_outstanding x close as of the fiscal period end
    (last trading day on or before report_date) — Piotroski's fiscal-year-end
    market value, used for the B/M sort.

    Uses the RAW close when available: as-reported share counts pair with
    unadjusted prices (adjusted prices are rescaled by later splits)."""
    price_col = "close_raw" if "close_raw" in prices.columns else "adj_close"
    f = fundamentals.copy()
    # normalize both keys to ns resolution (pandas 3 merge_asof requires
    # identical datetime units)
    f["report_date"] = pd.to_datetime(f["report_date"]).astype("datetime64[ns]")
    px = prices.copy()
    px["date"] = pd.to_datetime(px["date"]).astype("datetime64[ns]")
    px = px.sort_values("date")
    merged = pd.merge_asof(
        f.sort_values("report_date"), px[["ticker", "date", price_col]],
        left_on="report_date", right_on="date", by="ticker",
        direction="backward", tolerance=pd.Timedelta(days=14))
    merged["market_cap"] = merged["shares_outstanding"] * merged[price_col]
    return (merged.drop(columns=["date", price_col])
                  .sort_values(["ticker", "fiscal_year"]).reset_index(drop=True))


# ------------------------------ caching ------------------------------

def _paths(market: str, cache_dir: str | Path):
    d = Path(cache_dir)
    return (d / f"{market}_fundamentals.csv",
            d / f"{market}_prices.csv.gz",
            d / f"{market}_sectors.csv",
            d / f"{market}_benchmarks.csv.gz")


def fetch_and_cache(market: str, cache_dir: str | Path = "data",
                    max_names: int | None = None) -> None:
    """One-shot download for a market; writes the four cache files."""
    market = market.lower()
    fp, pp, sp, bp = _paths(market, cache_dir)
    fp.parent.mkdir(parents=True, exist_ok=True)

    members = constituents(market)
    if max_names:
        members = members.head(max_names)
    members[["ticker", "name", "sector"]].to_csv(sp, index=False)
    print(f"{market}: {len(members)} constituents", flush=True)

    prices = fetch_prices(members.ticker.tolist())
    prices.to_csv(pp, index=False, compression="gzip")
    print(f"{market}: prices {prices.shape}, "
          f"{prices.ticker.nunique()} tickers", flush=True)

    fund = fetch_fundamentals(members.ticker.tolist())
    fund = attach_market_cap(fund, prices)
    fund.to_csv(fp, index=False)
    print(f"{market}: fundamentals {fund.shape}, "
          f"{fund.ticker.nunique()} tickers", flush=True)

    bench = fetch_prices(BENCHMARKS[market])
    bench.to_csv(bp, index=False, compression="gzip")
    print(f"{market}: benchmarks {sorted(bench.ticker.unique())}", flush=True)


def load_cached(market: str, cache_dir: str | Path = "data"):
    """-> (fundamentals, prices, sectors: Series by ticker, benchmarks)."""
    fp, pp, sp, bp = _paths(market.lower(), cache_dir)
    for p in (fp, pp, sp, bp):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} missing — run scripts/fetch_us_japan.py first")
    fund = pd.read_csv(fp, parse_dates=["report_date"])
    prices = pd.read_csv(pp, parse_dates=["date"])
    sectors = pd.read_csv(sp).set_index("ticker")["sector"]
    bench = pd.read_csv(bp, parse_dates=["date"])
    return fund, prices, sectors, bench
