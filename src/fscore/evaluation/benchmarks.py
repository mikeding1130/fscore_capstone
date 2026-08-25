"""Benchmarking layer: investable alternatives and factor regressions.

Two of the proposal's three benchmark types live here (the third — the
Monte-Carlo random control — is in `backtest.vs_random`):

  * investable benchmarks: buy-and-hold market index ETF and a value-factor
    ETF, i.e. "can we beat what's actually buyable?"
  * factor regression: daily excess returns on the Fama-French three factors
    (market, size, value) with Newey-West errors — does alpha survive once
    known style exposures are stripped out?

Factor data: Ken French data library. US factors are US-dollar; the Japan
3-factor set is also computed in USD, so Japanese portfolio returns are
converted JPY->USD (via the cached JPY=X series) before regressing.
"""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd

from ..data.loaders import apply_reporting_lag
from .backtest import ALPHA, TRADING_DAYS, returns_panel

FF_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FF_FILES = {
    "us": "F-F_Research_Data_Factors_daily_CSV.zip",
    "japan": "Japan_3_Factors_Daily_CSV.zip",
}
_UA = {"User-Agent": "Mozilla/5.0 (capstone research)"}


def fetch_ff_factors(market: str) -> pd.DataFrame:
    """Daily FF3 factors as decimals, columns [Mkt-RF, SMB, HML, RF]."""
    import requests

    url = FF_BASE + FF_FILES[market.lower()]
    raw = requests.get(url, headers=_UA, timeout=60).content
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        text = z.read(z.namelist()[0]).decode("utf-8", errors="ignore")
    # locate the header row, then read until the first blank/annotation line
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if "Mkt-RF" in ln)
    body: list[str] = []
    for ln in lines[start + 1:]:
        if not ln.strip() or not ln.strip()[0].isdigit():
            break
        body.append(ln)
    df = pd.read_csv(io.StringIO("\n".join([lines[start]] + body)))
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"].astype(str).str.strip(), format="%Y%m%d")
    df = df.set_index("date").astype(float) / 100.0
    df.columns = [c.strip() for c in df.columns]
    return df


def factor_regression(daily: pd.Series, factors: pd.DataFrame) -> dict:
    """OLS of daily excess returns on [Mkt-RF, SMB, HML], Newey-West (5 lags).

    Returns annualized alpha, its t-stat/p-value, betas, R^2, and the verdict
    at the study's single significance level (ALPHA = 5%). Alphas with
    p >= 0.05 are reported as not significant — there is no "marginally
    significant" tier.
    """
    import statsmodels.api as sm

    df = pd.concat([daily.rename("port"), factors], axis=1, join="inner").dropna()
    y = df["port"] - df["RF"]
    X = sm.add_constant(df[["Mkt-RF", "SMB", "HML"]])
    res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    return {
        "alpha_daily": float(res.params["const"]),
        "alpha_annual": float(res.params["const"] * 252),
        "alpha_tstat": float(res.tvalues["const"]),
        "alpha_pvalue": float(res.pvalues["const"]),
        "alpha_significant": bool(res.pvalues["const"] < ALPHA),
        "alpha": ALPHA,
        "beta_mkt": float(res.params["Mkt-RF"]),
        "beta_smb": float(res.params["SMB"]),
        "beta_hml": float(res.params["HML"]),
        "r2": float(res.rsquared),
        "n_obs": int(res.nobs),
    }


def benchmark_returns(bench_prices: pd.DataFrame, ticker: str,
                      start=None, end=None) -> pd.Series:
    """Daily buy-and-hold returns of one benchmark from the cached long frame.

    Index ETFs do not move 40% in a day: isolated price points deviating that
    far from an 11-day rolling median are vendor bad ticks (e.g. Yahoo's
    1306.T series around a split) and are dropped before differencing.
    """
    px = (bench_prices[bench_prices.ticker == ticker]
          .set_index("date")["adj_close"].sort_index())
    med = px.rolling(11, center=True, min_periods=3).median()
    px = px[(px / med - 1).abs() < 0.4]
    if start is not None:
        px = px.loc[start:]
    if end is not None:
        px = px.loc[:end]
    return px.pct_change().dropna()


def to_usd(daily_local: pd.Series, fx_prices: pd.Series) -> pd.Series:
    """Convert local-currency daily returns to USD.

    `fx_prices` is JPY per USD (Yahoo 'JPY=X'), so USD return =
    (1 + r_local) * (fx_t-1 / fx_t) - 1.
    """
    fx = fx_prices.sort_index().reindex(daily_local.index).ffill()
    fx_ret = fx.shift(1) / fx - 1.0  # appreciation of JPY vs USD
    return ((1 + daily_local) * (1 + fx_ret) - 1).dropna()


# ----------------------------------------------------------------------
# Locally built factors, for markets Ken French does not cover
# ----------------------------------------------------------------------

def local_ff3_factors(fundamentals: pd.DataFrame, prices: pd.DataFrame,
                      years, lag_months: int = 6, rf_annual: float = 0.0,
                      min_names: int = 30) -> pd.DataFrame:
    """A Fama-French three-factor set built from the market's own panel.

    The Ken French library covers the US, Japan and the developed regions.
    It does **not** cover Vietnam, and no free substitute does, so the
    alternative to building the factors locally is to skip the regression for
    one of the three markets — which would leave the only emerging market in
    the study without the one test that strips style exposure out of the
    result. This builds the factors instead, from the same point-in-time
    panel the study already trusts.

    Construction follows the standard 2x3 sort, on this study's own calendar
    so the regression's factors and its portfolios are formed on the same
    day and from the same information set:

      * formed 1 July of each year in `years`, on fiscal year T-1 statements
        whose `report_date + lag_months` precedes formation;
      * size split at the median fiscal-year-end market cap;
      * book-to-market split at the 30th and 70th percentiles;
      * six value-weighted, buy-and-hold portfolios, weights set at formation
        and left to drift for twelve months (as the study's portfolios are);
      * SMB = mean(small) - mean(big); HML = mean(high B/M) - mean(low B/M);
      * `Mkt-RF` = value-weighted return of every eligible name, less RF.

    `RF` is constant at `rf_annual / 252` per day; the study's headline runs
    at rf = 0, and a factor set has to use the same rate its Sharpe does.

    Two differences from Ken French are worth stating. Market cap is the one
    at the fiscal year end rather than at the June formation, because that is
    the figure the panel carries and the one its B/M is built from. And the
    market leg is this panel's eligible universe, not the whole exchange, so
    it is a large-and-mid-cap market proxy rather than a total-market one.
    Both make the factors a local approximation; they are labelled as such
    wherever they are reported.
    """
    f = apply_reporting_lag(fundamentals, lag_months=lag_months)
    rf_daily = rf_annual / TRADING_DAYS
    legs: dict[str, list[pd.Series]] = {n: [] for n in
                                        ["SL", "SM", "SH", "BL", "BM", "BH", "MKT"]}

    for year in years:
        fd = pd.Timestamp(f"{year}-07-01")
        end = fd + pd.DateOffset(years=1) - pd.Timedelta(days=1)
        snap = f[(f.fiscal_year == year - 1) & (f.available_date <= fd)]
        snap = snap.dropna(subset=["book_value", "market_cap"])
        snap = snap[(snap.market_cap > 0) & (snap.book_value > 0)]
        if len(snap) < min_names:
            continue
        snap = snap.assign(bm=snap.book_value / snap.market_cap)

        hold = returns_panel(prices, snap.ticker.tolist(), fd, end)
        snap = snap[snap.ticker.isin(hold.columns)]
        if len(snap) < min_names or hold.empty:
            continue

        size_cut = snap.market_cap.median()
        lo, hi = snap.bm.quantile([0.30, 0.70])
        small = snap.market_cap <= size_cut
        bucket = np.where(snap.bm <= lo, "L", np.where(snap.bm >= hi, "H", "M"))
        label = pd.Series(np.where(small, "S", "B") + bucket,
                          index=snap.ticker.values)

        growth = (1.0 + hold.fillna(0.0)).cumprod()

        def vw(names) -> pd.Series | None:
            names = [t for t in names if t in growth.columns]
            if not names:
                return None
            w = snap.set_index("ticker").loc[names, "market_cap"]
            w = w / w.sum()
            value = (growth[names] * w).sum(axis=1)
            r = value.pct_change()
            r.iloc[0] = value.iloc[0] - 1.0
            return r

        for leg in ["SL", "SM", "SH", "BL", "BM", "BH"]:
            r = vw(label.index[label == leg])
            if r is not None:
                legs[leg].append(r)
        mkt = vw(snap.ticker.tolist())
        if mkt is not None:
            legs["MKT"].append(mkt)

    if not legs["MKT"]:
        raise ValueError("no formation year yielded a usable factor panel")

    cat = {k: (pd.concat(v).sort_index() if v else pd.Series(dtype=float))
           for k, v in legs.items()}
    df = pd.DataFrame(cat).sort_index()
    small = df[["SL", "SM", "SH"]].mean(axis=1)
    big = df[["BL", "BM", "BH"]].mean(axis=1)
    high = df[["SH", "BH"]].mean(axis=1)
    low = df[["SL", "BL"]].mean(axis=1)
    out = pd.DataFrame({
        "Mkt-RF": df["MKT"] - rf_daily,
        "SMB": small - big,
        "HML": high - low,
        "RF": rf_daily,
    }).dropna()
    out.index.name = "date"
    out.attrs["source"] = "locally constructed 2x3 sort on this study's panel"
    return out
