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

    Returns annualized alpha, its t-stat/p-value, betas, and R^2.
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
        "beta_mkt": float(res.params["Mkt-RF"]),
        "beta_smb": float(res.params["SMB"]),
        "beta_hml": float(res.params["HML"]),
        "r2": float(res.rsquared),
        "n_obs": int(res.nobs),
    }


def benchmark_returns(bench_prices: pd.DataFrame, ticker: str,
                      start=None, end=None) -> pd.Series:
    """Daily buy-and-hold returns of one benchmark from the cached long frame."""
    px = (bench_prices[bench_prices.ticker == ticker]
          .set_index("date")["adj_close"].sort_index())
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
