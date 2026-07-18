"""Multi-year study runner: the proposal's full loop for one market.

For each formation year T (portfolios formed July 1 of year T):

  1. point-in-time universe  — names with continuous listing and liquidity
     history up to the formation date, and fiscal-year T-1 statements whose
     `available_date` (report_date + reporting lag) precedes formation;
  2. high-B/M value subset   — top `value_quantile` of book-to-market;
  3. nine-signal F-Score     — computed within the value subset only;
  4. selection @ fixed k     — F-Score top-k vs value / market-cap /
     liquidity-matched controls vs a Monte-Carlo random distribution;
  5. construction            — EW / long-only GMV / sector-capped GMV on an
     RMT-cleaned covariance estimated from the year before formation;
  6. hold July T .. June T+1, annual rebalance; years are chained into one
     track record per strategy (random draw i chains with draw i).

All dates are point-in-time safe: nothing formed at T uses prices or
statements from after the formation date.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data.loaders import apply_reporting_lag, high_bm_subset
from .signal.piotroski import piotroski_signals
from .selection.baskets import (BASKET_SIZE, fscore_basket, liquidity_matched_basket,
                                mktcap_basket, random_baskets, value_basket)
from .construction.weights import (clean_rmt, equal_weight, gmv_weights,
                                   sector_constrained_gmv)
from .evaluation.backtest import metrics, turnover, vs_random

STRATEGIES = ["fscore_EW", "fscore_GMV", "fscore_GMVsec",
              "value_EW", "mktcap_EW", "liquidity_EW"]


# ----------------------------------------------------------------------
# point-in-time building blocks
# ----------------------------------------------------------------------

def formation_date(year: int) -> pd.Timestamp:
    return pd.Timestamp(f"{year}-07-01")


def pit_snapshot(fundamentals: pd.DataFrame, year: int,
                 lag_months: int = 5) -> pd.DataFrame:
    """Statements usable at the July-1 formation of `year`: fiscal years T-1
    (must be public, i.e. available_date <= formation) and T-2 (the delta
    base). Firms missing either year drop out."""
    f = apply_reporting_lag(fundamentals, lag_months=lag_months)
    fd = formation_date(year)
    t = f[(f.fiscal_year == year - 1) & (f.available_date <= fd)]
    tm1 = f[f.fiscal_year == year - 2]
    common = set(t.ticker) & set(tm1.ticker)
    snap = pd.concat([t[t.ticker.isin(common)], tm1[tm1.ticker.isin(common)]])
    return snap.reset_index(drop=True)


def build_universe(prices: pd.DataFrame, snapshot: pd.DataFrame, year: int,
                   n: int = 150, min_days: int = 200) -> pd.DataFrame:
    """Eligible universe at formation: continuous listing over the prior year,
    usable fundamentals, ranked by median daily dollar volume (top `n`).
    Returns fiscal T-1 fundamentals + [adv, bm] for eligible names."""
    fd = formation_date(year)
    est = prices[(prices.date >= fd - pd.DateOffset(years=1)) & (prices.date < fd)]
    g = est.groupby("ticker")
    stats = pd.DataFrame({
        "days": g.size(),
        "adv": (est.assign(dv=est.adj_close * est.volume)
                   .groupby("ticker")["dv"].median()),
        "last_date": g["date"].max(),
    })
    alive = stats[(stats.days >= min_days)
                  & (stats.last_date >= fd - pd.Timedelta(days=10))]

    t = snapshot[snapshot.fiscal_year == year - 1]  # fiscal T-1 rows
    need = ["total_assets", "net_income", "cfo", "current_assets",
            "current_liabilities", "shares_outstanding", "revenue", "cogs",
            "book_value", "market_cap"]
    t = t.dropna(subset=need)
    t = t[t.market_cap > 0]

    uni = t[t.ticker.isin(alive.index)].copy()
    uni["adv"] = uni.ticker.map(alive.adv)
    uni = uni.sort_values("adv", ascending=False).head(n)
    uni["bm"] = uni.book_value / uni.market_cap
    return uni.reset_index(drop=True)


def holding_returns(prices: pd.DataFrame, tickers: list[str],
                    start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Wide daily-return frame for `tickers` over (start, end]."""
    px = (prices[prices.ticker.isin(tickers)
                 & (prices.date >= start - pd.Timedelta(days=7))
                 & (prices.date <= end)]
          .pivot(index="date", columns="ticker", values="adj_close")
          .sort_index())
    rets = px.pct_change()
    return rets.loc[rets.index > start].dropna(how="all")


# ----------------------------------------------------------------------
# one formation year
# ----------------------------------------------------------------------

@dataclass
class YearResult:
    year: int
    universe: pd.DataFrame
    scored: pd.DataFrame
    baskets: dict
    weights: dict          # strategy -> pd.Series
    daily: pd.DataFrame    # columns = STRATEGIES
    mc_daily: dict         # construction -> DataFrame (cols = draw indices)
    diagnostics: dict = field(default_factory=dict)


def run_year(fundamentals, prices, sectors, year, *, k=BASKET_SIZE,
             universe_size=150, value_quantile=0.4, n_mc=1000,
             n_mc_opt=300, lag_months=5, seed=42) -> YearResult:
    fd = formation_date(year)
    hold_end = fd + pd.DateOffset(years=1) - pd.Timedelta(days=1)

    snap = pit_snapshot(fundamentals, year, lag_months=lag_months)
    uni = build_universe(prices, snap, year, n=universe_size)
    value_set = high_bm_subset(uni, quantile=value_quantile)

    scored = piotroski_signals(snap, year=year - 1)
    scored = scored[scored.ticker.isin(value_set.ticker)]
    scored = scored.merge(value_set[["ticker", "bm", "market_cap", "adv"]], on="ticker")

    k_eff = min(k, len(scored))
    baskets = {
        "fscore": fscore_basket(scored, k=k_eff),
        "value": value_basket(value_set, k=k_eff),
        "mktcap": mktcap_basket(value_set, k=k_eff),
    }
    baskets["liquidity"] = liquidity_matched_basket(
        value_set, baskets["fscore"], liquidity_col="adv", k=k_eff, seed=seed)
    mc = random_baskets(value_set.ticker.tolist(), k=k_eff,
                        n_draws=n_mc, seed=seed + year)

    # estimation window: the year *before* formation (point-in-time)
    est = holding_returns(prices, value_set.ticker.tolist(),
                          fd - pd.DateOffset(years=1), fd - pd.Timedelta(days=1))
    hold = holding_returns(prices, value_set.ticker.tolist(), fd, hold_end)

    def weights_for(basket: list[str], how: str) -> pd.Series:
        cols = [c for c in basket if c in est.columns]
        if how == "EW" or len(cols) < 5:
            return equal_weight(cols if cols else basket)
        cov = clean_rmt(est[cols], detone=True)
        if how == "GMV":
            return gmv_weights(cov, cols)
        return sector_constrained_gmv(cov, cols, sectors)

    weights = {
        "fscore_EW": weights_for(baskets["fscore"], "EW"),
        "fscore_GMV": weights_for(baskets["fscore"], "GMV"),
        "fscore_GMVsec": weights_for(baskets["fscore"], "GMVsec"),
        "value_EW": weights_for(baskets["value"], "EW"),
        "mktcap_EW": weights_for(baskets["mktcap"], "EW"),
        "liquidity_EW": weights_for(baskets["liquidity"], "EW"),
    }

    def port_ret(w: pd.Series) -> pd.Series:
        cols = [c for c in w.index if c in hold.columns]
        ww = w.reindex(cols)
        ww = ww / ww.sum()
        return (hold[cols].fillna(0.0) * ww).sum(axis=1)

    daily = pd.DataFrame({name: port_ret(w) for name, w in weights.items()})

    # Monte-Carlo control through the identical construction pipeline
    mc_daily: dict[str, pd.DataFrame] = {}
    ew_cols = {}
    for i, b in enumerate(mc):
        ew_cols[i] = port_ret(equal_weight(b))
    mc_daily["EW"] = pd.DataFrame(ew_cols)
    for how in ("GMV", "GMVsec"):
        cols_out = {}
        for i, b in enumerate(mc[:n_mc_opt]):
            cols_out[i] = port_ret(weights_for(b, how))
        mc_daily[how] = pd.DataFrame(cols_out)

    diag = {"universe": len(uni), "value_set": len(value_set),
            "scored": len(scored), "k": k_eff,
            "fscore_mean": float(scored.fscore.mean()),
            "fscore_basket_min": int(scored[scored.ticker.isin(baskets["fscore"])].fscore.min())
            if k_eff else np.nan}
    return YearResult(year, uni, scored, baskets, weights, daily, mc_daily, diag)


# ----------------------------------------------------------------------
# the chained multi-year study
# ----------------------------------------------------------------------

@dataclass
class StudyResult:
    market: str
    years: list
    yearly: list                  # list[YearResult]
    daily: pd.DataFrame           # chained, columns = STRATEGIES
    mc_daily: dict                # construction -> chained DataFrame

    def summary(self, rf_annual: float = 0.0) -> pd.DataFrame:
        rows = {s: metrics(self.daily[s].dropna(), rf_annual)
                for s in self.daily.columns}
        return pd.DataFrame(rows).T

    def mc_summary(self, construction: str = "EW") -> pd.DataFrame:
        mc = self.mc_daily[construction]
        return pd.DataFrame({i: metrics(mc[i].dropna()) for i in mc.columns}).T

    def placement(self, strategy: str = "fscore_EW",
                  construction: str = "EW") -> pd.DataFrame:
        """F-Score portfolio vs the random distribution, per metric."""
        stat = metrics(self.daily[strategy].dropna())
        dist = self.mc_summary(construction)
        rows = {}
        for m, hib in [("ann_return", True), ("sharpe", True),
                       ("max_drawdown", False)]:
            rows[m] = vs_random(stat[m], dist[m].tolist(), higher_is_better=hib)
            rows[m]["fscore"] = stat[m]
        return pd.DataFrame(rows).T[["fscore", "random_mean", "random_std",
                                     "percentile", "p_value", "n_draws"]]

    def turnover_table(self) -> pd.DataFrame:
        rows = []
        for prev, curr in zip(self.yearly, self.yearly[1:]):
            rows.append({"year": curr.year,
                         **{s: turnover(prev.weights[s], curr.weights[s])
                            for s in STRATEGIES}})
        return pd.DataFrame(rows).set_index("year")


def run_study(market: str, fundamentals, prices, sectors, years, **kw) -> StudyResult:
    yearly = [run_year(fundamentals, prices, sectors, y, **kw) for y in years]
    daily = pd.concat([yr.daily for yr in yearly]).sort_index()
    mc_daily = {how: pd.concat([yr.mc_daily[how] for yr in yearly]).sort_index()
                for how in yearly[0].mc_daily}
    return StudyResult(market, list(years), yearly, daily, mc_daily)
