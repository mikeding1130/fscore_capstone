"""Evaluation layer: hold-period backtest, performance metrics, and the
comparison of a deterministic portfolio against the random-basket
distribution (percentile ranks / p-values)."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def portfolio_returns(prices: pd.DataFrame, weights: pd.Series,
                      start: str, end: str) -> pd.Series:
    """Daily portfolio returns for a fixed-weight basket over [start, end].

    `prices` is the canonical long frame; weights indexed by ticker.
    (Annual rebalance = call this per holding year and chain the results.)
    """
    px = (prices[prices.ticker.isin(weights.index)]
          .pivot(index="date", columns="ticker", values="adj_close")
          .loc[start:end])
    rets = px.pct_change().dropna(how="all")
    return (rets * weights.reindex(rets.columns)).sum(axis=1)


def metrics(daily: pd.Series, rf_annual: float = 0.0) -> dict:
    """Standard performance metrics on a daily return series."""
    if len(daily) == 0:
        return {}
    cum = float((1 + daily).prod() - 1)
    ann = float((1 + cum) ** (TRADING_DAYS / len(daily)) - 1)
    vol = float(daily.std() * np.sqrt(TRADING_DAYS))
    sharpe = (ann - rf_annual) / vol if vol > 0 else np.nan
    nav = (1 + daily).cumprod()
    mdd = float((nav / nav.cummax() - 1).min())
    return {"cum_return": cum, "ann_return": ann, "ann_vol": vol,
            "sharpe": sharpe, "max_drawdown": mdd}


def turnover(prev_w: pd.Series, new_w: pd.Series) -> float:
    """One-way turnover between consecutive rebalances."""
    all_ix = prev_w.index.union(new_w.index)
    return float(np.abs(new_w.reindex(all_ix, fill_value=0)
                        - prev_w.reindex(all_ix, fill_value=0)).sum() / 2)


def vs_random(stat_fscore: float, stats_random: list[float],
              higher_is_better: bool = True) -> dict:
    """Place a deterministic portfolio's statistic inside the Monte Carlo
    distribution. Returns percentile rank and a one-sided p-value."""
    arr = np.asarray(stats_random, dtype=float)
    arr = arr[~np.isnan(arr)]
    if higher_is_better:
        pct = float((arr < stat_fscore).mean())
        p = float((arr >= stat_fscore).mean())
    else:
        pct = float((arr > stat_fscore).mean())
        p = float((arr <= stat_fscore).mean())
    return {"percentile": pct, "p_value": p, "n_draws": int(arr.size),
            "random_mean": float(arr.mean()), "random_std": float(arr.std())}
