"""Evaluation layer: hold-period backtest, performance metrics, and the
comparison of a deterministic portfolio against the random-basket
distribution (percentile ranks / p-values)."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# The single significance level used throughout the study, fixed in advance.
# Every test — Monte-Carlo placement and factor-regression alpha alike — is
# judged at 5% and nothing else: no 1% "highly significant" tier, no 10%
# "marginally significant" tier. A result with p = 0.06 is not significant.
ALPHA = 0.05


def returns_panel(prices: pd.DataFrame, tickers: list[str],
                  start, end, delisting_return: float = 0.0) -> pd.DataFrame:
    """Wide daily-return frame over (start, end], delisting-safe.

    Prices are forward-filled inside the window before differencing, which
    settles two things explicitly rather than by accident:

    * **Delisted names are carried at their most recent trading price.** A
      position that stops trading keeps its last quote, so it earns 0 from
      then on instead of vanishing from the portfolio. Dropping it instead
      would silently re-weight the book onto the survivors — a survivorship
      bias inside the holding year.
    * **Gaps no longer swallow returns.** Differencing a column with a hole
      in it yields NaN on both sides of the hole; zero-filling those then
      loses the move across the gap entirely (a name that gapped +9% over a
      missing session was booked at 0%). Forward-filling first attributes the
      whole move to the session when trading resumes.

    `delisting_return` books a one-off return on the first session after the
    last real trade (0.0 = full recovery at the last price, the default).
    Carrying at the last price assumes the position is liquidated at that
    quote; for performance-related delistings the realised recovery is far
    lower, so pass e.g. -0.30 to test that assumption.

    Counts of names carried after delisting are attached as
    `.attrs["delisted"]`.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    in_window = ((prices.date >= start - pd.Timedelta(days=7))
                 & (prices.date <= end))
    # The trading calendar comes from every name in the frame, not just the
    # selected ones: a delisted name has no rows of its own after it stops,
    # so without the market calendar there would be nothing to carry it onto.
    calendar = pd.DatetimeIndex(sorted(prices.loc[in_window, "date"].unique()))
    px = (prices[in_window & prices.ticker.isin(tickers)]
          .pivot(index="date", columns="ticker", values="adj_close")
          .sort_index())
    if px.empty or len(calendar) == 0:
        out = pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
        out.attrs["delisted"] = 0
        return out

    last_trade = px.apply(lambda c: c.last_valid_index())
    px = px.reindex(calendar).ffill()
    rets = px.pct_change().loc[px.index > start]

    delisted = 0
    if delisting_return:
        for tk, lt in last_trade.items():
            after = rets.index[rets.index > lt]
            if len(after):
                rets.loc[after[0], tk] = delisting_return
    for tk, lt in last_trade.items():
        if lt is not None and lt < calendar[-1]:
            delisted += 1
    rets = rets.dropna(how="all")
    rets.index.name = "date"
    rets.attrs["delisted"] = delisted
    rets.attrs["last_trade"] = last_trade
    return rets


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
    distribution.

    Returns the percentile rank, the one-sided empirical p-value, and
    `significant` — the verdict at the study's single level, ALPHA = 5%
    (p < 0.05). No other threshold is reported or implied.
    """
    arr = np.asarray(stats_random, dtype=float)
    arr = arr[~np.isnan(arr)]
    if higher_is_better:
        pct = float((arr < stat_fscore).mean())
        p = float((arr >= stat_fscore).mean())
    else:
        pct = float((arr > stat_fscore).mean())
        p = float((arr <= stat_fscore).mean())
    return {"percentile": pct, "p_value": p, "significant": bool(p < ALPHA),
            "alpha": ALPHA, "n_draws": int(arr.size),
            "random_mean": float(arr.mean()), "random_std": float(arr.std())}
