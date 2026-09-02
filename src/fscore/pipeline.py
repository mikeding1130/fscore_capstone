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
     RMT-denoised covariance estimated from the year before formation
     (detoning is out of scope — removing the market mode makes the matrix
     singular, so the minimum-variance solve would optimise residual risk
     only. The flag survives so the reason stays testable, but nothing in the
     study sets it);
  6. hold July T .. June T+1 — bought at formation and left to drift, with
     the ONLY rebalance at the next formation; turnover is measured from the
     drifted weights so the cost matches the trade actually made. Years are
     chained into one track record per strategy (random draw i chains with
     draw i).

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
                                mktcap_basket, random_baskets, rank_by_fscore,
                                tie_break_slots, value_basket)
from .construction.weights import (clean_rmt, equal_weight, gmv_weights,
                                   sector_constrained_gmv)
from .evaluation.backtest import metrics, returns_panel, turnover, vs_random
from .markets import SHORT_BORROW_ANNUAL, allows_shorting

# Covariance estimation window, uniform across markets: 36 months of daily
# returns ending the day before formation (see fscore.grid.COV_MONTHS).
COV_MONTHS = 36

STRATEGIES = ["fscore_EW", "fscore_GMV", "fscore_GMVsec", "fscore_LS",
              "value_EW", "mktcap_EW", "liquidity_EW"]
LONG_SHORT = "fscore_LS"   # dropped in markets where shorting is unavailable


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
                   n: int = 150, min_days: int = 200,
                   members: set[str] | None = None) -> pd.DataFrame:
    """Eligible universe at formation: (optional) index membership as of the
    formation date, continuous listing over the prior year, usable
    fundamentals, ranked by median daily dollar volume (top `n`).
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

    if members is not None:
        t = t[t.ticker.isin(members)]
    uni = t[t.ticker.isin(alive.index)].copy()
    uni["adv"] = uni.ticker.map(alive.adv)
    uni = uni.sort_values("adv", ascending=False).head(n)
    uni["bm"] = uni.book_value / uni.market_cap
    return uni.reset_index(drop=True)


def holding_returns(prices: pd.DataFrame, tickers: list[str],
                    start: pd.Timestamp, end: pd.Timestamp,
                    delisting_return: float = 0.0) -> pd.DataFrame:
    """Wide daily-return frame for `tickers` over (start, end].

    Delegates to `returns_panel`: delisted names are carried at their most
    recent trading price (they earn 0 from then on rather than dropping out
    of the book), and price gaps keep their return instead of losing it.
    """
    return returns_panel(prices, tickers, start, end,
                         delisting_return=delisting_return)


# ----------------------------------------------------------------------
# one formation year
# ----------------------------------------------------------------------

@dataclass
class YearResult:
    year: int
    universe: pd.DataFrame
    scored: pd.DataFrame
    baskets: dict
    weights: dict          # strategy -> target weights at formation
    weights_end: dict      # strategy -> weights after a year of drift
    daily: pd.DataFrame    # columns = STRATEGIES
    mc_daily: dict         # construction -> DataFrame (cols = draw indices)
    diagnostics: dict = field(default_factory=dict)


def run_year(fundamentals, prices, sectors, year, *, k=BASKET_SIZE,
             universe_size=150, value_quantile=0.4, n_mc=1000,
             n_mc_opt=300, lag_months=5, seed=42,
             membership: dict[int, set[str]] | None = None,
             end_cap: pd.Timestamp | None = None,   # see note in run_study
             detone: bool = False, allow_short: bool = False,
             delisting_return: float = 0.0,
             cov_months: int = COV_MONTHS) -> YearResult:
    fd = formation_date(year)
    hold_end = fd + pd.DateOffset(years=1) - pd.Timedelta(days=1)
    if end_cap is not None:
        # A cap truncates the last holding year, mixing a partial window in
        # with complete ones. The study instead ends at the last formation
        # whose full year finishes inside the sample, so this is left unset;
        # it stays available for one-off diagnostics.
        hold_end = min(hold_end, pd.Timestamp(end_cap))

    snap = pit_snapshot(fundamentals, year, lag_months=lag_months)
    uni = build_universe(prices, snap, year, n=universe_size,
                         members=membership.get(year) if membership else None)
    value_set = high_bm_subset(uni, quantile=value_quantile)

    scored = piotroski_signals(snap, year=year - 1)
    n_incomplete = scored.attrs.get("dropped_incomplete", 0)
    scored = scored[scored.ticker.isin(value_set.ticker)]
    scored = scored.merge(value_set[["ticker", "bm", "market_cap", "adv"]], on="ticker")

    k_eff = min(k, len(scored))
    # ties on the integer F-Score are broken at random, seeded per formation
    # year — a B/M tie-break would hand a third of the basket to the value
    # factor (see fscore.selection.baskets.rank_by_fscore)
    n_tie_slots = tie_break_slots(scored, k_eff)
    ranked = rank_by_fscore(scored, seed=seed + year)
    baskets = {
        "fscore": ranked.head(k_eff)["ticker"].tolist(),
        "fscore_short": ranked.tail(k_eff)["ticker"].tolist(),   # short leg
        "value": value_basket(value_set, k=k_eff),
        "mktcap": mktcap_basket(value_set, k=k_eff),
    }
    baskets["liquidity"] = liquidity_matched_basket(
        value_set, baskets["fscore"], liquidity_col="adv", k=k_eff, seed=seed)
    mc = random_baskets(value_set.ticker.tolist(), k=k_eff,
                        n_draws=n_mc, seed=seed + year)

    # covariance estimation window: `cov_months` before formation (PIT)
    est = holding_returns(prices, value_set.ticker.tolist(),
                          fd - pd.DateOffset(months=cov_months),
                          fd - pd.Timedelta(days=1))
    hold = holding_returns(prices, value_set.ticker.tolist(), fd, hold_end,
                           delisting_return=delisting_return)

    def weights_for(basket: list[str], how: str) -> pd.Series:
        cols = [c for c in basket if c in est.columns]
        if how == "EW" or len(cols) < 5:
            return equal_weight(cols if cols else basket)
        cov = clean_rmt(est[cols], detone=detone)
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
    if allow_short:
        # Piotroski spread portfolio: long top-k scores, short bottom-k,
        # equally weighted and dollar-neutral (+100% / -100%, gross 2)
        longs = baskets["fscore"]
        shorts = [t for t in baskets["fscore_short"] if t not in set(longs)]
        weights[LONG_SHORT] = pd.concat([
            pd.Series(1.0 / len(longs), index=longs),
            pd.Series(-1.0 / len(shorts), index=shorts),
        ]) if shorts else pd.Series(dtype=float)

    weights = {n: w for n, w in weights.items() if len(w)}

    # a held name with no post-formation quote is carried at its last traded
    # price (zero return) rather than dropped, which would re-weight the book
    # onto the survivors mid-year
    hold = hold.reindex(columns=sorted(set(hold.columns) | set(est.columns)))

    def _leg_path(w: pd.Series):
        """Buy-and-hold value path of one long book, and its drifted weights.

        Rebalancing is ANNUAL: the book is bought at formation and left to
        drift for twelve months. Applying a fixed weight vector to daily
        returns would rebalance back to target every day instead — a
        different strategy, and not the one the turnover figure prices.
        """
        cols = [c for c in w.index if c in hold.columns]
        ww = w.reindex(cols)
        ww = ww / ww.sum()
        growth = (1.0 + hold[cols].fillna(0.0)).cumprod()
        value = (growth * ww).sum(axis=1)
        end = ww * growth.iloc[-1]
        return value, end / end.sum()

    def port_ret(w: pd.Series):
        """Daily returns of the held book, and the weights it drifts to by
        the end of the holding year (what the next rebalance trades from)."""
        if abs(w.sum()) > 1e-9:
            value, end = _leg_path(w)
            r = value.pct_change()
            r.iloc[0] = value.iloc[0] - 1.0
            return r, end
        lv, le = _leg_path(w[w > 0])
        sv, se = _leg_path(-w[w < 0])
        rl, rs = lv.pct_change(), sv.pct_change()
        rl.iloc[0], rs.iloc[0] = lv.iloc[0] - 1.0, sv.iloc[0] - 1.0
        end = pd.Series({**le.to_dict(), **(-se).to_dict()})
        return rl - rs, end

    daily_cols, weights_end = {}, {}
    for name, w in weights.items():
        daily_cols[name], weights_end[name] = port_ret(w)
    daily = pd.DataFrame(daily_cols)

    # Monte-Carlo control through the identical construction pipeline
    mc_daily: dict[str, pd.DataFrame] = {}
    ew_cols = {}
    for i, b in enumerate(mc):
        ew_cols[i] = port_ret(equal_weight(b))[0]
    mc_daily["EW"] = pd.DataFrame(ew_cols)
    for how in ("GMV", "GMVsec"):
        cols_out = {}
        for i, b in enumerate(mc[:n_mc_opt]):
            cols_out[i] = port_ret(weights_for(b, how))[0]
        mc_daily[how] = pd.DataFrame(cols_out)

    diag = {"universe": len(uni), "value_set": len(value_set),
            "scored": len(scored),
            # firms excluded because they lacked a complete nine-signal score
            "dropped_incomplete_signals": n_incomplete,
            "tie_break_slots": n_tie_slots,
            "cov_est_days": int(est.notna().sum().median()) if len(est.columns) else 0,
            # names that stopped trading during the holding year; they are
            # carried at their last traded price, not dropped
            "delisted_in_holding_year": int(hold.attrs.get("delisted", 0)),
            "k": k_eff,
            "fscore_mean": float(scored.fscore.mean()),
            "fscore_basket_min": int(scored[scored.ticker.isin(baskets["fscore"])].fscore.min())
            if k_eff else np.nan}
    return YearResult(year, uni, scored, baskets, weights, weights_end,
                      daily, mc_daily, diag)


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

    def strategy_turnover(self, strategy: str) -> float:
        """Mean one-way turnover at each annual rebalance.

        Measured from the weights the old book has **drifted to** by the end
        of its holding year, not from last year's target: the drifted book is
        what has to be traded away from, which matches the annual rebalance
        the return series assumes.
        """
        tos = [turnover(a.weights_end[strategy], b.weights[strategy])
               for a, b in zip(self.yearly, self.yearly[1:])
               if strategy in a.weights_end and strategy in b.weights]
        return float(np.mean(tos)) if tos else 0.0

    def cost_drag(self, strategy: str, cost_per_side: float = 0.0020) -> float:
        """Trading costs, plus a stock-borrow fee on a long-short book."""
        drag = 2 * self.strategy_turnover(strategy) * cost_per_side
        if strategy == LONG_SHORT:
            shorts = [float(-w[w < 0].sum()) for y in self.yearly
                      for w in [y.weights.get(strategy, pd.Series(dtype=float))]
                      if len(w)]
            drag += SHORT_BORROW_ANNUAL * (float(np.mean(shorts)) if shorts else 0.0)
        return drag

    def effective_n(self, strategy: str) -> float:
        """Mean effective number of holdings, 1 / sum(w^2), across formations.

        Basket size k is an upper bound, not a headcount: an optimised or
        constrained book concentrates, so its effective N can sit well below
        the k names nominally held. Reported so the two are not conflated.
        """
        vals = []
        for y in self.yearly:
            w = y.weights.get(strategy)
            if w is None or not len(w):
                continue
            ww = w.abs()
            ww = ww / ww.sum()
            vals.append(1.0 / float((ww ** 2).sum()))
        return float(np.mean(vals)) if vals else np.nan

    def summary(self, rf_annual: float = 0.0,
                cost_per_side: float = 0.0020) -> pd.DataFrame:
        """Gross performance first — the cross-country convention — then
        concentration, turnover, and the net-of-cost sensitivity."""
        rows = {}
        for s in self.daily.columns:
            m = metrics(self.daily[s].dropna(), rf_annual)
            drag = self.cost_drag(s, cost_per_side)
            w0 = self.yearly[0].weights.get(s)
            m["nominal_k"] = float(len(w0)) if w0 is not None else np.nan
            m["effective_n"] = self.effective_n(s)
            m["turnover"] = self.strategy_turnover(s)
            m["cost_drag"] = drag
            m["net_ann_return"] = m["ann_return"] - drag
            m["net_sharpe"] = ((m["net_ann_return"] - rf_annual) / m["ann_vol"]
                               if m["ann_vol"] else np.nan)
            rows[s] = m
        cols = ["ann_return", "ann_vol", "sharpe", "max_drawdown",
                "nominal_k", "effective_n", "turnover",
                "cost_drag", "net_ann_return", "net_sharpe"]
        out = pd.DataFrame(rows).T
        return out[[c for c in cols if c in out.columns]]

    def mc_summary(self, construction: str = "EW") -> pd.DataFrame:
        mc = self.mc_daily[construction]
        return pd.DataFrame({i: metrics(mc[i].dropna()) for i in mc.columns}).T

    def placement(self, strategy: str = "fscore_EW",
                  construction: str = "EW") -> pd.DataFrame:
        """F-Score portfolio vs the random distribution, per metric.

        Gross of costs on both sides. The random baskets are redrawn every
        year, so their one-way turnover (~1 - k/|universe|) is close to the
        F-Score basket's own; charging both sides shifts the two sides by
        similar amounts and barely moves the percentile. Net-of-cost figures
        per strategy live in `summary`, turnover in `turnover_table`.
        """
        stat = metrics(self.daily[strategy].dropna())
        dist = self.mc_summary(construction)
        rows = {}
        # max_drawdown is stored as a negative number, so "higher" (less
        # negative, i.e. shallower) is better — same direction as the others
        for m, hib in [("ann_return", True), ("sharpe", True),
                       ("max_drawdown", True)]:
            rows[m] = vs_random(stat[m], dist[m].tolist(), higher_is_better=hib)
            rows[m]["fscore"] = stat[m]
        return pd.DataFrame(rows).T[["fscore", "random_mean", "random_std",
                                     "percentile", "p_value", "significant",
                                     "n_draws"]]

    def synergy(self, construction: str = "GMV") -> dict:
        """Optimisation gain D = Sharpe(optimised) - Sharpe(EW), placed inside
        the distribution of the same difference over random baskets.

        The test is **paired**: `mc_daily[construction]` optimises the first
        `n_mc_opt` of the very baskets `mc_daily["EW"]` holds equally, column
        for column, so D is measured on one basket at a time and the random
        distribution is of that same within-basket difference. Comparing two
        unpaired distributions would mix the optimiser's effect together with
        the luck of drawing different names.

        Reported per market so "does the optimiser add anything?" is answered
        separately from "does the screen pick anything?".
        """
        ew, opt = self.mc_daily["EW"], self.mc_daily[construction]
        n = opt.shape[1]
        d_rand = pd.Series(
            [metrics(opt[i].dropna())["sharpe"] - metrics(ew[i].dropna())["sharpe"]
             for i in range(n) if i in ew.columns]).dropna()
        d_f = (metrics(self.daily[f"fscore_{construction}"].dropna())["sharpe"]
               - metrics(self.daily["fscore_EW"].dropna())["sharpe"])
        out = vs_random(d_f, d_rand.tolist())
        out.update({"construction": construction, "D_fscore": d_f,
                    "D_random_mean": float(d_rand.mean()),
                    "D_random_std": float(d_rand.std()), "n": int(len(d_rand))})
        return out

    def synergy_table(self) -> pd.DataFrame:
        """D for every optimised construction this study ran."""
        rows = [self.synergy(c) for c in ("GMV", "GMVsec")
                if c in self.mc_daily and f"fscore_{c}" in self.daily.columns]
        return pd.DataFrame(rows).set_index("construction") if rows else pd.DataFrame()

    def turnover_table(self) -> pd.DataFrame:
        rows = []
        for prev, curr in zip(self.yearly, self.yearly[1:]):
            rows.append({"year": curr.year,
                         **{s: turnover(prev.weights_end[s], curr.weights[s])
                            for s in STRATEGIES
                            if s in prev.weights_end and s in curr.weights}})
        return pd.DataFrame(rows).set_index("year")


def run_study(market: str, fundamentals, prices, sectors, years,
              allow_short: bool | None = None, **kw) -> StudyResult:
    """Run the full study for one market. The long-short strategy runs only
    where shorting is available (see `fscore.markets`); Vietnam is long-only,
    so `fscore_LS` is absent from its results."""
    if allow_short is None:
        allow_short = allows_shorting(market)
    yearly = [run_year(fundamentals, prices, sectors, y,
                       allow_short=allow_short, **kw) for y in years]
    daily = pd.concat([yr.daily for yr in yearly]).sort_index()
    mc_daily = {how: pd.concat([yr.mc_daily[how] for yr in yearly]).sort_index()
                for how in yearly[0].mc_daily}
    return StudyResult(market, list(years), yearly, daily, mc_daily)
