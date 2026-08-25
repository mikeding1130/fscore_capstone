"""Grid study: peer-review-driven design on the team-computed F-Scores.

One study = (market, basket size k, number of random draws n_mc); the
reviewer-suggested grid is k in {20, 25, 30} x n_mc in {1000, 2000, 5000}
per market, one notebook per cell. Formations are July 1 of year T using
score_year T-1 — a single conservative timing rule for both markets.

Peer-review responses baked into the design:

  * Random basis is EXPLICIT (priority): random baskets are drawn from the
    full eligible universe (every scoreable name with usable prices), drawn
    FRESH EACH YEAR; the expected and realised overlap with the F-Score
    basket is reported per year. A second random control excludes the
    F-Score picks entirely.
  * The synergy claim is TESTED directly (priority): for each basket the
    optimisation gain D = Sharpe(GMV) - Sharpe(EW) is computed; the F-Score
    basket's D is placed within the distribution of D across random baskets.
  * Strict Piotroski cutoff (priority): an F>=8 portfolio is reported
    alongside the top-k rule (its size varies and is reported).
  * A long-short spread portfolio (`fscore_LS`): long the top-k scores,
    short the bottom-k, equally weighted and dollar-neutral, charged the
    trading costs of both legs plus a stock-borrow fee. It is only run where
    shorting is available — `fscore.markets` marks Vietnam long-only, so the
    strategy is absent from its output rather than reported as an untradable
    hypothetical.
  * One significance level, fixed in advance: ALPHA = 5%. Every test — the
    Monte-Carlo placements and the synergy test alike — is judged at p < 0.05
    and nothing else; p = 0.06 is reported as not significant.
  * One primary measure, fixed in advance: the GROSS Sharpe ratio (rf = 0).
    Gross is the cross-country convention, since cost models differ by market
    and would otherwise confound the comparison; turnover is reported beside
    it, and net-of-cost figures follow as a sensitivity rather than replacing
    the headline. Costs are charged per strategy on its OWN weights
    — an optimised portfolio pays for weight drift (GMV ~0.89 one-way vs EW
    ~0.67), a near-static control almost nothing (universe EW ~0.03) — and
    the random control pays its own turnover, since it is redrawn from
    scratch every year. That MC turnover (~1 - k/|universe|, ~0.67 here)
    lands close to the F-Score basket's, so `placement(..., net=True)` and
    the gross placement come out nearly identical; both are reported so the
    reader can see that costs are not driving the comparison.
  * Controls include the plain long-only minimum-variance portfolio of the
    WHOLE universe and the universe equal-weight.
  * GMV covariance is denoised (Marchenko-Pastur) but NOT detoned: minimum
    variance must see the common market mode to manage it. Detoning is out of
    scope for this study - removing the market eigenmode leaves the matrix
    singular, so inverting it optimises residual risk only. `clean_rmt` still
    takes the flag, and `tests/test_pipeline.py` pins why the default is off,
    but no notebook runs the variant.
  * Yearly returns and per-year diagnostics (universe size, B/M coverage,
    F>=8 count, names dropped for missing prices) are always reported.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .construction.weights import clean_rmt, equal_weight, gmv_weights, sector_constrained_gmv
from .selection.baskets import rank_by_fscore, tie_break_slots
from .evaluation.backtest import (ALPHA, metrics, returns_panel,
                                  turnover, vs_random)
from .markets import SHORT_BORROW_ANNUAL, allows_shorting

COST_PER_SIDE = 0.0020
# Covariance estimation window, uniform across markets and strategies: 36
# months of daily returns ending the day before formation. Longer than the
# holding year on purpose — with ~756 observations against 20-30 assets the
# sample covariance is far better conditioned (q = N/T ~ 0.03 rather than
# ~0.10), so fewer eigenvalues fall in the Marchenko-Pastur noise band and
# the minimum-variance solve rests on more signal. The cost is slower
# reaction to regime shifts, accepted for comparability.
COV_MONTHS = 36
# Minimum usable history for a name to be estimable. None = half the
# sessions actually present in the estimation window, so the rule scales with
# `cov_months` instead of silently emptying the universe when the window is
# shorter than a fixed threshold (and so the early formation years, which
# have less than 36 months of prices available, still run).
MIN_EST_DAYS = None
MC_TURNOVER_SAMPLE = 500   # draws whose names are kept to estimate MC turnover
STRATEGIES = ["fscore_EW", "fscore_GMV", "fscore_GMVsec", "fscore_high_EW",
              "fscore_LS", "value_EW", "universe_EW", "universe_GMV"]
LONG_SHORT = "fscore_LS"   # dropped in markets where shorting is unavailable


def formation_date(year: int) -> pd.Timestamp:
    return pd.Timestamp(f"{year}-07-01")


def _returns_window(prices, tickers, start, end, delisting_return=0.0):
    """Delisting-safe daily returns — see `fscore.evaluation.returns_panel`."""
    return returns_panel(prices, tickers, start, end,
                         delisting_return=delisting_return)


@dataclass
class GridYear:
    year: int
    universe: list
    baskets: dict
    weights: dict                  # strategy -> target weights at formation
    weights_end: dict              # strategy -> weights after a year of drift
    daily: pd.DataFrame            # strategy columns
    mc_ew: pd.DataFrame            # n_mc random baskets, EW (fresh draws)
    mc_nonf_ew: pd.DataFrame       # random from universe minus fscore picks
    mc_gmv: pd.DataFrame           # first n_gmv random baskets under GMV
    mc_names: list                 # sampled draws' name sets (turnover only)
    diagnostics: dict = field(default_factory=dict)


def run_grid_year(scores, prices, sectors, year, *, k, n_mc, n_gmv, seed,
                  allow_short: bool = False, delisting_return: float = 0.0,
                  cov_months: int = COV_MONTHS, min_est_days: int | None = MIN_EST_DAYS):
    fd = formation_date(year)
    # Every formation is held for a FULL year. Truncating the last one at the
    # sample end would mix a half-year window in with complete ones, so the
    # final formation is chosen to be one whose year finishes inside the
    # sample instead (July 2024 -> June 2025).
    hold_end = fd + pd.DateOffset(years=1) - pd.Timedelta(days=1)
    s = scores[scores.score_year == year - 1].copy()

    # covariance estimation window: `cov_months` of daily returns ending the
    # day before formation (point-in-time)
    est = _returns_window(prices, s.ticker.tolist(),
                          fd - pd.DateOffset(months=cov_months),
                          fd - pd.Timedelta(days=1))
    hold = _returns_window(prices, s.ticker.tolist(), fd, hold_end,
                           delisting_return=delisting_return)
    # Eligibility uses ONLY pre-formation data. Requiring a name to have
    # holding-period prices would be look-ahead: at formation we cannot know
    # whether it keeps trading, and it would drop exactly the names that
    # delist right after formation — the losses most worth capturing.
    need = (min_est_days if min_est_days is not None
            else max(126, len(est) // 2))
    usable = [t for t in s.ticker
              if t in est.columns and est[t].notna().sum() >= need]
    if not usable:
        raise ValueError(
            f"{year}: no name has >= {need} estimation sessions in the "
            f"{cov_months}-month window (window holds {len(est)} sessions). "
            "Extend the price cache or lower cov_months/min_est_days.")
    dropped_no_price = len(s) - len(usable)
    s = s[s.ticker.isin(usable)]
    est = est[usable]
    n_delisted = int(hold.attrs.get("delisted", 0))
    # names with no post-formation quote at all are carried at their last
    # traded price (zero return), not silently excluded from the book
    no_hold_prices = [t for t in usable if t not in hold.columns]
    hold = hold.reindex(columns=usable).fillna(0.0)

    # Ties on the integer F-Score are broken AT RANDOM (seeded per formation
    # year), not by B/M: ~a third of the basket sits on the cut-off score, so
    # a B/M tie-break would let the value factor choose a third of the
    # "F-Score" basket — the very thing the value control tests against.
    n_tie_slots = tie_break_slots(s, k)
    s = rank_by_fscore(s, seed=seed + year)
    value_pool = s.dropna(subset=["bm"]).sort_values("bm", ascending=False)
    # B/M coverage starts only when the fundamentals cache does (US FY2009+,
    # Japan FY2021+): earlier years fall back to the universe and are flagged
    value_fallback = len(value_pool) < max(5, k // 4)
    baskets = {
        "fscore": s.head(k).ticker.tolist(),
        "fscore_short": s.tail(k).ticker.tolist(),   # lowest scores (short leg)
        "fscore_high": s[s.fscore >= 8].ticker.tolist(),
        "value": (list(usable) if value_fallback
                  else value_pool.head(k).ticker.tolist()),
        "universe": list(usable),
    }

    cov_cache: dict[tuple, np.ndarray] = {}

    def cov_for(cols: tuple):
        if cols not in cov_cache:
            cov_cache[cols] = clean_rmt(est[list(cols)], detone=False)
        return cov_cache[cols]

    def w_gmv(basket):
        cols = tuple(basket)
        return gmv_weights(cov_for(cols), list(cols))

    def _leg_path(w: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Buy-and-hold value path of one long book, and its drifted weights.

        Rebalancing is ANNUAL: the book is bought at formation and left to
        drift for twelve months. Multiplying daily returns by a fixed weight
        vector would instead rebalance back to target every single day, which
        is neither the stated design nor what the turnover figure prices.
        """
        cols = list(w.index)
        ww = w / w.sum()
        growth = (1.0 + hold[cols].fillna(0.0)).cumprod()
        value = (growth * ww).sum(axis=1)
        end = ww * growth.iloc[-1]
        return value, end / end.sum()

    def port(w: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Daily returns of the held book, plus the weights it drifts to by
        the end of the holding year (what the next rebalance trades away
        from)."""
        if abs(w.sum()) > 1e-9:                       # long-only
            value, end = _leg_path(w)
            r = value.pct_change()
            r.iloc[0] = value.iloc[0] - 1.0
            return r, end
        # dollar-neutral: each leg drifts on its own, and the spread return is
        # the difference of the two buy-and-hold legs
        lv, le = _leg_path(w[w > 0])
        sv, se = _leg_path(-w[w < 0])
        rl, rs = lv.pct_change(), sv.pct_change()
        rl.iloc[0], rs.iloc[0] = lv.iloc[0] - 1.0, sv.iloc[0] - 1.0
        end = pd.Series({**le.to_dict(), **(-se).to_dict()})
        return rl - rs, end

    # weights are kept per strategy so trading costs are charged on each
    # strategy's OWN rebalancing, using its actual weights (a GMV portfolio
    # trades on weight drift as well as on name changes)
    weights = {
        "fscore_EW": equal_weight(baskets["fscore"]),
        "fscore_GMV": w_gmv(baskets["fscore"]),
        "fscore_GMVsec": sector_constrained_gmv(
            cov_for(tuple(baskets["fscore"])), baskets["fscore"], sectors),
        "fscore_high_EW": (equal_weight(baskets["fscore_high"])
                           if baskets["fscore_high"] else pd.Series(dtype=float)),
        "value_EW": equal_weight(baskets["value"]),
        "universe_EW": equal_weight(baskets["universe"]),
        "universe_GMV": w_gmv(baskets["universe"]),
    }
    if allow_short:
        # Piotroski's spread portfolio: long the top-k scores, short the
        # bottom-k, equally weighted, dollar-neutral (+100% / -100%, gross 2).
        # Names in both legs would cancel; the two legs are disjoint by
        # construction unless the universe is smaller than 2k.
        longs, shorts = baskets["fscore"], [t for t in baskets["fscore_short"]
                                            if t not in set(baskets["fscore"])]
        weights[LONG_SHORT] = pd.concat([
            pd.Series(1.0 / len(longs), index=longs),
            pd.Series(-1.0 / len(shorts), index=shorts),
        ]) if shorts else pd.Series(dtype=float)
    daily_cols, weights_end = {}, {}
    for name, w in weights.items():
        if len(w):
            daily_cols[name], weights_end[name] = port(w)
        else:
            daily_cols[name] = pd.Series(0.0, index=hold.index)
            weights_end[name] = pd.Series(dtype=float)
    daily = pd.DataFrame(daily_cols)

    # ---- Monte-Carlo controls (fresh draws every year), vectorised EW ----
    rng = np.random.default_rng(seed + year)
    uni = np.array(usable)
    H = hold.fillna(0.0).to_numpy()
    col_ix = {t: i for i, t in enumerate(usable)}

    G = np.cumprod(1.0 + H, axis=0)      # per-name growth, for buy-and-hold

    def mc_ew_frame(pool: np.ndarray, draws: int) -> tuple[pd.DataFrame, np.ndarray]:
        idx = np.stack([rng.choice(len(pool), size=k, replace=False)
                        for _ in range(draws)])
        W = np.zeros((draws, len(usable)))
        for i in range(draws):
            W[i, [col_ix[pool[j]] for j in idx[i]]] = 1.0 / k
        # value paths of buy-and-hold books, then daily returns from them
        V = G @ W.T
        R = np.vstack([V[0] - 1.0, V[1:] / V[:-1] - 1.0])
        return pd.DataFrame(R, index=hold.index), idx

    mc_ew, idx = mc_ew_frame(uni, n_mc)
    pool_nonf = np.array([t for t in usable if t not in set(baskets["fscore"])])
    k_nonf = min(k, len(pool_nonf))
    mc_nonf, _ = (mc_ew_frame(pool_nonf, n_mc) if k_nonf == k
                  else (pd.DataFrame(index=hold.index), None))

    # GMV on the first n_gmv random baskets (same draws as the EW MC, so the
    # per-basket optimisation gain D is well defined)
    gmv_cols = {}
    for i in range(min(n_gmv, n_mc)):
        b = [uni[j] for j in idx[i]]
        gmv_cols[i] = port(w_gmv(b))[0]
    mc_gmv = pd.DataFrame(gmv_cols)

    # names behind the first draws, kept so the random control's own turnover
    # (it is redrawn from scratch every year) can be charged the same costs
    mc_names = [frozenset(uni[j] for j in row) for row in idx[:MC_TURNOVER_SAMPLE]]

    overlap = (np.isin(idx, [col_ix[t] for t in baskets["fscore"]])
               .sum(axis=1).mean() / k)
    diag = {
        "universe": len(usable), "dropped_no_price": dropped_no_price,
        "bm_coverage": int(s.bm.notna().sum()), "k": k,
        "value_fallback": value_fallback,
        "value_basket_size": len(baskets["value"]),
        "n_fscore_high": len(baskets["fscore_high"]),
        "fscore_mean": float(s.fscore.mean()),
        "fscore_basket_min": float(s.head(k).fscore.min()),
        # slots filled by the random tie-break rather than by score rank
        "tie_break_slots": n_tie_slots,
        "overlap_random_vs_fscore": float(overlap),
        "overlap_expected": k / len(usable),
        "cov_est_days": int(est[usable].notna().sum().median()),
        "cov_min_est_days": int(need),
        "delisted_in_holding_year": n_delisted,
        "no_price_after_formation": len(no_hold_prices),
        "long_short_run": bool(allow_short and len(weights.get(LONG_SHORT, []))),
        "short_leg_max_fscore": (float(s.tail(k).fscore.max())
                                 if allow_short else np.nan),
    }
    return GridYear(year, usable, baskets, weights, weights_end, daily,
                    mc_ew, mc_nonf, mc_gmv, mc_names, diag)


@dataclass
class GridStudy:
    market: str
    k: int
    n_mc: int
    years: list
    yearly: list
    daily: pd.DataFrame
    mc_ew: pd.DataFrame
    mc_nonf_ew: pd.DataFrame
    mc_gmv: pd.DataFrame

    # ---------------------------- trading costs ----------------------------
    def strategy_turnover(self, name: str) -> float:
        """Mean one-way turnover at each annual rebalance.

        Measured from the weights the old book has **drifted to** by the end
        of its holding year, not from last year's target: the drifted book is
        what actually has to be traded away from, so this matches the annual
        rebalance the return series assumes. Comparing two target vectors
        would price trades that were never made.
        """
        tos = [turnover(a.weights_end[name], b.weights[name])
               for a, b in zip(self.yearly, self.yearly[1:])
               if len(a.weights_end.get(name, [])) and len(b.weights[name])]
        return float(np.mean(tos)) if tos else 0.0

    def mc_turnover(self) -> float:
        """Mean one-way turnover of a random basket. The control is redrawn
        from scratch every year, so it trades far more than a persistent
        screen — charging costs to the strategy but not to the control would
        flatter the strategy."""
        tos = []
        for a, b in zip(self.yearly, self.yearly[1:]):
            for A, B in zip(a.mc_names, b.mc_names):
                tos.append(1.0 - len(A & B) / len(A))
        return float(np.mean(tos)) if tos else 0.0

    def cost_drag(self, name: str | None = None, mc: bool = False) -> float:
        """Annual cost drag = 2 x one-way turnover x cost per side, plus the
        stock-borrow fee on the short leg's notional for a long-short book."""
        if mc:
            return 2 * self.mc_turnover() * COST_PER_SIDE
        drag = 2 * self.strategy_turnover(name) * COST_PER_SIDE
        if name == LONG_SHORT:
            shorts = [float(-w[w < 0].sum()) for y in self.yearly
                      for w in [y.weights.get(name, pd.Series(dtype=float))]
                      if len(w)]
            drag += SHORT_BORROW_ANNUAL * (float(np.mean(shorts)) if shorts else 0.0)
        return drag

    @staticmethod
    def _apply_drag(m: dict, drag: float, stat: str) -> float:
        if stat == "ann_return":
            return m["ann_return"] - drag
        if stat == "sharpe":
            return ((m["ann_return"] - drag) / m["ann_vol"]
                    if m["ann_vol"] else np.nan)
        return m.get(stat, np.nan)      # vol / drawdown unaffected by the drag

    # ------------------------------ reporting ------------------------------
    def effective_n(self, name: str) -> float:
        """Mean effective number of holdings, 1 / sum(w^2), across formations.

        The basket size k is an upper bound, not a headcount: a constrained
        or optimised book concentrates, so its effective N can sit well below
        k even though k names are nominally held. Reported so the two are
        never read as the same quantity.
        """
        vals = []
        for y in self.yearly:
            w = y.weights.get(name)
            if w is None or not len(w):
                continue
            ww = w.abs()
            ww = ww / ww.sum()
            vals.append(1.0 / float((ww ** 2).sum()))
        return float(np.mean(vals)) if vals else np.nan

    def summary(self) -> pd.DataFrame:
        """Gross performance is the headline convention; turnover and the
        net-of-cost columns follow it rather than replacing it."""
        rows = {}
        for name in self.daily.columns:
            m = metrics(self.daily[name].dropna())
            drag = self.cost_drag(name)
            m["nominal_k"] = float(len(self.yearly[0].weights.get(name, [])))
            m["effective_n"] = self.effective_n(name)
            m["turnover"] = self.strategy_turnover(name)
            m["cost_drag"] = drag
            m["net_ann_return"] = self._apply_drag(m, drag, "ann_return")
            m["net_sharpe"] = self._apply_drag(m, drag, "sharpe")
            rows[name] = m
        cols = ["ann_return", "ann_vol", "sharpe", "max_drawdown",   # gross first
                "nominal_k", "effective_n", "turnover",
                "cost_drag", "net_ann_return", "net_sharpe"]
        out = pd.DataFrame(rows).T
        return out[[c for c in cols if c in out.columns]]

    def mc_metric(self, frame, stat="sharpe", net=False) -> pd.Series:
        drag = self.cost_drag(mc=True) if net else 0.0
        return pd.Series({i: self._apply_drag(metrics(frame[i].dropna()), drag, stat)
                          for i in frame.columns})

    def placement(self, strategy="fscore_EW", pool="mc_ew", stat="sharpe",
                  net=False) -> dict:
        """Place a strategy inside the random distribution. With net=True both
        sides are charged their own trading costs."""
        dist = self.mc_metric(getattr(self, pool), stat, net=net)
        m = metrics(self.daily[strategy].dropna())
        val = self._apply_drag(m, self.cost_drag(strategy) if net else 0.0, stat)
        higher_is_better = stat != "ann_vol"
        out = vs_random(val, dist.tolist(), higher_is_better=higher_is_better)
        out["fscore"] = val
        out["basis"] = "net" if net else "gross"
        return out

    def synergy(self) -> dict:
        """Reviewer priority 1: per-basket optimisation gain
        D = Sharpe(GMV) - Sharpe(EW), F-Score basket vs random baskets."""
        n = self.mc_gmv.shape[1]
        d_rand = (self.mc_metric(self.mc_gmv, "sharpe")
                  - self.mc_metric(self.mc_ew[list(range(n))], "sharpe")).dropna()
        d_f = (metrics(self.daily["fscore_GMV"].dropna())["sharpe"]
               - metrics(self.daily["fscore_EW"].dropna())["sharpe"])
        out = vs_random(d_f, d_rand.tolist())
        out.update({"D_fscore": d_f, "D_random_mean": float(d_rand.mean()),
                    "D_random_std": float(d_rand.std()), "n": int(len(d_rand))})
        return out

    def yearly_returns(self) -> pd.DataFrame:
        r = self.daily.fillna(0)
        return ((1 + r).groupby(r.index.year).prod() - 1)

    def diagnostics(self) -> pd.DataFrame:
        return pd.DataFrame([{"year": y.year, **y.diagnostics}
                             for y in self.yearly]).set_index("year")


def run_grid(market, scores, prices, sectors, years, *, k, n_mc,
             n_gmv=300, seed=42, allow_short: bool | None = None,
             delisting_return: float = 0.0, cov_months: int = COV_MONTHS,
             min_est_days: int | None = MIN_EST_DAYS) -> GridStudy:
    """Run the grid for one market. The long-short strategy is included only
    where shorting is actually available (see `fscore.markets`): Vietnam and
    Malaysia run long-only, so `fscore_LS` is absent from their output rather
    than reported as an untradable hypothetical."""
    if allow_short is None:
        allow_short = allows_shorting(market)
    yearly = [run_grid_year(scores, prices, sectors, y, k=k, n_mc=n_mc,
                            n_gmv=n_gmv, seed=seed, allow_short=allow_short,
                            delisting_return=delisting_return,
                            cov_months=cov_months, min_est_days=min_est_days)
              for y in years]
    cat = lambda attr: pd.concat([getattr(y, attr) for y in yearly]).sort_index()
    return GridStudy(market, k, n_mc, list(years), yearly,
                     cat("daily"), cat("mc_ew"), cat("mc_nonf_ew"), cat("mc_gmv"))
