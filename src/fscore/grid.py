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
  * One primary measure, fixed in advance (priority): Sharpe ratio net of
    costs (20 bp per side on one-way turnover), rf = 0. Other metrics are
    reported as secondary.
  * Controls include the plain long-only minimum-variance portfolio of the
    WHOLE universe and the universe equal-weight.
  * GMV covariance is denoised (Marchenko-Pastur) but NOT detoned: minimum
    variance must see the common market mode to manage it. (The detoned
    variant lives in the main study notebooks for comparison.)
  * Yearly returns and per-year diagnostics (universe size, B/M coverage,
    F>=8 count, names dropped for missing prices) are always reported.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .construction.weights import clean_rmt, equal_weight, gmv_weights, sector_constrained_gmv
from .evaluation.backtest import metrics, turnover, vs_random

COST_PER_SIDE = 0.0020
STRATEGIES = ["fscore_EW", "fscore_GMV", "fscore_GMVsec", "fscore_high_EW",
              "value_EW", "universe_EW", "universe_GMV"]


def formation_date(year: int) -> pd.Timestamp:
    return pd.Timestamp(f"{year}-07-01")


def _returns_window(prices, tickers, start, end):
    px = (prices[prices.ticker.isin(tickers)
                 & (prices.date >= start - pd.Timedelta(days=7))
                 & (prices.date <= end)]
          .pivot(index="date", columns="ticker", values="adj_close")
          .sort_index())
    rets = px.pct_change()
    return rets.loc[rets.index > start].dropna(how="all")


@dataclass
class GridYear:
    year: int
    universe: list
    baskets: dict
    daily: pd.DataFrame            # strategy columns
    mc_ew: pd.DataFrame            # n_mc random baskets, EW (fresh draws)
    mc_nonf_ew: pd.DataFrame       # random from universe minus fscore picks
    mc_gmv: pd.DataFrame           # first n_gmv random baskets under GMV
    diagnostics: dict = field(default_factory=dict)


def run_grid_year(scores, prices, sectors, year, *, k, n_mc, n_gmv, seed):
    fd = formation_date(year)
    hold_end = min(fd + pd.DateOffset(years=1) - pd.Timedelta(days=1),
                   pd.Timestamp("2025-12-31"))
    s = scores[scores.score_year == year - 1].copy()

    est = _returns_window(prices, s.ticker.tolist(),
                          fd - pd.DateOffset(years=1), fd - pd.Timedelta(days=1))
    hold = _returns_window(prices, s.ticker.tolist(), fd, hold_end)
    usable = [t for t in s.ticker
              if t in hold.columns and t in est.columns
              and est[t].notna().sum() >= 126]
    dropped_no_price = len(s) - len(usable)
    s = s[s.ticker.isin(usable)]
    est, hold = est[usable], hold[usable]

    s = s.sort_values(["fscore", "bm"], ascending=False, na_position="last")
    baskets = {
        "fscore": s.head(k).ticker.tolist(),
        "fscore_high": s[s.fscore >= 8].ticker.tolist(),
        "value": s.dropna(subset=["bm"]).sort_values("bm", ascending=False)
                  .head(k).ticker.tolist(),
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

    def port(w: pd.Series) -> pd.Series:
        ww = w / w.sum()
        return (hold[list(w.index)].fillna(0.0) * ww).sum(axis=1)

    daily = pd.DataFrame({
        "fscore_EW": port(equal_weight(baskets["fscore"])),
        "fscore_GMV": port(w_gmv(baskets["fscore"])),
        "fscore_GMVsec": port(sector_constrained_gmv(
            cov_for(tuple(baskets["fscore"])), baskets["fscore"], sectors)),
        "fscore_high_EW": port(equal_weight(baskets["fscore_high"]))
        if baskets["fscore_high"] else pd.Series(0.0, index=hold.index),
        "value_EW": port(equal_weight(baskets["value"])),
        "universe_EW": port(equal_weight(baskets["universe"])),
        "universe_GMV": port(w_gmv(baskets["universe"])),
    })

    # ---- Monte-Carlo controls (fresh draws every year), vectorised EW ----
    rng = np.random.default_rng(seed + year)
    uni = np.array(usable)
    H = hold.fillna(0.0).to_numpy()
    col_ix = {t: i for i, t in enumerate(usable)}

    def mc_ew_frame(pool: np.ndarray, draws: int) -> tuple[pd.DataFrame, np.ndarray]:
        idx = np.stack([rng.choice(len(pool), size=k, replace=False)
                        for _ in range(draws)])
        W = np.zeros((draws, len(usable)))
        for i in range(draws):
            W[i, [col_ix[pool[j]] for j in idx[i]]] = 1.0 / k
        return pd.DataFrame(H @ W.T, index=hold.index), idx

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
        gmv_cols[i] = port(w_gmv(b))
    mc_gmv = pd.DataFrame(gmv_cols)

    overlap = (np.isin(idx, [col_ix[t] for t in baskets["fscore"]])
               .sum(axis=1).mean() / k)
    diag = {
        "universe": len(usable), "dropped_no_price": dropped_no_price,
        "bm_coverage": int(s.bm.notna().sum()), "k": k,
        "n_fscore_high": len(baskets["fscore_high"]),
        "fscore_mean": float(s.fscore.mean()),
        "fscore_basket_min": float(s.head(k).fscore.min()),
        "overlap_random_vs_fscore": float(overlap),
        "overlap_expected": k / len(usable),
    }
    return GridYear(year, usable, baskets, daily, mc_ew, mc_nonf, mc_gmv, diag)


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

    # ------------------------------------------------------------------
    def _net(self, name, m):
        tos = [turnover(equal_weight(a.baskets["fscore"]),
                        equal_weight(b.baskets["fscore"]))
               for a, b in zip(self.yearly, self.yearly[1:])] or [0.0]
        drag = 2 * float(np.mean(tos)) * COST_PER_SIDE
        m["net_ann_return"] = m["ann_return"] - drag
        m["net_sharpe"] = (m["net_ann_return"]) / m["ann_vol"] if m["ann_vol"] else np.nan
        return m

    def summary(self) -> pd.DataFrame:
        rows = {}
        for sName in self.daily.columns:
            m = metrics(self.daily[sName].dropna())
            rows[sName] = self._net(sName, m)
        return pd.DataFrame(rows).T

    def mc_metric(self, frame, stat="sharpe") -> pd.Series:
        return pd.Series({i: metrics(frame[i].dropna()).get(stat, np.nan)
                          for i in frame.columns})

    def placement(self, strategy="fscore_EW", pool="mc_ew",
                  stat="sharpe") -> dict:
        dist = self.mc_metric(getattr(self, pool), stat)
        val = metrics(self.daily[strategy].dropna())[stat]
        out = vs_random(val, dist.tolist())
        out["fscore"] = val
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
             n_gmv=300, seed=42) -> GridStudy:
    yearly = [run_grid_year(scores, prices, sectors, y, k=k, n_mc=n_mc,
                            n_gmv=n_gmv, seed=seed) for y in years]
    cat = lambda attr: pd.concat([getattr(y, attr) for y in yearly]).sort_index()
    return GridStudy(market, k, n_mc, list(years), yearly,
                     cat("daily"), cat("mc_ew"), cat("mc_nonf_ew"), cat("mc_gmv"))
