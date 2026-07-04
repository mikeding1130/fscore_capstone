"""Construction layer: portfolio weights on a given basket.

Selection and construction are decoupled by design: these functions accept
any basket, so the incremental value of optimization can be measured on
identical stock lists.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------- RMT cleaning ----------------------------

def marchenko_pastur_bound(n_obs: int, n_assets: int, sigma2: float = 1.0) -> float:
    """Upper eigenvalue bound for a pure-noise correlation matrix (MP law)."""
    q = n_assets / n_obs
    return sigma2 * (1 + np.sqrt(q)) ** 2


def clean_rmt(returns: pd.DataFrame, detone: bool = True) -> np.ndarray:
    """Denoise (flatten sub-MP eigenvalues) and optionally detone (remove the
    dominant market mode) a covariance matrix estimated from `returns`
    (rows = dates, cols = tickers). Returns the cleaned covariance."""
    X = returns.dropna(how="any")
    n_obs, n_assets = X.shape
    corr = np.corrcoef(X.T)
    vols = X.std().values

    vals, vecs = np.linalg.eigh(corr)          # ascending
    lam_max = marchenko_pastur_bound(n_obs, n_assets)

    noise = vals < lam_max
    if noise.any():                             # denoise: average the noise band
        vals = vals.copy()
        vals[noise] = vals[noise].mean()
    corr_c = (vecs * vals) @ vecs.T

    if detone:                                  # remove market eigenmode
        i = np.argmax(vals)
        corr_c -= vals[i] * np.outer(vecs[:, i], vecs[:, i])
        d = np.sqrt(np.clip(np.diag(corr_c), 1e-10, None))
        corr_c = corr_c / np.outer(d, d)        # re-normalize diagonal to 1

    cov = corr_c * np.outer(vols, vols)
    return cov


# ------------------------------ weights -------------------------------

def equal_weight(tickers: list[str]) -> pd.Series:
    return pd.Series(1.0 / len(tickers), index=tickers)


def gmv_weights(cov: np.ndarray, tickers: list[str],
                long_only: bool = True) -> pd.Series:
    """Global Minimum Variance: min w'Σw s.t. Σw = 1 (long-only via clipping
    fallback; swap in cvxpy for exact constrained solve).

    TODO(optimization): replace fallback with cvxpy problem incl. box bounds.
    """
    n = len(tickers)
    try:
        inv = np.linalg.pinv(cov)
        ones = np.ones(n)
        w = inv @ ones / (ones @ inv @ ones)
    except np.linalg.LinAlgError:
        w = np.ones(n) / n
    if long_only and (w < 0).any():
        w = np.clip(w, 0, None)
        w = w / w.sum()
    return pd.Series(w, index=tickers)


def sector_constrained_gmv(cov: np.ndarray, tickers: list[str],
                           sectors: pd.Series, cap: float = 0.20) -> pd.Series:
    """Sector-capped GMV via iterative projection on the naive solution.

    TODO(optimization): exact cvxpy formulation with per-sector sum caps.
    """
    w = gmv_weights(cov, tickers)
    for _ in range(50):
        sec_tot = w.groupby(sectors.reindex(w.index)).sum()
        over = sec_tot[sec_tot > cap]
        if over.empty:
            break
        for sec in over.index:                  # scale down offending sectors
            mask = sectors.reindex(w.index) == sec
            w[mask] *= cap / sec_tot[sec]
        w = w / w.sum()
    return w
