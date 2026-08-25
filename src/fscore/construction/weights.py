"""Construction layer: portfolio weights on a given basket.

Selection and construction are decoupled by design: these functions accept
any basket, so the incremental value of optimization can be measured on
identical stock lists.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


# ---------------------------- RMT cleaning ----------------------------

def marchenko_pastur_bound(n_obs: int, n_assets: int, sigma2: float = 1.0) -> float:
    """Upper eigenvalue bound for a pure-noise correlation matrix (MP law)."""
    q = n_assets / n_obs
    return sigma2 * (1 + np.sqrt(q)) ** 2


def clean_rmt(returns: pd.DataFrame, detone: bool = False) -> np.ndarray:
    """Denoise (flatten sub-MP eigenvalues) and optionally detone (remove the
    dominant market mode) a covariance matrix estimated from `returns`
    (rows = dates, cols = tickers). Returns the cleaned covariance.

    Denoising is always safe: the noise-band eigenvalues are replaced by their
    mean, which preserves the trace and leaves the matrix well conditioned.

    **Detoning defaults to False and must not be combined with an inverse.**
    Removing the market eigenmode makes the correlation matrix singular (its
    smallest eigenvalue goes to ~0), so feeding it to a minimum-variance
    solver optimises residual risk only: on a 25-name basket the detoned
    solution predicted 0.03% annualised vol against ~14% realised, and the
    weights collapsed towards equal weight (effective N 23 of 25). Detoning
    belongs to correlation/clustering analysis, not to portfolio inversion
    (López de Prado 2020). Pass detone=True only for that kind of diagnostic,
    or to reproduce the pre-2026-08 results.
    """
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


def _solve_qp(cov: np.ndarray, n: int, extra_constraints: list) -> np.ndarray:
    """min w'Σw  s.t.  Σw = 1, w >= 0 (+ extras), via SLSQP."""
    from scipy.optimize import minimize

    cov = np.asarray(cov)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0,
             "jac": lambda w: np.ones(n)}] + extra_constraints
    res = minimize(lambda w: w @ cov @ w, np.ones(n) / n,
                   jac=lambda w: 2 * cov @ w,
                   bounds=[(0.0, 1.0)] * n, constraints=cons,
                   method="SLSQP", options={"maxiter": 500, "ftol": 1e-12})
    w = np.clip(res.x, 0, None)
    return w / w.sum()


# Memo for repeated *identical* minimum-variance solves. The grid sweeps
# (k x N) inside one process, and the whole-universe control does not depend
# on either: the same several-hundred-name QP would otherwise be re-solved
# once per cell, and on Vietnam's ~800-name universe that single solve is
# ~90 s — more than the rest of a formation year put together. The key is the
# exact input (tickers + covariance bytes), and SLSQP from a fixed start is
# deterministic, so a hit returns precisely what recomputing would; no result
# changes. Bounded and FIFO-evicted so a long sweep cannot grow it without
# limit.
_GMV_CACHE: dict = {}
_GMV_CACHE_MAX = 64
# Small baskets solve in milliseconds; hashing their covariance would cost
# more than the solve, so only the expensive ones are cached.
_CACHE_MIN_ASSETS = 100


def _gmv_key(cov: np.ndarray, tickers: list[str], long_only: bool):
    arr = np.ascontiguousarray(np.asarray(cov, dtype=float))
    return (tuple(tickers), long_only, arr.shape,
            hashlib.blake2b(arr.tobytes(), digest_size=16).digest())


def gmv_weights(cov: np.ndarray, tickers: list[str],
                long_only: bool = True) -> pd.Series:
    """Global Minimum Variance: min w'Σw s.t. Σw = 1.

    Unconstrained GMV has the closed form Σ⁻¹1 / (1'Σ⁻¹1); the long-only
    variant is solved exactly as a QP (SLSQP with analytic gradients).

    Identical (tickers, covariance) inputs are memoised — see `_GMV_CACHE`.
    """
    n = len(tickers)
    key = _gmv_key(cov, tickers, long_only) if n >= _CACHE_MIN_ASSETS else None
    if key is not None and key in _GMV_CACHE:
        return _GMV_CACHE[key].copy()
    inv = np.linalg.pinv(cov)
    ones = np.ones(n)
    w = inv @ ones / (ones @ inv @ ones)
    if long_only and (w < 0).any():
        w = _solve_qp(cov, n, [])
    out = pd.Series(w, index=tickers)
    if key is not None:
        if len(_GMV_CACHE) >= _GMV_CACHE_MAX:
            _GMV_CACHE.pop(next(iter(_GMV_CACHE)))
        _GMV_CACHE[key] = out
    return out.copy()


def sector_constrained_gmv(cov: np.ndarray, tickers: list[str],
                           sectors: pd.Series, cap: float = 0.20) -> pd.Series:
    """Long-only GMV with each sector's total weight capped at `cap`,
    solved exactly as a QP with one linear inequality per sector."""
    n = len(tickers)
    labels = sectors.reindex(tickers).fillna("Unknown")
    # feasibility: k sectors need k*cap >= 1, else relax the cap minimally
    cap = max(cap, 1.0 / labels.nunique() + 1e-6)
    cons = []
    for sec in labels.unique():
        mask = (labels == sec).to_numpy().astype(float)
        if mask.sum() * (1.0 / n) > 0:  # always constrain; cheap
            cons.append({"type": "ineq",
                         "fun": (lambda w, m=mask: cap - m @ w),
                         "jac": (lambda w, m=mask: -m)})
    return pd.Series(_solve_qp(np.asarray(cov), n, cons), index=tickers)
