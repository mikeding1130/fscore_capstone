"""
Selection and construction primitives for cross-sectional portfolio experiments.

A **portfolio** here is one pair: a *selection arm* (which K names) crossed with a
*weighting scheme* (how much of each). Everything downstream — metrics, Monte-Carlo
p-values, characteristics, capacity — is a function of that pair, so every analysis in
this module takes a `Portfolio` and returns a single DataFrame for it. Nothing joins
EW to GMV; that is the caller's `pd.concat`.

No I/O lives here. The caller reads `final_panel.csv` and the monthly price history and
passes DataFrames in, matching the convention of `preprocessing_pipelines/statement_fields.py`.

Usage
-----
    import pandas as pd
    from portfolio_util import *

    panel = pd.read_csv("../data/preprocessing_pipeline_results/final_panel.csv") \\
              .set_index(["symbol", "period"])
    keep = (panel["tradeable"].eq(True) & panel["f_score"].notna()
            & panel["bm"].notna() & panel["fwd_return_1y"].notna())

    uni = build_universe(panel.loc[keep])
    cov = build_cov(monthly_returns, uni)          # only needed for GMV

    P = {(arm, wt): build_portfolio(uni, arm, 30, weighting=wt, cov=cov,
                                    n_sim=2000, seed=SEEDS[arm])
         for arm in ("f_rand", "random") for wt in ("ew", "gmv")}

    metrics_table(P[("f_rand", "ew")], P[("random", "ew")])

Reproducibility
---------------
`select` consumes ``rng.random((n_paths, n))`` exactly once per year, in ascending year
order. That call order is part of the contract: change it and previously published
numbers stop reproducing. Deterministic arms ignore the generator entirely and collapse
to a single path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "Arm", "ARMS", "WEIGHTINGS", "DIRECTION",
    "Universe", "build_universe",
    "select", "build_cov", "gmv_longonly", "weights",
    "Portfolio", "build_portfolio",
    "metrics_table", "annual_table", "paired_test", "characteristics",
    "buckets", "capacity",
    "rankable", "plot_null", "plot_equity",
]


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Arm:
    """A ranking rule.

    The ranking key is always ``(score if use_score else 0) + tiebreak``, with the
    tiebreak confined to [0, 1). Because a score is an integer, the tiebreak can only
    reorder names *within* a score, never across two — that bound is what keeps this a
    tiebreak rather than a blended signal.

    `fixed` marks an arm whose key carries no randomness, so it collapses to one path.
    """

    use_score: bool
    tiebreak: str          # "random" | "alt"
    fixed: bool
    doc: str = ""


ARMS: dict[str, Arm] = {
    "random":  Arm(False, "random", False, "uniform draw — the no-skill null"),
    "f_rand":  Arm(True,  "random", False, "score ranking, ties broken at random"),
    "f_alt":   Arm(True,  "alt",    True,  "score ranking, ties broken by the alt key"),
    "alt_only": Arm(False, "alt",   True,  "alt key alone, score ignored"),
}

# Back-compatible aliases for the notebook's book-to-market naming.
ARMS["f_bm"] = ARMS["f_alt"]
ARMS["bm_only"] = ARMS["alt_only"]


def select(uni: "Universe", arm: str, k: int, *, n_sim: int, seed: int) -> dict:
    """Pick `k` names per year.

    Returns ``{year: (n_paths, k)}`` of positions into that year's universe arrays.
    A deterministic arm returns ``n_paths == 1``.
    """
    spec = ARMS[arm]
    n_paths = 1 if spec.fixed else n_sim
    rng = np.random.default_rng(seed)
    out = {}
    for y in uni.years:                       # ascending — part of the RNG contract
        n = len(uni.RET[y])
        kk = min(k, n)
        base = uni.SCORE[y] if spec.use_score else np.zeros(n)
        if spec.tiebreak == "random":
            key = base[None, :] + rng.random((n_paths, n))
        else:
            key = np.broadcast_to(base + uni.ALT[y], (n_paths, n))
        out[y] = np.argpartition(-key, kk - 1, axis=1)[:, :kk]
    return out


# --------------------------------------------------------------------------- #
# universe
# --------------------------------------------------------------------------- #

@dataclass
class Universe:
    """Per-year arrays the engine works on, plus the index that provenances them.

    Every ``dict`` is keyed by formation year and holds one row per eligible name, in
    the row order of `frame`. Positions returned by `select` index into these.
    """

    years: np.ndarray = field(repr=False)
    T: int
    symbols: pd.Index = field(repr=False)
    frame: pd.DataFrame = field(repr=False)   # the eligible rows, in array order
    RET: dict = field(repr=False, default_factory=dict)
    CODE: dict = field(repr=False, default_factory=dict)
    SCORE: dict = field(repr=False, default_factory=dict)
    ALT: dict = field(repr=False, default_factory=dict)

    def __repr__(self) -> str:
        return (f"Universe(T={self.T}, years={self.years[0]}-{self.years[-1]}, "
                f"rows={len(self.frame):,}, symbols={len(self.symbols):,})")

    def n(self, year) -> int:
        return len(self.RET[year])

    def sizes(self) -> pd.Series:
        """Names available per formation year — the pool both arms draw from."""
        return pd.Series({int(y): self.n(y) for y in self.years},
                         name="n_names").rename_axis("formation_year")

    def column(self, name: str) -> dict:
        """``{year: values}`` for any column of `frame`, aligned to the arrays."""
        col = self.frame[name].astype(float)
        fy = self.frame["__formation_year"]
        return {y: col[fy == y].to_numpy() for y in self.years}


def build_universe(panel: pd.DataFrame, *,
                   formation_col: str = "formation_year",
                   return_col: str = "fwd_return_1y",
                   score_col: str = "f_score",
                   alt_col: str = "bm") -> Universe:
    """Freeze an already-filtered panel into per-year arrays.

    `panel` must contain only rows that are actually investable — this function does no
    screening, so whatever is passed in is what both the strategy and the null draw from.

    The alt key is mapped to ``rank(pct) * 0.999`` so it lands in [0, 1) and cannot
    outrank a whole score point.

    Raises if any required column is missing or carries NaN: a name the engine cannot
    rank, price, or date is a name neither arm could have held, so silently tolerating it
    would put rows in the universe that no portfolio can draw.
    """
    required = {"formation year": formation_col, "return": return_col,
                "score": score_col, "alt key": alt_col}
    missing = {role: col for role, col in required.items() if col not in panel.columns}
    if missing:
        raise KeyError(
            "build_universe is missing columns: "
            + ", ".join(f"{col!r} (the {role})" for role, col in missing.items())
            + f". Available: {list(panel.columns)}"
        )
    holes = {col: int(panel[col].isna().sum()) for col in required.values()
             if panel[col].isna().any()}
    if holes:
        raise ValueError(
            "build_universe expects an already-filtered panel, but found NaNs in "
            + ", ".join(f"{col!r} ({n:,} rows)" for col, n in holes.items())
            + ". Filter these out first — build_universe deliberately screens nothing, "
              "so whatever is passed in is the pool BOTH the strategy and the null draw from."
        )

    df = panel.copy()
    df["__formation_year"] = df[formation_col].astype(int)
    df = df.sort_values(["__formation_year", score_col, alt_col],
                        ascending=[True, False, False])

    years = np.sort(df["__formation_year"].unique())
    symbols = pd.Index(sorted(df.index.get_level_values("symbol").unique()))

    uni = Universe(years=years, T=len(years), symbols=symbols, frame=df)
    for y in years:
        d = df[df["__formation_year"] == y]
        uni.RET[y] = d[return_col].to_numpy(float)
        uni.CODE[y] = symbols.get_indexer(d.index.get_level_values("symbol"))
        uni.SCORE[y] = d[score_col].to_numpy(float)
        uni.ALT[y] = d[alt_col].rank(pct=True).to_numpy() * 0.999
    return uni


# --------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------- #

def build_cov(monthly_returns: pd.DataFrame, uni: Universe, *,
              window_m: int = 36, min_obs: int = 12, shrink: float = 0.30,
              as_of_month: Callable[[int], str] | None = None) -> dict:
    """Point-in-time covariance per formation year.

    `monthly_returns` is months x symbols. For formation year `y` the estimator sees only
    the `window_m` months ending ``as_of_month(y)`` (June of `y` by default), so nothing
    from the holding period leaks in.

    Pairwise-complete covariance is shrunk toward the diagonal, names with fewer than
    `min_obs` observations fall back to the cross-sectional median variance with zero
    correlation, and an eigenvalue floor makes the result PSD — which also makes every
    principal submatrix a selection can pick out PSD, so the GMV solver cannot blow up.
    """
    if as_of_month is None:
        def as_of_month(y):
            return f"{y}-06"

    out = {}
    for y in uni.years:
        syms = uni.frame.index.get_level_values("symbol")[uni.frame["__formation_year"] == y]
        cutoff = as_of_month(y)
        months = [m for m in monthly_returns.index if m <= cutoff][-window_m:]
        sub = monthly_returns.loc[months].reindex(columns=syms)

        cov = sub.cov(min_periods=min_obs).to_numpy()
        var = np.diag(cov).copy()
        ok = np.isfinite(var) & (var > 0)
        var[~ok] = np.nanmedian(var[ok])
        sd = np.sqrt(var)

        corr = cov / np.outer(sd, sd)
        corr[~np.isfinite(corr)] = 0.0        # too little overlap -> assume independent
        np.fill_diagonal(corr, 1.0)
        corr = (1 - shrink) * corr + shrink * np.eye(len(corr))

        sig = np.outer(sd, sd) * corr
        eigval, vec = np.linalg.eigh(sig)     # pairwise estimates need not be PSD
        out[y] = (vec * np.maximum(eigval, 1e-6 * var.mean())) @ vec.T
    return out


def gmv_longonly(sigma: np.ndarray) -> np.ndarray:
    """Long-only global minimum variance weights, by active-set elimination.

    Solve the unconstrained GMV on the active set, drop whatever came out negative,
    repeat. The active set shrinks strictly, so this terminates.
    """
    n = len(sigma)
    active = np.arange(n)
    while True:
        w = np.linalg.solve(sigma[np.ix_(active, active)], np.ones(len(active)))
        w /= w.sum()
        if (w >= -1e-10).all():
            out = np.zeros(n)
            out[active] = np.clip(w, 0.0, None)
            return out / out.sum()
        active = active[w > 1e-10]
        if len(active) == 1:
            out = np.zeros(n)
            out[active[0]] = 1.0
            return out


def _weights_equal(uni: Universe, sel: dict, **_) -> dict:
    return {y: np.full(sel[y].shape, 1.0 / sel[y].shape[1]) for y in uni.years}


def _weights_gmv(uni: Universe, sel: dict, *, cov: dict | None = None, **_) -> dict:
    if cov is None:
        raise ValueError("gmv weighting needs cov=... from build_cov()")
    out = {}
    for y in uni.years:
        idx = sel[y]
        w = np.empty(idx.shape)
        for r in range(idx.shape[0]):
            w[r] = gmv_longonly(cov[y][np.ix_(idx[r], idx[r])])
        out[y] = w
    return out


WEIGHTINGS: dict[str, Callable[..., dict]] = {
    "ew": _weights_equal,
    "gmv": _weights_gmv,
}


def weights(uni: Universe, sel: dict, scheme: str, *, cov: dict | None = None) -> dict:
    """``{year: (n_paths, k)}`` weights, each row summing to 1."""
    return WEIGHTINGS[scheme](uni, sel, cov=cov)


# --------------------------------------------------------------------------- #
# portfolio
# --------------------------------------------------------------------------- #

@dataclass
class Portfolio:
    """One selection arm under one weighting scheme, evaluated.

    `metrics` holds per-path arrays of length `n_paths` (plus the (n_paths, T) `annual`
    and `equity` matrices), so a deterministic arm still returns arrays — of length 1.
    """

    arm: str
    weighting: str
    k: int
    n_paths: int
    sel: dict = field(repr=False)
    W: dict = field(repr=False)
    metrics: dict = field(repr=False)

    @property
    def name(self) -> str:
        return f"{self.arm}_{self.weighting}"

    def __getitem__(self, key: str) -> np.ndarray:
        return self.metrics[key]

    def point(self, metric: str) -> float:
        """The reported value of a metric — the mean across paths."""
        return float(self.metrics[metric].mean())

    def annual(self) -> np.ndarray:
        """(T,) expected annual return, averaged over paths."""
        return self.metrics["annual"].mean(axis=0)

    @property
    def years(self) -> np.ndarray:
        return np.array(sorted(self.sel))

    def by_year(self, *, quantiles: Iterable[int] = (5, 95)) -> pd.DataFrame:
        """Year-indexed view — the only frame a formation year can index.

        Only `annual` and `equity` carry a year axis; CAGR, Sharpe, drawdown and turnover
        are properties of a whole path, so they have no place here. For those, see
        `draws` (per path) or `metrics_table` (summarised).
        """
        r = self.metrics["annual"]
        out = {"mean": r.mean(axis=0), "median": np.median(r, axis=0)}
        for q in quantiles:
            out[f"p{q:02d}"] = np.percentile(r, q, axis=0)
        out["equity"] = self.metrics["equity"].mean(axis=0)
        return pd.DataFrame(out, index=pd.Index(self.years, name="formation_year"))

    def draws(self, metrics: Sequence[str] | None = None) -> pd.DataFrame:
        """Path-indexed view — one row per Monte-Carlo path.

        For a null arm this *is* the null distribution, so it is what a histogram or a
        custom percentile should be built from. Defaults to every path-level metric.
        """
        keys = list(metrics) if metrics is not None else [
            m for m, v in self.metrics.items() if v.ndim == 1
        ]
        return pd.DataFrame({m: self.metrics[m] for m in keys},
                            index=pd.RangeIndex(self.n_paths, name="path"))


def _evaluate(uni: Universe, sel: dict, W: dict, *, rf: float = 0.0) -> dict:
    n_paths, T = sel[uni.years[0]].shape[0], uni.T
    r = np.empty((n_paths, T))
    effn = np.empty((n_paths, T))
    maxw = np.empty((n_paths, T))
    wturn = np.empty((n_paths, T - 1))

    rows = np.arange(n_paths)[:, None]
    prev = np.zeros((n_paths, len(uni.symbols)))
    for i, y in enumerate(uni.years):
        w = W[y]
        r[:, i] = (uni.RET[y][sel[y]] * w).sum(axis=1)
        effn[:, i] = 1.0 / (w ** 2).sum(axis=1)
        maxw[:, i] = w.max(axis=1)

        cur = np.zeros((n_paths, len(uni.symbols)))
        cur[rows, uni.CODE[y][sel[y]]] = w
        if i > 0:
            wturn[:, i - 1] = 0.5 * np.abs(cur - prev).sum(axis=1)
        prev = cur

    equity = np.cumprod(1.0 + r, axis=1)
    mdd = (equity / np.maximum.accumulate(equity, axis=1) - 1.0).min(axis=1)
    mean_ann = r.mean(axis=1)
    vol_ann = r.std(axis=1, ddof=1)

    return {"annual": r, "equity": equity,
            "cagr": equity[:, -1] ** (1.0 / T) - 1.0,
            "mean_ann": mean_ann, "vol_ann": vol_ann,
            "sharpe": (mean_ann - rf) / vol_ann,
            "mdd": mdd, "wturn": wturn.mean(axis=1),
            "effn": effn.mean(axis=1), "maxw": maxw.mean(axis=1)}


def build_portfolio(uni: Universe, arm: str, k: int, *, n_sim: int, seed: int,
                    weighting: str = "ew", cov: dict | None = None,
                    rf: float = 0.0) -> Portfolio:
    """Select, weight, evaluate — the one entry point.

    `k` is a parameter, never global state, so a basket-size sweep is a loop over calls.
    Pass the *same* `seed` for a given arm across weightings and the two portfolios share
    their selections exactly, which is what isolates construction from selection.
    """
    sel = select(uni, arm, k, n_sim=n_sim, seed=seed)
    W = weights(uni, sel, weighting, cov=cov)
    return Portfolio(arm=arm, weighting=weighting, k=k,
                     n_paths=sel[uni.years[0]].shape[0],
                     sel=sel, W=W, metrics=_evaluate(uni, sel, W, rf=rf))


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #

# +1 higher is better, -1 lower is better, 0 neither (two-sided).
DIRECTION: dict[str, int] = {
    "cagr": 1, "mean_ann": 1, "sharpe": 1, "mdd": 1,
    "vol_ann": -1, "wturn": 0, "effn": 0, "maxw": 0,
}

_DEFAULT_METRICS = ("cagr", "mean_ann", "vol_ann", "sharpe", "mdd", "wturn", "effn", "maxw")


def _mc_pvalue(point: float, null: np.ndarray, direction: int) -> float:
    """Monte-Carlo p-value with the conservative +1 correction."""
    n = len(null)
    p_hi = (1 + (null >= point).sum()) / (1 + n)
    p_lo = (1 + (null <= point).sum()) / (1 + n)
    if direction > 0:
        return float(p_hi)
    if direction < 0:
        return float(p_lo)
    return float(min(1.0, 2 * min(p_hi, p_lo)))


def _zscore(point: float, null: np.ndarray) -> float:
    """Standardised distance, NaN when the null is degenerate.

    A constant-by-construction metric — `effn` and `maxw` under equal weight — has zero
    spread, so there is no scale to standardise against and no comparison to make.
    """
    sd = null.std(ddof=1) if len(null) > 1 else 0.0
    return float((point - null.mean()) / sd) if sd > 0 else float("nan")


def metrics_table(port: Portfolio, benchmark: Portfolio, *,
                  metrics: Sequence[str] = _DEFAULT_METRICS) -> pd.DataFrame:
    """Portfolio metrics against a benchmark's Monte-Carlo distribution.

    `benchmark` should be the null arm **at the same weighting** — comparing an EW
    strategy to a GMV null would confound selection with construction.
    """
    if port.weighting != benchmark.weighting:
        raise ValueError(
            f"weighting mismatch: {port.name} vs {benchmark.name}. Compare like with like."
        )
    rows = []
    for m in metrics:
        s, z = port[m], benchmark[m]
        point = float(s.mean())
        rows.append({
            "metric": m,
            "value": point,
            "value_sd": float(s.std(ddof=1)) if len(s) > 1 else np.nan,
            "null_mean": float(z.mean()),
            "null_sd": float(z.std(ddof=1)),
            "null_p05": float(np.percentile(z, 5)),
            "null_p95": float(np.percentile(z, 95)),
            "z": _zscore(point, z),
            "p_mc": _mc_pvalue(point, z, DIRECTION.get(m, 0)),
            "win_rate": float((s[:, None] > z[None, :]).mean()),
        })
    return pd.DataFrame(rows).set_index("metric")


def annual_table(port: Portfolio, benchmark: Portfolio,
                 *, quantiles: Iterable[int] = (5, 95)) -> pd.DataFrame:
    """Year by year, against the benchmark's null **for that same year**.

    The ``(n_paths, T)`` matrix of annual returns carries a null distribution along both
    axes, and they answer different questions. Reducing across years first — what
    `metrics_table` does — gives the null of a whole 14-year outcome (CAGR, Sharpe).
    Holding a year fixed instead, as here, gives the null of *that year's* return.

    The second is much the wider of the two: averaging over years cancels noise, so a
    single year's null typically has several times the spread of the CAGR null. A
    portfolio can sit at the 97th percentile of one year and still be unremarkable over
    the full sample.
    """
    if port.weighting != benchmark.weighting:
        raise ValueError(f"weighting mismatch: {port.name} vs {benchmark.name}")

    s, z = port["annual"], benchmark["annual"]
    out = {"value": s.mean(axis=0), "null_mean": z.mean(axis=0)}
    out["spread"] = out["value"] - out["null_mean"]
    for q in quantiles:
        out[f"null_p{q:02d}"] = np.percentile(z, q, axis=0)
    out["pctile"] = [(z[:, i] < out["value"][i]).mean() * 100 for i in range(z.shape[1])]
    out["beat_null"] = out["spread"] > 0
    return pd.DataFrame(out, index=pd.Index(port.years, name="formation_year"))


def paired_test(a: Portfolio, b: Portfolio, label: str | None = None) -> dict:
    """Year-by-year paired t-test on ``a - b``.

    Differencing removes the common market year effect, so this speaks to the edge rather
    than to the market having risen. With annual non-overlapping data it is a low-power
    test — read it alongside `metrics_table`'s `p_mc`, not instead of it.
    """
    d = a.annual() - b.annual()
    res = stats.ttest_1samp(d, 0.0)
    return {
        "test": label or f"{a.name} - {b.name}",
        "mean": float(d.mean()),
        "sd": float(d.std(ddof=1)),
        "T": len(d),
        "t": float(res.statistic),
        "p_two_sided": float(res.pvalue),
        "p_one_sided": float(stats.ttest_1samp(d, 0.0, alternative="greater").pvalue),
        "wins": int((d > 0).sum()),
    }


def _char_paths(uni: Universe, port: Portfolio, values: dict) -> np.ndarray:
    """(n_paths, T) weight-averaged characteristic, renormalising around NaNs."""
    out = np.empty((port.n_paths, uni.T))
    for i, y in enumerate(uni.years):
        v = values[y][port.sel[y]]
        w = port.W[y]
        ok = np.isfinite(v)
        num = (np.where(ok, v, 0.0) * np.where(ok, w, 0.0)).sum(axis=1)
        out[:, i] = num / np.where(ok, w, 0.0).sum(axis=1)
    return out


def _percentiles(uni: Universe, column: str) -> dict:
    col = uni.frame[column].astype(float)
    fy = uni.frame["__formation_year"]
    return {y: (col[fy == y].rank(pct=True) * 100).to_numpy() for y in uni.years}


def _levels(uni: Universe, column: str, log: bool) -> dict:
    col = uni.frame[column].astype(float)
    fy = uni.frame["__formation_year"]
    out = {}
    for y in uni.years:
        v = col[fy == y]
        out[y] = (np.log10(v.where(v > 0)) if log else v).to_numpy(float)
    return out


def characteristics(uni: Universe, chars: Mapping[str, tuple[str, str]],
                    port: Portfolio, benchmark: Portfolio | None = None,
                    *, by_year: bool = False) -> pd.DataFrame:
    """What the portfolio holds, as cross-sectional percentiles and as levels.

    `chars` maps a display name to ``(column, "raw" | "log")``. Percentiles are ranked
    **within each formation year**, so they are comparable across years and immune to the
    market growing; levels are the raw number, log10 for anything spanning orders of
    magnitude.

    With a `benchmark`, adds the tilt against it and a Monte-Carlo p-value on the
    percentile version — the level version is not comparable across years, so it is
    reported but never tested.
    """
    if benchmark is not None and port.weighting != benchmark.weighting:
        raise ValueError(f"weighting mismatch: {port.name} vs {benchmark.name}")

    rows = []
    for name, (column, mode) in chars.items():
        pct = _percentiles(uni, column)
        lvl = _levels(uni, column, log=(mode == "log"))
        f_pct = _char_paths(uni, port, pct)
        f_lvl = _char_paths(uni, port, lvl)

        if by_year:
            base = {"characteristic": name, "weighting": port.weighting}
            for i, y in enumerate(uni.years):
                row = {**base, "formation_year": int(y),
                       "pct": float(f_pct[:, i].mean()),
                       "level": float(f_lvl[:, i].mean())}
                if benchmark is not None:
                    z = _char_paths(uni, benchmark, pct)
                    row["null_pct"] = float(z[:, i].mean())
                    row["tilt"] = row["pct"] - row["null_pct"]
                rows.append(row)
            continue

        row = {"characteristic": name, "weighting": port.weighting,
               "pct": float(f_pct.mean()), "level": float(f_lvl.mean())}
        if benchmark is not None:
            z_paths = _char_paths(uni, benchmark, pct).mean(axis=1)
            point = f_pct.mean(axis=1).mean()
            row["null_pct"] = float(z_paths.mean())
            row["tilt"] = point - float(z_paths.mean())
            row["z"] = _zscore(point, z_paths)
            row["p_mc"] = _mc_pvalue(point, z_paths, 0)
        rows.append(row)

    index = ["characteristic", "formation_year"] if by_year else "characteristic"
    return pd.DataFrame(rows).set_index(index)


def rankable(uni: Universe, column: str, *, min_max_rank: float = 0.95) -> bool:
    """Whether a column can be split into extreme buckets at all.

    A heavily tied column cannot: average-rank percentiles give every tied name the same
    value, so the top bucket comes out empty for *every* portfolio and the chart shows an
    artefact rather than a tilt. `traded_pct` is the live example — most names trade every
    session, so its attainable percentile caps out well below 1.
    """
    col = uni.frame[column].astype(float)
    fy = uni.frame["__formation_year"]
    return max(float(np.nanmax(col[fy == y].rank(pct=True))) for y in uni.years) > min_max_rank


def buckets(uni: Universe, chars: Mapping[str, tuple[str, str]], port: Portfolio,
            *, n_buckets: int = 5, by: str = "weight",
            skip_unrankable: bool = True) -> pd.DataFrame:
    """Distribution of the portfolio across universe buckets — the shape, not the centre.

    A mean percentile of 65 is equally consistent with thirty moderately-strong names and
    with fifteen extreme ones paired with fifteen average ones. This shows which.

    ``by="weight"`` reports weight share x k, so the scale stays "equivalent names" and
    EW and GMV are directly comparable; under EW it collapses to the name count exactly.
    ``by="name"`` counts holdings and is therefore weighting-blind — use it to describe
    what was *selected*, not what is *held*.
    """
    if by not in ("weight", "name"):
        raise ValueError("by must be 'weight' or 'name'")

    rows = []
    for name, (column, _) in chars.items():
        if skip_unrankable and not rankable(uni, column):
            continue
        col = uni.frame[column].astype(float)
        fy = uni.frame["__formation_year"]
        counts = np.zeros((port.n_paths, uni.T, n_buckets))
        for i, y in enumerate(uni.years):
            v = col[fy == y]
            q = np.ceil(v.rank(pct=True) * n_buckets).clip(1, n_buckets).to_numpy()
            q = np.where(v.notna().to_numpy(), q, np.nan)[port.sel[y]]
            w = port.W[y] * port.k if by == "weight" else np.ones(port.sel[y].shape)
            for b in range(1, n_buckets + 1):
                counts[:, i, b - 1] = np.where(q == b, w, 0.0).sum(axis=1)
        mean = counts.mean(axis=(0, 1))
        rows.append({"characteristic": name, "arm": port.arm, "weighting": port.weighting,
                     **{f"Q{b + 1}": float(mean[b]) for b in range(n_buckets)},
                     "held": float(mean.sum())})
    return pd.DataFrame(rows).set_index(["characteristic", "arm", "weighting"])


def capacity(uni: Universe, port: Portfolio, adtv_col: str = "adtv_vnd", *,
             participation: float = 0.10, days: int = 5,
             scale: float = 1e9) -> pd.DataFrame:
    """How much money the portfolio holds, per formation year.

    A position weighted `w` in a fund of size `A` means buying ``w * A`` of a stock that
    trades `adtv` per day. Allowing `participation` of daily volume over `days` sessions
    caps the fund at ``adtv * participation * days / w``, and the portfolio is capped by
    its **worst** name since every position has to be filled.

    `binding` is that minimum; `median` repeats it at the median holding, so the gap
    between the two shows how much of the constraint is one name. Zero-weight names are
    not constraints and drop out — which matters under GMV, where most of them are.
    """
    adtv = uni.column(adtv_col)
    binding = np.empty((port.n_paths, uni.T))
    median = np.empty_like(binding)
    for i, y in enumerate(uni.years):
        a = adtv[y][port.sel[y]]
        w = port.W[y]
        cap = np.where(w > 1e-9, a * participation * days / np.maximum(w, 1e-12), np.inf)
        binding[:, i] = cap.min(axis=1)
        median[:, i] = np.nanmedian(np.where(np.isfinite(cap), cap, np.nan), axis=1)
    return pd.DataFrame({"binding": binding.mean(axis=0) / scale,
                         "median": median.mean(axis=0) / scale},
                        index=pd.Index(uni.years, name="formation_year"))


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #

def plot_null(port: Portfolio, benchmark: Portfolio, metric: str = "cagr",
              *, ax=None, bins: int = 60, label: str | None = None):
    """Null histogram with the strategy marked on it.

    The shaded band is the strategy's own 5-95% spread. It is not decoration: with a
    random tiebreak the strategy is a distribution too, and a point estimate sitting
    outside the null means much less if its own spread straddles it.
    """
    import matplotlib.pyplot as plt

    if port.weighting != benchmark.weighting:
        raise ValueError(f"weighting mismatch: {port.name} vs {benchmark.name}")
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.4))

    s, z = port[metric], benchmark[metric]
    point = float(s.mean())
    ax.hist(z, bins=bins, color="steelblue", alpha=0.75, label=f"null ({benchmark.arm})")
    if len(s) > 1:
        ax.axvspan(np.percentile(s, 5), np.percentile(s, 95), color="crimson", alpha=0.15)
    ax.axvline(point, color="crimson", lw=2.2, label=label or port.arm)

    row = metrics_table(port, benchmark, metrics=[metric]).loc[metric]
    ax.set_title(f"{metric} — {port.weighting.upper()}\n"
                 f"z = {row['z']:.2f},  p = {row['p_mc']:.4f}", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return ax


def plot_equity(port: Portfolio, benchmark: Portfolio | None = None, *, ax=None,
                quantiles: tuple[int, int] = (5, 95), log: bool = True):
    """Growth of 1, from the first formation date to the last exit.

    Curves are the mean across paths and the x axis is labelled by **exit** year, since
    equity at position `i` is what a portfolio formed in year `i` is worth once its
    holding period closes.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4.2))
    years = port.years
    x = np.arange(len(years) + 1)
    ticks = [str(years[0])] + [str(y + 1) for y in years]

    if benchmark is not None:
        lo = np.concatenate([[1.0], np.percentile(benchmark["equity"], quantiles[0], axis=0)])
        hi = np.concatenate([[1.0], np.percentile(benchmark["equity"], quantiles[1], axis=0)])
        ax.fill_between(x, lo, hi, alpha=0.25, color="steelblue",
                        label=f"null {quantiles[0]}-{quantiles[1]}%")
        med = np.concatenate([[1.0], np.median(benchmark["equity"], axis=0)])
        ax.plot(x, med, lw=1.6, ls="--", color="steelblue", label="null median")

    curve = np.concatenate([[1.0], port["equity"].mean(axis=0)])
    ax.plot(x, curve, lw=2.2, color="crimson", label=f"{port.arm} ({port.weighting})")

    if log:
        ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(ticks, rotation=45, fontsize=8)
    ax.set_ylabel("growth of 1")
    ax.set_title(f"Cumulative return, 1 VND at 30/6/{years[0]}"
                 f"{' (log scale)' if log else ''}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return ax


# --------------------------------------------------------------------------- #
# smoke test — `python portfolio_util.py` from empirical_runs/
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import sqlite3
    from contextlib import closing
    from pathlib import Path

    BASE = Path("../data/preprocessing_pipeline_results")
    DB = "../fscore.db"
    K, N_SIM, SEED = 30, 2000, 20260810

    panel = pd.read_csv(BASE / "final_panel.csv").set_index(["symbol", "period"])
    keep = (panel["tradeable"].eq(True) & panel["f_score"].notna()
            & panel["bm"].notna() & panel["fwd_return_1y"].notna())

    per = pd.Series(panel.index.get_level_values("period"), index=panel.index)
    panel["mom_12m"] = panel["fwd_return_1y"].groupby(level="symbol").shift(1).where(
        (per - per.groupby(level="symbol").shift(1)).eq(1))
    panel["mktcap"] = panel["shares_issued"] * panel["close_raw"]

    uni = build_universe(panel.loc[keep])
    print(f"universe: {len(uni.frame):,} rows, {uni.years[0]}-{uni.years[-1]} "
          f"(T = {uni.T}), {len(uni.symbols):,} symbols")

    MONTHLY_SQL = """
    select symbol, substr(date,1,7) as month, price_close*unit/adj_ratio as close_adj
    from (select symbol, date, price_close, unit, adj_ratio,
                 row_number() over (partition by symbol, substr(date,1,7)
                                    order by date desc) rn
          from fireant_prices where unit=1000 and price_close>0 and adj_ratio>0)
    where rn=1
    """
    with closing(sqlite3.connect(f"file:{DB}?mode=ro", uri=True)) as conn:
        month_end = pd.read_sql(MONTHLY_SQL, conn)
    mret = (month_end.pivot(index="month", columns="symbol", values="close_adj")
            .sort_index().pct_change(fill_method=None))
    cov = build_cov(mret, uni)

    SEEDS = {"f_rand": SEED, "random": SEED + 1}
    P = {(arm, wt): build_portfolio(uni, arm, K, weighting=wt, cov=cov,
                                    n_sim=N_SIM, seed=SEEDS[arm])
         for arm in ("f_rand", "random") for wt in ("ew", "gmv")}

    for wt in ("ew", "gmv"):
        print(f"\n=== {wt.upper()} — f_rand vs random, K={K}, N={N_SIM} ===")
        print(metrics_table(P[("f_rand", wt)], P[("random", wt)]).round(4).to_string())

    print("\n=== paired tests ===")
    print(pd.DataFrame([
        paired_test(P[("f_rand", "ew")], P[("random", "ew")], "selection under EW"),
        paired_test(P[("f_rand", "gmv")], P[("random", "gmv")], "selection under GMV"),
        paired_test(P[("f_rand", "gmv")], P[("f_rand", "ew")], "construction (GMV - EW)"),
    ]).set_index("test").round(4).to_string())

    CHARS = {"size": ("mktcap", "log"), "value": ("bm", "raw"),
             "momentum": ("mom_12m", "raw"), "adtv": ("adtv_vnd", "log"),
             "turnover": ("turnover", "raw"), "traded_pct": ("traded_pct", "raw")}
    for wt in ("ew", "gmv"):
        print(f"\n=== characteristics — {wt.upper()} ===")
        print(characteristics(uni, CHARS, P[("f_rand", wt)],
                              P[("random", wt)]).round(3).to_string())

    print("\n=== buckets, by weight (equivalent names out of 30) ===")
    print(pd.concat([buckets(uni, CHARS, P[("f_rand", wt)]) for wt in ("ew", "gmv")])
          .round(2).to_string())

    print("\n=== capacity, bn VND ===")
    print(pd.concat({wt: capacity(uni, P[("f_rand", wt)]) for wt in ("ew", "gmv")},
                    axis=1).round(3).to_string())
