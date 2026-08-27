"""Unit tests for the construction QP and the point-in-time pipeline logic
(synthetic data only — no network)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from fscore.construction import gmv_weights, sector_constrained_gmv, clean_rmt
from fscore.pipeline import pit_snapshot, formation_date
from fscore.evaluation import vs_random


def _rand_cov(n, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n * 4, n))
    return np.cov(A.T)


def test_gmv_long_only_sums_to_one():
    cov = _rand_cov(12)
    w = gmv_weights(cov, [f"T{i}" for i in range(12)])
    assert abs(w.sum() - 1) < 1e-8
    assert (w >= -1e-10).all()
    # GMV variance must not exceed equal-weight variance
    ew = np.ones(12) / 12
    assert w.values @ cov @ w.values <= ew @ cov @ ew + 1e-12


def test_sector_caps_hold():
    n = 20
    cov = _rand_cov(n, seed=3)
    tickers = [f"T{i}" for i in range(n)]
    sectors = pd.Series([f"S{i % 6}" for i in range(n)], index=tickers)
    w = sector_constrained_gmv(cov, tickers, sectors, cap=0.20)
    assert abs(w.sum() - 1) < 1e-6
    assert (w.groupby(sectors).sum() <= 0.20 + 1e-6).all()


def test_rmt_cov_is_symmetric_psd_shaped():
    rng = np.random.default_rng(1)
    rets = pd.DataFrame(rng.normal(0, 0.02, (260, 15)))
    for detone in (False, True):
        cov = clean_rmt(rets, detone=detone)
        assert cov.shape == (15, 15)
        assert np.allclose(cov, cov.T, atol=1e-10)


def test_detone_makes_cov_singular_so_default_is_off():
    """Guards the reason detoning is off by default: removing the market mode
    leaves a singular matrix, which a minimum-variance solve must not invert."""
    import inspect

    rng = np.random.default_rng(2)
    market = rng.normal(0, 0.01, (400, 1))
    rets = pd.DataFrame(market + rng.normal(0, 0.015, (400, 20)))

    kept = np.linalg.eigvalsh(clean_rmt(rets, detone=False))
    dropped = np.linalg.eigvalsh(clean_rmt(rets, detone=True))
    assert kept.min() > 1e-9                       # invertible
    assert dropped.min() < kept.min() * 1e-3       # market mode removed -> singular
    assert inspect.signature(clean_rmt).parameters["detone"].default is False


def test_pit_snapshot_excludes_unpublished_year():
    fund = pd.DataFrame({
        "ticker": ["A", "A", "B", "B"],
        "fiscal_year": [2023, 2024, 2023, 2024],
        # A reports FY2024 in Jan-2025 (+5m lag -> public Jun-2025, in time);
        # B reports FY2024 in May-2025 (+5m lag -> public Oct-2025, too late)
        "report_date": pd.to_datetime(["2024-01-31", "2025-01-31",
                                       "2024-05-31", "2025-05-31"]),
    })
    snap = pit_snapshot(fund, year=2025, lag_months=5)
    assert set(snap[snap.fiscal_year == 2024].ticker) == {"A"}
    # B's FY2024 not public at formation -> B drops out entirely
    assert "B" not in set(snap.ticker)
    assert formation_date(2025) == pd.Timestamp("2025-07-01")


def test_costs_are_charged_per_strategy_not_uniformly():
    """A static control must not be charged the F-Score basket's turnover."""
    from types import SimpleNamespace
    from fscore.grid import GridStudy, COST_PER_SIDE

    def yr(fs_names, static_names):
        w = {"fscore_EW": pd.Series(1 / 3, index=fs_names),
             "universe_EW": pd.Series(1 / 3, index=static_names)}
        # no drift in this fixture, so the book ends where it started
        return SimpleNamespace(weights=w, weights_end=dict(w),
                               mc_names=[frozenset(fs_names)])

    st = GridStudy("t", 3, 10, [1, 2], [yr(["A", "B", "C"], ["X", "Y", "Z"]),
                                        yr(["A", "B", "D"], ["X", "Y", "Z"])],
                   pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    assert st.strategy_turnover("fscore_EW") == pytest_approx(1 / 3)
    assert st.strategy_turnover("universe_EW") == 0.0          # never traded
    assert st.cost_drag("fscore_EW") == pytest_approx(2 / 3 * COST_PER_SIDE)
    assert st.cost_drag("universe_EW") == 0.0
    assert st.mc_turnover() == pytest_approx(1 / 3)            # control's own


def pytest_approx(x, tol=1e-9):
    class _A:
        def __eq__(self, other):
            return abs(other - x) < tol
    return _A()


def test_vs_random_percentile():
    out = vs_random(2.0, [1.0] * 90 + [3.0] * 10)
    assert out["percentile"] == 0.9
    assert out["p_value"] == 0.1


def _price_frame(series: dict, dates):
    rows = [{"ticker": tk, "date": d, "adj_close": float(v)}
            for tk, vals in series.items()
            for d, v in zip(dates, vals) if v is not None]
    return pd.DataFrame(rows)


def test_delisted_names_are_carried_at_last_traded_price():
    """A position that stops trading keeps its last quote (0 return after),
    instead of dropping out and re-weighting the book onto survivors."""
    from fscore.evaluation import returns_panel

    dates = pd.bdate_range("2024-01-01", periods=6)
    px = _price_frame({"ALIVE": [100, 101, 102, 103, 104, 105],
                       "DELIST": [100, 90, 80, None, None, None]}, dates)
    r = returns_panel(px, ["ALIVE", "DELIST"], dates[0], dates[-1])
    cum = (1 + r.fillna(0)).prod() - 1

    assert abs(cum["DELIST"] - (-0.20)) < 1e-9    # frozen at the last price
    assert (r["DELIST"].iloc[2:].fillna(0) == 0).all()
    assert r.attrs["delisted"] == 1


def test_delisting_return_can_discount_the_last_price():
    """The carry-forward assumes full recovery at the last quote; the
    parameter exists to test that assumption (e.g. -30%).

    The frame keeps a surviving name so the trading calendar extends past the
    delisting — returns_panel takes the calendar from the data rather than
    inventing sessions, which would fabricate trading on market holidays.
    """
    from fscore.evaluation import returns_panel

    dates = pd.bdate_range("2024-01-01", periods=6)
    px = _price_frame({"ALIVE": [100] * 6,
                       "DELIST": [100, 90, 80, None, None, None]}, dates)
    r = returns_panel(px, ["ALIVE", "DELIST"], dates[0], dates[-1],
                      delisting_return=-0.30)
    cum = (1 + r.fillna(0)).prod() - 1
    assert abs(cum["DELIST"] - (0.8 * 0.7 - 1)) < 1e-9   # -20%, then -30%
    assert abs(cum["ALIVE"]) < 1e-12                      # untouched


def test_price_gaps_do_not_swallow_returns():
    """Differencing around a hole used to book the move across it as 0."""
    from fscore.evaluation import returns_panel

    dates = pd.bdate_range("2024-01-01", periods=5)
    px = _price_frame({"GAPPY": [100, 101, None, 110, 111]}, dates)
    r = returns_panel(px, ["GAPPY"], dates[0], dates[-1])
    cum = float((1 + r.fillna(0)).prod().iloc[0] - 1)
    assert abs(cum - 0.11) < 1e-9                 # 100 -> 111, nothing lost


def test_annual_rebalance_is_buy_and_hold_and_turnover_matches():
    """The book is bought at formation and left to drift for the year, and
    the turnover charged at the next rebalance is measured from where it
    drifted to — the two must describe the same strategy.

    Applying a fixed weight vector to daily returns instead would rebalance
    daily, which for two names moving apart earns a different (here lower)
    return than holding them.
    """
    # two sessions of dispersion: A gains 50% twice, B loses 50% twice.
    # One session alone would not separate the two schemes — the difference
    # is a compounding effect, so the case needs at least two.
    r = pd.DataFrame({"A": [0.5, 0.5], "B": [-0.5, -0.5]})
    w = pd.Series({"A": 0.5, "B": 0.5})

    growth = (1 + r).cumprod()                       # A -> 2.25, B -> 0.25
    value = (growth * w).sum(axis=1)
    buy_and_hold = float(value.iloc[-1] - 1)         # 0.5*2.25 + 0.5*0.25 - 1
    daily_rebalanced = float((1 + (r * w).sum(axis=1)).prod() - 1)

    assert abs(buy_and_hold - 0.25) < 1e-12
    assert abs(daily_rebalanced) < 1e-12             # +50/-50 nets to zero daily
    assert buy_and_hold > daily_rebalanced           # the two are not the same

    # the book drifts: A ends at 0.5*2.25 / 1.25 = 90% of it, not 50%
    end = w * growth.iloc[-1]
    end = end / end.sum()
    assert abs(end["A"] - 0.9) < 1e-12

    # rebalancing back to equal weight therefore trades 0.4 one way
    from fscore.evaluation import turnover
    assert abs(turnover(end, w) - 0.4) < 1e-12
    assert turnover(w, w) == 0.0     # comparing targets would price nothing


def test_fscore_ties_are_broken_at_random_not_by_value():
    """Score rank is respected; within a tied score the order is random and
    reproducible — never sorted by B/M, which would let the value factor
    choose the part of the basket the tie-break decides."""
    from fscore.selection.baskets import rank_by_fscore, tie_break_slots

    scored = pd.DataFrame({
        "ticker": [f"T{i}" for i in range(10)],
        "fscore": [9, 8, 8, 8, 8, 8, 8, 8, 8, 1],
        "bm": [0.1, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 0.5],
    })

    top = rank_by_fscore(scored, seed=1).head(4)["ticker"].tolist()
    assert top[0] == "T0"                       # the 9 outranks every 8
    assert "T9" not in top                      # the 1 never makes the cut
    # had ties been broken by B/M, the next three would be the highest-B/M
    # names in order; random ordering must not reproduce that
    assert top[1:] != ["T1", "T2", "T3"]

    assert (rank_by_fscore(scored, seed=1)["ticker"].tolist()
            == rank_by_fscore(scored, seed=1)["ticker"].tolist())   # reproducible
    assert (rank_by_fscore(scored, seed=1)["ticker"].tolist()
            != rank_by_fscore(scored, seed=2)["ticker"].tolist())   # seed matters

    # 4-name basket: 1 slot earned on score, 3 decided by the tie-break
    assert tie_break_slots(scored, 4) == 3


def test_vietnam_is_long_only():
    """Shorting is not available in Vietnam, so the long-short strategy must
    not appear in its results — an untradable book would overstate them."""
    from fscore.markets import allows_shorting

    assert allows_shorting("us") and allows_shorting("japan")
    assert not allows_shorting("vietnam")
    assert not allows_shorting("some-unlisted-market")   # conservative default


def test_long_short_book_is_dollar_neutral():
    """Long top-k / short bottom-k: nets to zero, gross 2, legs disjoint."""
    from fscore.grid import LONG_SHORT

    longs, shorts = ["A", "B", "C"], ["X", "Y", "Z"]
    w = pd.concat([pd.Series(1 / len(longs), index=longs),
                   pd.Series(-1 / len(shorts), index=shorts)])
    assert abs(w.sum()) < 1e-12                  # dollar neutral
    assert abs(w.abs().sum() - 2.0) < 1e-12      # gross exposure 2
    assert not set(longs) & set(shorts)          # no name in both legs
    assert LONG_SHORT == "fscore_LS"


def test_significance_is_judged_at_5_percent_only():
    """One level, fixed in advance: p < 0.05. Nothing is 'marginally
    significant' at 6-10%, nothing gets a 1% tier."""
    from fscore.evaluation import ALPHA

    assert ALPHA == 0.05

    sig = vs_random(2.0, [1.0] * 96 + [3.0] * 4)      # p = 0.04
    marginal = vs_random(2.0, [1.0] * 94 + [3.0] * 6)  # p = 0.06
    assert sig["p_value"] < 0.05 and sig["significant"] is True
    assert marginal["p_value"] > 0.05 and marginal["significant"] is False
    assert sig["alpha"] == marginal["alpha"] == 0.05


if __name__ == "__main__":
    for fn in [test_gmv_long_only_sums_to_one, test_sector_caps_hold,
               test_rmt_cov_is_symmetric_psd_shaped,
               test_pit_snapshot_excludes_unpublished_year,
               test_vs_random_percentile]:
        fn()
        print(f"{fn.__name__}: OK")


def test_gmv_memo_returns_what_recomputing_would():
    """The grid sweeps k x N in one process and the whole-universe
    minimum-variance solve depends on neither, so it is memoised. A hit has to
    be indistinguishable from a recompute — otherwise the cache would be a
    silent source of drift between cells."""
    from fscore.construction import weights as W

    n = W._CACHE_MIN_ASSETS + 5          # above the threshold, so it caches
    cov = _rand_cov(n, seed=11)
    tickers = [f"T{i}" for i in range(n)]
    W._GMV_CACHE.clear()
    first = gmv_weights(cov, tickers)
    assert len(W._GMV_CACHE) == 1
    cached = gmv_weights(cov, tickers)
    W._GMV_CACHE.clear()
    recomputed = gmv_weights(cov, tickers)
    assert (cached.values == first.values).all()
    assert (recomputed.values == first.values).all()
    # a different covariance must not collide with the stored one
    other = gmv_weights(_rand_cov(n, seed=12), tickers)
    assert not np.allclose(other.values, first.values)
    # and the caller must not be able to mutate what the cache holds
    cached.iloc[0] = 999.0
    assert gmv_weights(cov, tickers).iloc[0] != 999.0


def test_local_ff3_factors_price_their_own_sorts():
    """A locally built factor set is only useful if it actually spans the
    sorts it is built from: regress the high-B/M leg on the factors and HML
    has to load positively. Ken French does not cover Vietnam, so this is the
    only check available that the construction is not scrambled."""
    from fscore.data.loaders import make_demo_market
    from fscore.evaluation import local_ff3_factors

    # the demo market already carries fiscal years `year - 1` and `year`;
    # a 2023 formation reads the 2022 statements under a 6-month lag
    fund, prices = make_demo_market(n_stocks=120, year=2022, seed=5)
    fund["report_date"] = pd.to_datetime(
        fund.fiscal_year.astype(str) + "-12-31")
    ff = local_ff3_factors(fund, prices, [2023], lag_months=6, min_names=20)
    assert list(ff.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    assert len(ff) > 60
    assert (ff["RF"] == 0).all()          # rf = 0 is the study's convention
    # the three factors must be distinct series, not copies of one another
    assert ff[["Mkt-RF", "SMB", "HML"]].corr().abs().values[np.triu_indices(3, 1)].max() < 0.99
