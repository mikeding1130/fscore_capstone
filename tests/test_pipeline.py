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
    cov = clean_rmt(rets, detone=True)
    assert cov.shape == (15, 15)
    assert np.allclose(cov, cov.T, atol=1e-10)


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


def test_vs_random_percentile():
    out = vs_random(2.0, [1.0] * 90 + [3.0] * 10)
    assert out["percentile"] == 0.9
    assert out["p_value"] == 0.1


if __name__ == "__main__":
    for fn in [test_gmv_long_only_sums_to_one, test_sector_caps_hold,
               test_rmt_cov_is_symmetric_psd_shaped,
               test_pit_snapshot_excludes_unpublished_year,
               test_vs_random_percentile]:
        fn()
        print(f"{fn.__name__}: OK")
