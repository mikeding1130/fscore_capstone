"""Smoke test: nine signals compute and bound correctly on demo data."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from fscore.data import make_demo_market
from fscore.signal import piotroski_signals, SIGNALS


def test_fscore_bounds():
    fund, _ = make_demo_market(n_stocks=50, year=2023, seed=1)
    scored = piotroski_signals(fund, year=2023)
    assert len(scored) == 50
    assert set(SIGNALS).issubset(scored.columns)
    assert scored["fscore"].between(0, 9).all()
    assert scored[SIGNALS].isin([0, 1]).all().all()


if __name__ == "__main__":
    test_fscore_bounds()
    print("signal smoke test: OK")
