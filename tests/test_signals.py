"""Smoke test: nine signals compute and bound correctly on demo data."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from fscore.data import make_demo_market
from fscore.signal import piotroski_signals, SIGNALS


def test_eq_offer_uses_cash_flow_not_share_count():
    """EQ_OFFER must read the cash-flow issuance line. A buyback that shrinks
    the share count while the firm also raised equity is the case the old
    share-count test got wrong: it scored 1 (no issuance) when the firm did
    in fact issue."""
    import pandas as pd
    from fscore.signal import piotroski_signals

    def rows(fy, shares, issued):
        return pd.DataFrame({
            "ticker": ["BUYBACK", "CLEAN"], "fiscal_year": fy,
            "total_assets": [1000.0, 1000.0], "net_income": [50.0, 50.0],
            "cfo": [80.0, 80.0], "long_term_debt": [100.0, 100.0],
            "current_assets": [400.0, 400.0], "current_liabilities": [200.0, 200.0],
            "shares_outstanding": shares, "equity_issued": issued,
            "revenue": [900.0, 900.0], "cogs": [500.0, 500.0],
        })

    fund = pd.concat([rows(2022, [100.0, 100.0], [0.0, 0.0]),
                      # BUYBACK repurchased below its issuance -> share count FELL
                      # even though it raised equity; CLEAN raised nothing.
                      rows(2023, [95.0, 95.0], [40.0, 0.0])], ignore_index=True)
    out = piotroski_signals(fund, year=2023).set_index("ticker")

    assert out.loc["BUYBACK", "no_issuance"] == 0   # issued -> no point
    assert out.loc["CLEAN", "no_issuance"] == 1     # did not issue -> point
    assert out.attrs["eq_offer_from_cashflow"] == 2
    assert out.attrs["eq_offer_from_shares"] == 0


def test_eq_offer_falls_back_to_shares_when_cashflow_absent():
    import pandas as pd
    from fscore.signal import piotroski_signals

    fund, _ = make_demo_market(n_stocks=20, year=2023, seed=3)
    out = piotroski_signals(fund.drop(columns=["equity_issued"]), year=2023)
    assert out.attrs["eq_offer_from_cashflow"] == 0
    assert out.attrs["eq_offer_from_shares"] == len(out)


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
