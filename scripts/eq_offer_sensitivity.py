"""How much does the EQ_OFFER measure matter? A US test, and a bound for Japan.

Piotroski's eighth signal asks whether the firm issued common equity. The
faithful measure is the cash-flow statement's equity-issuance line; the
common substitute is "did the share count rise?". They disagree often,
because buybacks net against issuance and splits move the count without
raising capital.

Japan has to use the share-count substitute. The Bloomberg tree was expected
to end that — it ships a `Proceeds_Issuance_Common_Stock` column — but the
column is empty in every sheet of both markets, so nothing changed. The United
States is not stuck: SEC EDGAR supplies `ProceedsFromIssuanceOfCommonStock`
from FY2009, so the US sample can be scored both ways and the disagreement
measured. That is what puts a number on the assumption Japan is forced into.

The substitution is not neutral, and its sign matters for reading Japan's
results. Share counts miss issuance that buybacks net away, so the substitute
scores firms **more generously**: on the US sample the mean F-Score is 6.08 by
share count against 5.76 by cash flow, and the strategy's placement against
random baskets is flattered by roughly 14 percentile points. Japan's headline
therefore sits on the generous side of that gap, and the report should say so
rather than quote its percentile unqualified.

Run:  python scripts/eq_offer_sensitivity.py
Writes results/eq_offer_sensitivity.csv and results/eq_offer_headline.csv
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fscore.data.edgar import load_membership          # noqa: E402
from fscore.data.yahoo import load_cached              # noqa: E402
from fscore.pipeline import run_study                  # noqa: E402
from fscore.signal import piotroski_signals            # noqa: E402

YEARS = list(range(2012, 2026))
END_CAP = pd.Timestamp("2025-12-31")


def signal_level(fund: pd.DataFrame) -> pd.DataFrame:
    """Per formation year: how often the two measures disagree."""
    rows = []
    for year in range(YEARS[0] - 1, YEARS[-1]):
        snap = fund[fund.fiscal_year.isin([year, year - 1])]
        cash = piotroski_signals(snap, year=year)
        share = piotroski_signals(snap.drop(columns=["equity_issued"]), year=year)
        m = cash[["ticker", "no_issuance", "fscore"]].merge(
            share[["ticker", "no_issuance", "fscore"]],
            on="ticker", suffixes=("_cash", "_share"))
        rows.append({
            "score_year": year,
            "firms": len(m),
            "cash_line_available": cash.attrs["eq_offer_from_cashflow"],
            "signal_flips": int((m.no_issuance_cash != m.no_issuance_share).sum()),
            "mean_fscore_cash": m.fscore_cash.mean(),
            "mean_fscore_share": m.fscore_share.mean(),
        })
    d = pd.DataFrame(rows)
    d["flip_rate"] = d.signal_flips / d.firms
    return d


def study_level(fund, prices, sectors, membership) -> pd.DataFrame:
    """Does the disagreement survive into the reported conclusions?"""
    out = {}
    for label, f in [("cash_flow", fund),
                     ("share_count", fund.drop(columns=["equity_issued"]))]:
        st = run_study("us", f, prices, sectors, YEARS, n_mc=400, n_mc_opt=120,
                       lag_months=1, membership=membership, end_cap=END_CAP,
                       seed=42)
        s, pl = st.summary(), st.placement("fscore_EW", "EW")
        out[label] = {
            "ann_return": s.loc["fscore_EW", "ann_return"],
            "sharpe": s.loc["fscore_EW", "sharpe"],
            "net_sharpe": s.loc["fscore_EW", "net_sharpe"],
            "percentile_vs_random": pl.loc["sharpe", "percentile"],
            "p_value": pl.loc["sharpe", "p_value"],
            "significant_at_5pct": bool(pl.loc["sharpe", "significant"]),
        }
    return pd.DataFrame(out).T


def main() -> None:
    fund, prices, sectors, _ = load_cached("us", ROOT / "data")
    if "equity_issued" not in fund.columns:
        raise SystemExit("cache predates equity_issued — rerun fetch_us_edgar.py")
    membership = load_membership(ROOT / "data")

    per_year = signal_level(fund)
    head = study_level(fund, prices, sectors, membership)

    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    per_year.to_csv(res / "eq_offer_sensitivity.csv", index=False)
    head.to_csv(res / "eq_offer_headline.csv")

    n, flips = per_year.firms.sum(), per_year.signal_flips.sum()
    print(per_year.round(3).to_string(index=False))
    print(f"\nsignal disagreement: {flips} of {n} firm-years ({flips / n:.1%})")
    print(f"mean F-Score: {per_year.mean_fscore_cash.mean():.2f} (cash flow) "
          f"vs {per_year.mean_fscore_share.mean():.2f} (share count)")
    print(f"\n{head.round(3).to_string()}")
    print(f"\nSharpe gap: "
          f"{abs(head.loc['cash_flow','sharpe'] - head.loc['share_count','sharpe']):.3f}; "
          f"percentile gap: "
          f"{abs(head.loc['cash_flow','percentile_vs_random'] - head.loc['share_count','percentile_vs_random']) * 100:.0f}pp")
    print(f"saved -> {res / 'eq_offer_sensitivity.csv'}")


if __name__ == "__main__":
    main()
