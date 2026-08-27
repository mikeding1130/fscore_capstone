"""Build Japan's canonical fundamentals cache from the Bloomberg tree.

Japan's study was stuck at two formations because the Yahoo statement cache
began at FY2021 and carried no equity line, so no high-B/M universe could be
formed before 2023. The `data/processed/Japan/` tree fixes both: fiscal years
run from 2000, and `Common_Shareholders_Equity` stands in for the book value
the vendor leaves empty. That is what lets Japan run the same 2012-2024
window as the US.

Two things this cannot fix, both recorded rather than papered over:

  * **Prices.** The vendor price workbook in that tree is empty - every cell,
    both markets - so prices come from the existing Yahoo cache. Yahoo only
    serves symbols that are still listed, so the names it cannot supply are
    disproportionately the delisted ones. Per-year coverage is written to
    `results/japan_bbg_price_coverage.csv` so the size of that gap is visible
    rather than assumed.
  * **Market capitalisation.** `Historical_Market_Cap` is empty too, so it is
    rebuilt as raw close x shares outstanding at the fiscal year end. The raw
    close is deliberate: adjusted closes are restated by later splits and
    dividends, and multiplying one by an as-reported share count gives a
    market cap that is wrong by the cumulative adjustment factor.

Run:  python scripts/build_japan_bbg.py
Writes data/japan_bbg_fundamentals.csv and two coverage reports.
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fscore.data.bbg_processed import (constituents,  # noqa: E402
                                       field_coverage, load_fundamentals)

DATA = ROOT / "data"
RESULTS = ROOT / "results"
MARKET = "japan"

# The headline study forms 2012-2024, but the vendor's sheets start at 2000 and
# the within-country robustness check asks how far back the data really goes.
# Building the wider frame costs seconds and lets `full_period.py` measure that
# span instead of being capped by whatever this script happened to load; the
# headline run simply ignores the extra years.
BUILD_YEARS = list(range(2005, 2025))
HEADLINE_YEARS = list(range(2012, 2025))
YEARS = BUILD_YEARS


def year_end_price(prices: pd.DataFrame, fiscal_year: int) -> pd.Series:
    """Last raw close on or before the fiscal year end, per ticker.

    Bloomberg reports Japanese statements on a March fiscal year, but the
    dataset stamps `Financial_Year` on a calendar basis and the study applies
    its reporting lag on top of that, so the December year end is the
    consistent anchor for the share price behind market cap.
    """
    cutoff = pd.Timestamp(f"{fiscal_year}-12-31")
    window = prices[(prices.date <= cutoff)
                    & (prices.date > cutoff - pd.Timedelta(days=30))]
    if window.empty:
        return pd.Series(dtype=float)
    return window.sort_values("date").groupby("ticker").close_raw.last()


def main() -> None:
    print("loading Bloomberg financials ...")
    f = load_fundamentals(MARKET, DATA, YEARS)
    print(f"  {f.ticker.nunique()} tickers, {len(f)} firm-years, "
          f"FY{f.fiscal_year.min()}-{f.fiscal_year.max()}")

    prices = pd.read_csv(DATA / "japan_prices.csv.gz", parse_dates=["date"])
    have = set(prices.ticker.unique())

    # --- market cap: raw close at fiscal year end x shares outstanding ---
    caps = []
    for fy in sorted(f.fiscal_year.unique()):
        px = year_end_price(prices, int(fy))
        if px.empty:
            continue
        caps.append(pd.DataFrame({"fiscal_year": int(fy), "ticker": px.index,
                                  "px": px.values}))
    cap = pd.concat(caps, ignore_index=True)
    f = f.merge(cap, on=["ticker", "fiscal_year"], how="left")
    f["market_cap"] = f.px * f.shares_outstanding
    f = f.drop(columns=["px"])

    keep = ["ticker", "report_date", "fiscal_year", "total_assets",
            "long_term_debt", "current_assets", "current_liabilities",
            "shares_outstanding", "book_value", "net_income", "revenue",
            "cogs", "cfo", "market_cap", "equity_issued"]
    out = f[[c for c in keep if c in f.columns]].copy()
    out.to_csv(DATA / "japan_bbg_fundamentals.csv", index=False)
    print(f"  wrote {DATA / 'japan_bbg_fundamentals.csv'} ({len(out)} rows)")

    # --- coverage reports ---
    cons = constituents(MARKET, DATA, YEARS)
    rows = []
    for y in YEARS:
        uni = cons[y]
        # the covariance window reaches 36 months back, so a name is only
        # usable at formation y if it is priced well before it
        need_from = pd.Timestamp(f"{y - 3}-07-01")
        px_y = prices[(prices.date >= need_from)
                      & (prices.date < pd.Timestamp(f"{y}-07-01"))]
        deep = set(px_y.ticker.unique())
        rows.append({"formation": y, "constituents": len(uni),
                     "priced_at_all": len(uni & have),
                     "priced_36m_before": len(uni & deep),
                     "pct_priced": round(100 * len(uni & have) / len(uni), 1)})
    cov = pd.DataFrame(rows)
    cov.to_csv(RESULTS / "japan_bbg_price_coverage.csv", index=False)
    print("\nprice coverage of the TPX100 universe (Yahoo cache):")
    print(cov.to_string(index=False))

    field_coverage(MARKET, DATA, YEARS).to_csv(
        RESULTS / "japan_bbg_field_coverage.csv")
    print(f"\nsaved -> {RESULTS / 'japan_bbg_price_coverage.csv'}")


if __name__ == "__main__":
    main()
