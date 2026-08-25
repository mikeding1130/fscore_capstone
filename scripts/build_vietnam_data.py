"""Build the Vietnam data cache from the sibling preprocessing repository.

The US and Japan caches are written by `fetch_us_edgar.py` / `fetch_us_japan.py`
off a vendor API. Vietnam has no such API in this study: its statements come
from `../thesis`, which crawls three Vietnamese sources into `fscore.db`,
reconciles them, and writes a checked per-firm-year panel. That repository
already ships the price panel and the score panel; this script writes the
three files the MAIN study additionally needs, in the same canonical schemas
the other two markets use:

    data/vietnam_fundamentals.csv     canonical statement lines + FY-end market cap
    data/vietnam_sectors.csv          ticker -> sector
    data/vietnam_benchmarks.csv.gz    VN30 and VNINDEX levels
    data/vietnam_exclusions.csv       what the source holds vs what is scored

Run:  python scripts/build_vietnam_data.py
      python scripts/build_vietnam_data.py --thesis /path/to/thesis
"""
from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd                                        # noqa: E402

from fscore.data import vietnam as vn                      # noqa: E402
from fscore.data.loaders import REQUIRED_FUND_COLS         # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--thesis", default=str(vn.THESIS_DIR),
                    help="checkout of the preprocessing repository")
    ap.add_argument("--out", default=str(ROOT / "data"))
    args = ap.parse_args()

    thesis = pathlib.Path(args.thesis)
    paths = vn.build_cache(
        out_dir=args.out,
        pipeline_dir=thesis / "data" / "preprocessing_pipeline_results",
        db_path=thesis / "fscore.db",
        sector_csv=thesis / "data" / "tickers_non_financials_sectors.csv")

    fund = pd.read_csv(paths["fundamentals"], parse_dates=["report_date"])
    missing = [c for c in REQUIRED_FUND_COLS if c not in fund.columns]
    assert not missing, f"fundamentals missing canonical columns: {missing}"
    assert not fund.duplicated(["ticker", "fiscal_year"]).any(), \
        "fundamentals are not unique on (ticker, fiscal_year)"

    bench = pd.read_csv(paths["benchmarks"], parse_dates=["date"])
    drops = pd.read_csv(paths["exclusions"])
    sectors = pd.read_csv(paths["sectors"])

    print(f"fundamentals : {len(fund):,} firm-years, {fund.ticker.nunique()} tickers, "
          f"FY{fund.fiscal_year.min()}-FY{fund.fiscal_year.max()}  -> {paths['fundamentals'].name}")
    print(f"sectors      : {len(sectors):,} tickers, {sectors.sector.nunique()} labels"
          f"  -> {paths['sectors'].name}")
    for tk, g in bench.groupby("ticker"):
        print(f"benchmark    : {tk:<8} {g.date.min():%Y-%m-%d} -> {g.date.max():%Y-%m-%d}"
              f" ({len(g):,} sessions)")
    kept = drops.scored.sum()
    print(f"exclusions   : {drops.rows_in_source.sum():,} firm-years in the source -> "
          f"{kept:,} scored ({100 * kept / drops.rows_in_source.sum():.1f}%)"
          f"  -> {paths['exclusions'].name}")
    print("\nnote: VN30/VNINDEX are CAPITAL indices (no cash dividends); the "
          "portfolios are total-return. See fscore.data.vietnam.")


if __name__ == "__main__":
    main()
