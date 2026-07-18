"""Build the long-history US cache (SEC EDGAR + Yahoo prices, 2008-).

Replaces the Yahoo-only US cache with:
  us_fundamentals.csv   EDGAR XBRL annual statements FY2008.., true filing
                        dates in report_date, first-reported values
  us_prices.csv.gz      adjusted daily closes + volume from 2008-01
  us_membership.csv     S&P 500 members as of each July-1 formation date
  us_sectors.csv        augmented with historical members (best effort)
  us_benchmarks.csv.gz  SPY / VTV from 2008-01

Run:  python scripts/fetch_us_edgar.py
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from fscore.data.edgar import fetch_and_cache_us_long  # noqa: E402

if __name__ == "__main__":
    root = pathlib.Path(__file__).resolve().parents[1]
    fetch_and_cache_us_long(cache_dir=root / "data", years=range(2011, 2026))
    print("done.", flush=True)
