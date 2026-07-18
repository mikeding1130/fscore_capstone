"""One-shot data download for the developed pair (US, Japan).

Writes git-ignored caches under data/:
  {market}_fundamentals.csv   canonical annual statements + market_cap
  {market}_prices.csv.gz      adjusted daily closes + volume (2022-01 ..)
  {market}_sectors.csv        constituent list with sector labels
  {market}_benchmarks.csv.gz  investable benchmarks (+ FX for Japan)

Run:  python scripts/fetch_us_japan.py [us|japan|both]
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from fscore.data.yahoo import fetch_and_cache  # noqa: E402

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    root = pathlib.Path(__file__).resolve().parents[1]
    for market in (["us", "japan"] if which == "both" else [which]):
        print(f"=== fetching {market} ===", flush=True)
        fetch_and_cache(market, cache_dir=root / "data")
    print("done.", flush=True)
