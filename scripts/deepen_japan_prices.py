"""Re-fetch the Japanese names whose cached price history is too short.

The Japan price cache was assembled in two passes, and the second pass asked
Yahoo for a 2020 start, so 107 of its 236 symbols carry no history before
2020. Twenty of those are current or former TPX100 constituents - the three
megabanks among them - and a formation in 2012 needs prices back to mid-2009
for its 36-month covariance window. Without them the eligible universe at the
early formations falls to ~84 names, which pushes a 30-name basket to ~88% of
the high-B/M subset and leaves the random-basket null with almost nothing to
distinguish it from the F-Score basket.

This re-fetches only the short symbols, from 2009, and merges them into the
existing cache. Rows already present are kept: the fetch adds history, it does
not restate what is already there.

Run:  python scripts/deepen_japan_prices.py
"""
from __future__ import annotations

import pathlib
import sys
import time

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fscore.data.bbg_processed import constituents  # noqa: E402

DATA = ROOT / "data"
CACHE = DATA / "japan_prices.csv.gz"
START = "2009-01-01"
NEED_BY = pd.Timestamp("2009-07-01")


def main() -> None:
    import yfinance as yf

    px = pd.read_csv(CACHE, parse_dates=["date"])
    first = px.groupby("ticker").date.min()

    cons = constituents("japan", DATA, range(2012, 2025))
    universe = set().union(*cons.values())
    targets = sorted(t for t in universe
                     if t not in first.index or first[t] > NEED_BY)
    print(f"{len(targets)} symbols need deeper history: {targets}")

    frames, failed = [], []
    for i, tk in enumerate(targets, 1):
        try:
            d = yf.Ticker(tk).history(start=START, auto_adjust=False,
                                      actions=False)
        except Exception as exc:                       # network / symbol gone
            failed.append((tk, f"{type(exc).__name__}: {exc}"))
            continue
        if d is None or d.empty:
            failed.append((tk, "empty"))
            continue
        d = d.reset_index()
        d["date"] = pd.to_datetime(d["Date"]).dt.tz_localize(None)
        out = pd.DataFrame({
            "date": d["date"], "ticker": tk,
            "adj_close": d["Adj Close"] if "Adj Close" in d else d["Close"],
            "close_raw": d["Close"], "volume": d["Volume"]})
        frames.append(out)
        print(f"  [{i:>2}/{len(targets)}] {tk}: {len(out)} rows "
              f"from {out.date.min():%Y-%m-%d}", flush=True)
        time.sleep(0.4)

    if not frames:
        print("nothing fetched; cache unchanged")
        return

    new = pd.concat(frames, ignore_index=True)
    merged = (pd.concat([px, new], ignore_index=True)
                .drop_duplicates(subset=["ticker", "date"], keep="first")
                .sort_values(["ticker", "date"]))
    merged.to_csv(CACHE, index=False, compression="gzip")

    after = merged.groupby("ticker").date.min()
    fixed = sum(1 for t in targets if t in after.index and after[t] <= NEED_BY)
    print(f"\nmerged: {len(px)} -> {len(merged)} rows")
    print(f"symbols now reaching {NEED_BY:%Y-%m}: {fixed}/{len(targets)}")
    if failed:
        print(f"still unavailable ({len(failed)}):")
        for tk, why in failed:
            print(f"   {tk}: {why[:60]}")


if __name__ == "__main__":
    main()
