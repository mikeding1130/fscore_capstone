"""SEC EDGAR XBRL adapter: long-history, point-in-time US fundamentals.

Extends the US sample beyond Yahoo's ~5 statement periods:

  * fundamentals 2009..today from the EDGAR `companyfacts` API (XBRL was
    mandated 2009-2011, so coverage is solid from FY2009 for large caps);
  * `report_date` is the **actual 10-K filing date** (first-reported value of
    each period), so point-in-time discipline needs only a small buffer
    instead of the 5-month lag used when true filing dates are unknown;
  * universe membership per formation date from a maintained historical
    S&P 500 constituent dataset (fja05680/sp500), which removes
    index-inclusion look-ahead. Names whose price history is gone from Yahoo
    (many delistings) still drop out — the residual survivorship documented
    in the README.

Output uses the same canonical schemas as `fscore.data.yahoo`, cached as
`us_fundamentals.csv` / `us_prices.csv.gz` / `us_membership.csv`, so the
pipeline and notebooks are source-agnostic.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .yahoo import _fiscal_year, attach_market_cap, fetch_prices

UA = {"User-Agent": "MScFE capstone research mike.ding.work@gmail.com"}
MEMBERSHIP_URL = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
                  "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv")

# canonical field -> ordered (taxonomy, tag, unit) fallbacks.  Values are
# merged per fiscal-period end across tags because tag usage changed over the
# years (e.g. Revenues -> RevenueFromContractWithCustomer... after ASC 606).
FLOW_TAGS = {
    "net_income": [("us-gaap", "NetIncomeLoss"), ("us-gaap", "ProfitLoss")],
    "cfo": [("us-gaap", "NetCashProvidedByUsedInOperatingActivities"),
            ("us-gaap", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations")],
    "revenue": [("us-gaap", "Revenues"),
                ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
                ("us-gaap", "SalesRevenueNet"),
                ("us-gaap", "SalesRevenueGoodsNet"),
                ("us-gaap", "RevenueFromContractWithCustomerIncludingAssessedTax")],
    "cogs": [("us-gaap", "CostOfRevenue"),
             ("us-gaap", "CostOfGoodsAndServicesSold"),
             ("us-gaap", "CostOfGoodsSold")],
    "_gross_profit": [("us-gaap", "GrossProfit")],   # helper for cogs fallback
    # EQ_OFFER input: cash raised by issuing equity during the year. Ordered
    # from the most specific common-stock tag outward; option exercises and
    # employee-plan proceeds are included because they are equity issuance in
    # Piotroski's sense (the firm took in external equity capital).
    "equity_issued": [("us-gaap", "ProceedsFromIssuanceOfCommonStock"),
                      ("us-gaap", "ProceedsFromIssuanceOfCommonStockNetOfIssuanceCosts"),
                      ("us-gaap", "ProceedsFromIssuanceOrSaleOfEquity"),
                      ("us-gaap", "ProceedsFromStockOptionsExercised"),
                      ("us-gaap", "ProceedsFromIssuanceOfSharesUnderIncentiveAndShareBasedCompensationPlans")],
}
INSTANT_TAGS = {
    "total_assets": [("us-gaap", "Assets")],
    "current_assets": [("us-gaap", "AssetsCurrent")],
    "current_liabilities": [("us-gaap", "LiabilitiesCurrent")],
    "book_value": [("us-gaap", "StockholdersEquity"),
                   ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")],
    "long_term_debt": [("us-gaap", "LongTermDebtNoncurrent"),
                       ("us-gaap", "LongTermDebt")],
}
ANNUAL_FORMS = {"10-K"}


def cik_map() -> dict[str, str]:
    """Yahoo-style ticker -> zero-padded CIK for current SEC registrants."""
    js = requests.get("https://www.sec.gov/files/company_tickers.json",
                      headers=UA, timeout=60).json()
    return {v["ticker"].replace(".", "-").upper(): f"{v['cik_str']:010d}"
            for v in js.values()}


def sp500_membership(formation_dates: list[pd.Timestamp],
                     cache_path: str | Path | None = None) -> pd.DataFrame:
    """[date, ticker] rows: S&P 500 members as of each formation date."""
    raw = requests.get(MEMBERSHIP_URL, headers=UA, timeout=60).text
    import io
    hist = pd.read_csv(io.StringIO(raw), parse_dates=["date"]).sort_values("date")
    rows = []
    for fd in formation_dates:
        snap = hist[hist.date <= fd]
        if snap.empty:
            continue
        tickers = str(snap.iloc[-1].tickers).split(",")
        rows += [{"date": fd, "ticker": t.strip().replace(".", "-").upper()}
                 for t in tickers if t.strip()]
    out = pd.DataFrame(rows)
    if cache_path:
        out.to_csv(cache_path, index=False)
    return out


# ------------------------- companyfacts parsing -------------------------

def _entries(concept: dict, unit: str, forms=ANNUAL_FORMS):
    for e in concept.get("units", {}).get(unit, []):
        if e.get("form") in forms and e.get("val") is not None:
            yield e


def _first_reported(facts: dict, tag_list, flow: bool) -> dict:
    """fiscal-period end -> (value, filed): across all fallback tags, take the
    value from the **earliest** filing that reported the period (true
    point-in-time first print; ties broken by tag priority)."""
    out: dict[pd.Timestamp, tuple[float, pd.Timestamp, int]] = {}
    for prio, (tax, tag) in enumerate(tag_list):
        concept = facts.get(tax, {}).get(tag)
        if not concept or "USD" not in concept.get("units", {}):
            continue
        for e in _entries(concept, "USD"):
            end = pd.Timestamp(e["end"])
            if flow:
                if "start" not in e:
                    continue
                days = (end - pd.Timestamp(e["start"])).days
                if not 330 <= days <= 400:      # annual durations only
                    continue
            filed = pd.Timestamp(e["filed"])
            cur = out.get(end)
            if cur is None or (filed, prio) < (cur[1], cur[2]):
                out[end] = (float(e["val"]), filed, prio)
    return {end: (v, filed) for end, (v, filed, _) in out.items()}


def _share_count(facts: dict, end: pd.Timestamp):
    """Best point-in-time share count for fiscal end `end`.

    Priority: us-gaap instant at the FYE -> dei cover-page count from the 10-K
    or a nearby 10-Q -> weighted-average basic shares.  Multi-class companies
    report one fact per class with the same (end, filed); those are summed.
    """
    def grouped(concept, unit, forms, lo, hi, flow=False):
        buckets: dict[tuple, float] = {}
        for e in _entries(concept or {}, unit, forms):
            d = pd.Timestamp(e["end"])
            if flow and "start" in e:
                if not 330 <= (d - pd.Timestamp(e["start"])).days <= 400:
                    continue
            if lo <= d <= hi:
                buckets[(pd.Timestamp(e["filed"]), d)] = \
                    buckets.get((pd.Timestamp(e["filed"]), d), 0.0) + float(e["val"])
        if not buckets:
            return None
        key = min(buckets)                       # earliest filed
        return buckets[key], key[0]

    gaap, dei = facts.get("us-gaap", {}), facts.get("dei", {})
    day = pd.Timedelta(days=1)
    for concept, unit, forms, lo, hi, flow in [
        (gaap.get("CommonStockSharesOutstanding"), "shares", ANNUAL_FORMS,
         end - 15 * day, end + 15 * day, False),
        (dei.get("EntityCommonStockSharesOutstanding"), "shares",
         {"10-K", "10-K/A", "10-Q"}, end - 30 * day, end + 240 * day, False),
        (gaap.get("WeightedAverageNumberOfSharesOutstandingBasic"), "shares",
         ANNUAL_FORMS, end, end, True),
    ]:
        got = grouped(concept, unit, forms, lo, hi, flow)
        if got:
            return got
    return None


CORE_FIELDS = ("net_income", "cfo", "revenue", "book_value")  # define report_date


def parse_companyfacts(js: dict, ticker: str) -> pd.DataFrame:
    """One company's facts -> canonical annual rows (no market_cap yet)."""
    facts = js.get("facts", {})
    fields = {f: _first_reported(facts, tags, flow=True)
              for f, tags in FLOW_TAGS.items()}
    fields |= {f: _first_reported(facts, tags, flow=False)
               for f, tags in INSTANT_TAGS.items()}

    rows = []
    for end, (assets, filed_a) in sorted(fields["total_assets"].items()):
        row = {"ticker": ticker, "total_assets": assets}
        filed = {"total_assets": filed_a}
        for f in ("net_income", "cfo", "revenue", "cogs", "_gross_profit",
                  "equity_issued", "current_assets", "current_liabilities",
                  "book_value", "long_term_debt"):
            v = fields[f].get(end)
            row[f] = v[0] if v else np.nan
            if v:
                filed[f] = v[1]
        # report_date = original 10-K: latest first-print among core fields
        report = max([filed_a] + [filed[f] for f in CORE_FIELDS if f in filed])
        # PIT hygiene: a value first printed in a *later* filing (period only
        # tagged retroactively) was not public at report time -> drop it
        for f, fdt in list(filed.items()):
            if fdt > report + pd.Timedelta(days=45):
                row[f] = np.nan
        if np.isnan(row["cogs"]) and not np.isnan(row["_gross_profit"]) \
                and not np.isnan(row["revenue"]):
            row["cogs"] = row["revenue"] - row["_gross_profit"]
        del row["_gross_profit"]
        if np.isnan(row["long_term_debt"]):
            row["long_term_debt"] = 0.0          # no LT debt tagged
        sh = _share_count(facts, end)
        row["shares_outstanding"] = sh[0] if sh else np.nan
        row["report_date"] = report
        row["fiscal_year"] = _fiscal_year(end)
        row["period_end"] = end
        rows.append(row)
    return pd.DataFrame(rows)


def fetch_fundamentals_edgar(tickers: list[str], pause: float = 0.12,
                             progress_every: int = 50) -> pd.DataFrame:
    """companyfacts for each ticker with a known CIK -> canonical frame."""
    cmap = cik_map()
    frames = []
    missing = 0
    for i, tk in enumerate(tickers):
        if progress_every and i % progress_every == 0:
            print(f"  edgar {i}/{len(tickers)}", flush=True)
        cik = cmap.get(tk.upper())
        if cik is None:
            missing += 1
            continue
        try:
            r = requests.get(
                f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
                headers=UA, timeout=60)
            if r.status_code != 200:
                continue
            df = parse_companyfacts(r.json(), tk)
            if not df.empty:
                frames.append(df)
        except Exception as e:
            print(f"  ! {tk}: {e}", flush=True)
        time.sleep(pause)
    print(f"  edgar done; no CIK for {missing} tickers", flush=True)
    out = pd.concat(frames, ignore_index=True)
    return (out.sort_values(["ticker", "fiscal_year", "report_date"])
               .drop_duplicates(["ticker", "fiscal_year"], keep="first")
               .reset_index(drop=True))


# ----------------------------- one-shot build -----------------------------

def fetch_and_cache_us_long(cache_dir: str | Path = "data",
                            years: range = range(2011, 2026)) -> None:
    """Build the long-history US cache: membership, EDGAR fundamentals,
    Yahoo prices from 2008, sectors (existing file augmented via yfinance)."""
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)

    fdates = [pd.Timestamp(f"{y}-07-01") for y in years]
    members = sp500_membership(fdates, cache_path=d / "us_membership.csv")
    union = sorted(members.ticker.unique())
    print(f"membership: {len(union)} distinct tickers over {len(fdates)} formations",
          flush=True)

    # checkpointed: EDGAR (~1h) and prices are reused on a re-run
    raw_path = d / "us_fundamentals_raw.csv"
    if raw_path.exists():
        fund = pd.read_csv(raw_path, parse_dates=["report_date", "period_end"])
        print(f"edgar fundamentals: reusing checkpoint {fund.shape}", flush=True)
    else:
        fund = fetch_fundamentals_edgar(union)
        fund.to_csv(raw_path, index=False)
        print(f"edgar fundamentals: {fund.shape}, {fund.ticker.nunique()} tickers, "
              f"FY{fund.fiscal_year.min()}-FY{fund.fiscal_year.max()}", flush=True)

    px_path = d / "us_prices.csv.gz"
    prices = None
    if px_path.exists():
        prices = pd.read_csv(px_path, parse_dates=["date"])
        if (prices.date.min() > pd.Timestamp("2008-06-30")
                or "close_raw" not in prices.columns):
            prices = None                       # stale or missing raw closes
    if prices is None:
        prices = fetch_prices(union, start="2008-01-01")
        prices.to_csv(px_path, index=False, compression="gzip")
    print(f"prices: {prices.shape}, {prices.ticker.nunique()} tickers", flush=True)

    # market cap at fiscal period end (price asof period_end x cover shares)
    fund = fund.rename(columns={"report_date": "_filed"})
    fund = fund.rename(columns={"period_end": "report_date"})
    fund = attach_market_cap(fund, prices)
    fund = fund.rename(columns={"report_date": "period_end",
                                "_filed": "report_date"})
    fund.to_csv(d / "us_fundamentals.csv", index=False)
    print(f"fundamentals cached: {fund.shape}", flush=True)

    # sectors: keep the current-constituent file, best-effort fill for the rest
    sec_path = d / "us_sectors.csv"
    sectors = pd.read_csv(sec_path) if sec_path.exists() else pd.DataFrame(
        columns=["ticker", "name", "sector"])
    have = set(sectors.ticker)
    todo = [t for t in fund.ticker.unique() if t not in have]
    print(f"sector fill: {len(todo)} names via yfinance", flush=True)
    import yfinance as yf
    extra = []
    for tk in todo:
        try:
            s = yf.Ticker(tk).info.get("sector")
            if s:
                extra.append({"ticker": tk, "name": tk, "sector": s})
        except Exception:
            pass
        time.sleep(0.1)
    pd.concat([sectors, pd.DataFrame(extra)], ignore_index=True) \
        .drop_duplicates("ticker").to_csv(sec_path, index=False)

    # benchmarks over the long window
    from .yahoo import BENCHMARKS
    bench = fetch_prices(BENCHMARKS["us"], start="2008-01-01")
    bench.to_csv(d / "us_benchmarks.csv.gz", index=False, compression="gzip")
    print("us long-history cache complete.", flush=True)


def load_membership(cache_dir: str | Path = "data") -> dict[int, set[str]]:
    m = pd.read_csv(Path(cache_dir) / "us_membership.csv", parse_dates=["date"])
    return {d.year: set(g.ticker) for d, g in m.groupby("date")}
