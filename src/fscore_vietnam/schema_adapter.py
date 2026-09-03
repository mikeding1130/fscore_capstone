"""Vietnam adapter: consume the preprocessing pipeline. Never re-derive it.

The Vietnamese data does not come from a vendor API. It comes from
`src/fscore_vietnam`, which crawls FireAnt/CafeF/TCBS into `fscore.db`,
reconciles the three sources, applies accounting checks, scores the nine
Piotroski signals, and writes a per-firm-year panel. **That pipeline is the
source of record for every Vietnamese number in this study.** It ships the
two panels both studies rank on — `vietnam_prices.csv.gz` and the score
panel `vietnam_scores.csv` — through its `run_grid_export.ipynb`.

Every location it reads is resolved by `fscore_vietnam.paths`, the one module
in this package that knows where the pipeline's tree sits.

This module therefore does exactly three things, none of which recompute
anything the pipeline already decided:

  * `build_fundamentals` — book equity and fiscal-year-end market value per
    firm-year, read from the pipeline's `final_panel.csv`, in the reduced
    `loaders.SCORED_MARKET_FUND_COLS` schema. This is what the high-B/M
    subset sorts on and nothing more.
  * `build_sectors` — ticker -> sector, the same map the export used;
  * `fetch_benchmarks` — the VN30 and VNINDEX levels out of `fscore.db`.

An earlier version rebuilt the eleven raw statement lines here and re-scored
them with `signal.piotroski`, because `pipeline.run_study` used to insist on
scoring every market itself. That re-derivation is gone. It reproduced the
pipeline's panel exactly — all 16,564 jointly scored firm-years, all nine flags
— so it was never adding a number; what it did add was a second copy of the
Vietnamese scores that could drift from the first, and it drifted twice: it
read the extract BEFORE the accounting checks (2,260 firm-years the pipeline
rejects), and the main study handed it only two fiscal years, which silenced
the beginning-of-year asset scaling in ΔROA and Δturnover. `run_study` now
takes `scores=` instead, so both studies rank the same panel.

Two conventions, stated rather than implied:

  * `report_date` is the fiscal year end, 31 December. Vietnamese statutory
    deadlines are 90 days for the audited annual report (100 with the usual
    consolidation extension), but the panel carries no filing date, so the
    lag is applied at formation by `lag_months` exactly as for Japan. The
    Vietnam notebooks pass **6** — 31 December + 6 months = 30 June, the last
    day before a 1 July formation. That is the most conservative rule that
    still admits the prior fiscal year, and it is the same screening date the
    pipeline itself screens on, so both halves select on the same information.
  * `book_value` / `market_cap` are the pipeline's `book_equity` and
    `market_equity`, so `book_value / market_cap` here IS the `bm` that panel
    already carries — the same share count and the same December price it
    resolved, not a second opinion about either.

**The benchmark is a price index.** VN30 and VNINDEX are capital indices —
they exclude cash dividends. The portfolios are built on FireAnt's
`adj_ratio`-adjusted closes, which do adjust for cash dividends, so the
portfolios are total-return and the benchmark is not. The gap is the index
dividend yield, historically ~1.5-2% a year for VN30, and it flatters every
portfolio-vs-benchmark comparison by that much. No total-return version of
either index is in `fscore.db`, so this is disclosed rather than corrected.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

# `paths` is the one module that knows where the pipeline's inputs and
# outputs sit. Importing from it rather than restating the layout is what
# keeps this adapter from drifting away from the notebooks beside it: every
# path below would otherwise be a second opinion about where the panels are.
from . import paths as vn_paths

PIPELINE_DIR = vn_paths.RESULTS
DB_PATH = vn_paths.DB           # ~1.7 GB, not in the repo; see paths.py

SECTOR_MAP_CSV = vn_paths.DATA / "tickers_non_financials_sectors.csv"
SECTOR_FIELD = "Sector_EN"
SECTOR_UNKNOWN = "Unknown"

BENCHMARKS = ["VN30", "VNINDEX"]
FY_END = (12, 31)          # Vietnamese fiscal years end 31 December


def build_fundamentals(pipeline_dir: str | Path = PIPELINE_DIR) -> pd.DataFrame:
    """Book equity and fiscal-year-end market value per firm-year.

    Read straight from the pipeline's `final_panel.csv` — the file the
    pipeline treats as its own output of record, already restricted to the
    firm-years that passed its accounting checks. `book_equity` and
    `market_equity` are where it resolved the share-count question (issued vs
    outstanding vs par-implied) and priced the December close.

    No statement lines and no signals: this study reads Vietnamese scores, it
    does not compute them (see the module docstring).
    """
    d = Path(pipeline_dir)
    fp = pd.read_csv(d / "final_panel.csv",
                     usecols=["symbol", "period", "book_equity", "market_equity"])
    out = pd.DataFrame({
        "ticker": fp["symbol"],
        "fiscal_year": fp["period"].astype(int),
        "book_value": fp["book_equity"],
        "market_cap": fp["market_equity"],
    })
    month, day = FY_END
    out["report_date"] = pd.to_datetime(
        {"year": out.fiscal_year, "month": month, "day": day})
    out = (out.sort_values(["ticker", "fiscal_year"])
              .drop_duplicates(["ticker", "fiscal_year"], keep="last")
              .reset_index(drop=True))
    # a non-positive book value is not a value stock, it is a broken B/M
    # sort: B/M would come out negative and rank below every positive one
    # rather than above them. Dropped here, counted for the write-up.
    bad_equity = int((out.book_value <= 0).sum())
    out.loc[out.book_value <= 0, "book_value"] = np.nan
    out.attrs["nonpositive_book_value"] = bad_equity
    out.attrs["rows"] = len(out)
    return out


def build_sectors(sector_csv: str | Path = SECTOR_MAP_CSV) -> pd.Series:
    """ticker -> sector, from the pipeline's crawled classification.

    The label is NOT point-in-time: it is the classification as crawled,
    applied to every year of a ticker's history. It only enters the
    sector-capped GMV constraint, so it is disclosed rather than corrected —
    there is no historical sector series to correct it against.
    """
    s = pd.read_csv(sector_csv, usecols=["Symbol", SECTOR_FIELD])
    return (s.rename(columns={"Symbol": "ticker", SECTOR_FIELD: "sector"})
             .dropna(subset=["ticker"])
             .drop_duplicates("ticker", keep="last")
             .set_index("ticker")["sector"].fillna(SECTOR_UNKNOWN))


def fetch_benchmarks(db_path: str | Path = DB_PATH,
                     symbols: list[str] | None = None) -> pd.DataFrame:
    """Index levels from `fscore.db`, in the canonical benchmark schema.

    Indices are quoted at `unit = 1` with `adj_ratio = 1`, so `adj_close` is
    the published level. They are CAPITAL indices — see the module docstring.
    """
    symbols = list(symbols or BENCHMARKS)
    placeholders = ",".join("?" * len(symbols))
    sql = f"""
    select symbol as ticker, date as date,
           price_close * unit / adj_ratio as adj_close,
           total_volume as volume
    from fireant_prices
    where symbol in ({placeholders}) and price_close > 0 and adj_ratio > 0
    order by symbol, date
    """
    with closing(sqlite3.connect(str(db_path))) as conn:
        px = pd.read_sql(sql, conn, params=symbols, parse_dates=["date"])
    missing = sorted(set(symbols) - set(px.ticker))
    if missing:
        raise ValueError(f"no rows in fireant_prices for {missing}")
    return px


EXCLUSIONS_FILENAME = "vietnam_exclusions.csv"


def build_cache(out_dir: str | Path = "data",
                pipeline_dir: str | Path = PIPELINE_DIR,
                db_path: str | Path = DB_PATH,
                sector_csv: str | Path = SECTOR_MAP_CSV,
                exclusions_filename: str = EXCLUSIONS_FILENAME) -> dict[str, Path]:
    """Write the three files `yahoo.load_cached('vietnam')` still needs.

    The fourth, `vietnam_prices.csv.gz`, is already written by the
    pipeline's export notebook and is left untouched — rebuilding it here
    would fork the price convention the score panel was built against.

    `exclusions_filename` has to match the name `fs_clean.exclusion_report`
    looks for, or the grid notebook's accounting cell finds nothing. It is a
    parameter rather than an import so that this package depends on nothing
    in `fscore`; `schema_adapter_util` passes the authoritative value and
    asserts on it, which is the one place already coupled to both.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fund = build_fundamentals(pipeline_dir)
    fund_path = out / "vietnam_fundamentals.csv"
    fund.to_csv(fund_path, index=False)

    sectors = build_sectors(sector_csv)
    sec_path = out / "vietnam_sectors.csv"
    sectors.rename("sector").rename_axis("ticker").reset_index().to_csv(
        sec_path, index=False)

    bench = fetch_benchmarks(db_path)
    bench_path = out / "vietnam_benchmarks.csv.gz"
    bench.to_csv(bench_path, index=False, compression="gzip")

    # written under the name `fs_clean.exclusion_report` reads, so the grid
    # notebook's accounting cell is the same code in all three markets
    drops = build_exclusions(pipeline_dir)
    drop_path = out / exclusions_filename
    drops.to_csv(drop_path, index=False)

    return {"fundamentals": fund_path, "sectors": sec_path,
            "benchmarks": bench_path, "exclusions": drop_path}


# ----------------------------------------------------------------------
# What the source holds, and what the study uses
# ----------------------------------------------------------------------

# Order the reasons are stacked in, coarsest first. Ordered explicitly so the
# exclusion chart reads the same way every time it is regenerated.
DROP_ORDER = [
    "dropped_failed_accounting_checks",
    "dropped_no_prior_year",
    "dropped_incomplete_signals",
    "dropped_unresolved_identifier",
    "dropped_no_book_to_market",
    "dropped_no_june_turnover",
    "dropped_no_formation_price",
    "dropped_below_liquidity_gate",
]


def build_exclusions(pipeline_dir: str | Path = PIPELINE_DIR) -> pd.DataFrame:
    """Account for every firm-year the Vietnamese source holds but the study
    does not rank, by reason, in the schema `fs_clean.exclusion_report` reads.

    The three reasons the US and Japan panels report all have exact Vietnamese
    analogues, and two of them mean the same thing:

      * `dropped_no_prior_year` — no t-1 row to difference against. In this
        panel that is the firm-year where only one of the nine signals is
        computable at all, which is what a missing prior year leaves.
      * `dropped_incomplete_signals` — a prior year exists but the nine
        signals are still not all computable (most often no t-2 row, which
        ΔROA and Δturnover need).
      * `dropped_unresolved_identifier` — **always zero here.** Vietnamese
        identifiers in this panel are exchange tickers, not vendor internal
        IDs, so nothing has to be resolved and nothing fails to. It is
        reported rather than omitted so the three markets' tables line up.

    Vietnam then applies four gates the other two markets do not, and they
    are counted separately rather than folded into the three above, because
    they remove far more than the three do:

      * accounting checks (the pipeline's reconciliation of three
        vendors) reject a firm-year outright;
      * no book-to-market, no June turnover, or no formation price;
      * `tradeable` false — the liquidity screen, which is the single largest
        exclusion in this market by a wide margin. It is a deliberate design
        choice on the Vietnamese branch, not a data limitation, and the
        difference matters when the three markets' universes are compared.
    """
    d = Path(pipeline_dir)
    fx = pd.read_csv(d / "f_score_fields_extract_corrected.csv",
                     usecols=["symbol", "period"])
    fp = pd.read_csv(d / "final_panel.csv",
                     usecols=["symbol", "period", "f_score", "bm", "turnover",
                              "close_raw", "tradeable", "signals_available"])

    src = fx.groupby("period").size()
    in_panel = fp.groupby("period").size()

    scored = fp.f_score.notna()
    # only one signal computable = no prior-year row to difference against
    no_prior = fp.signals_available.eq(1)
    incomplete = ~scored & ~no_prior
    has_bm = scored & fp.bm.notna()
    has_to = has_bm & fp.turnover.notna()
    has_px = has_to & fp.close_raw.notna()
    tradeable = has_px & fp.tradeable.eq(True)

    def by_year(mask) -> pd.Series:
        return fp[mask].groupby("period").size()

    out = pd.DataFrame({
        "score_year": src.index,
        "rows_in_source": src.values,
    }).set_index("score_year")
    out["dropped_failed_accounting_checks"] = (src - in_panel).fillna(src)
    out["dropped_no_prior_year"] = by_year(no_prior)
    out["dropped_incomplete_signals"] = by_year(incomplete)
    out["dropped_unresolved_identifier"] = 0
    out["dropped_no_book_to_market"] = by_year(scored) - by_year(has_bm)
    out["dropped_no_june_turnover"] = by_year(has_bm) - by_year(has_to)
    out["dropped_no_formation_price"] = by_year(has_to) - by_year(has_px)
    out["dropped_below_liquidity_gate"] = by_year(has_px) - by_year(tradeable)
    out["scored"] = by_year(tradeable)
    out = out.fillna(0).astype(int).reset_index()

    accounted = out[DROP_ORDER + ["scored"]].sum(axis=1)
    assert (accounted == out.rows_in_source).all(), \
        "the exclusion reasons do not add up to the rows in the source"
    return out
