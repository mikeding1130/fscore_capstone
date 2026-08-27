"""Run `schema_adapter` once: pipeline panels -> the study's data cache.

The US and Japan caches are written by `scripts/fetch_us_edgar.py` /
`fetch_us_japan.py` off a vendor API. Vietnam has no such API: its statements
come from the notebooks beside this file, which crawl three Vietnamese
sources into `fscore.db`, reconcile them, apply accounting checks, score the
nine signals, and write a checked per-firm-year panel. That pipeline is the
source of record for every Vietnamese number in the study — nothing below
recomputes a signal.

Two of the six files the study reads come straight out of the pipeline's
`run_grid_export.ipynb` (`vietnam_scores.csv`, `vietnam_prices.csv.gz`) and
are never touched here. This script writes the other four:

    data/vietnam_fundamentals.csv     book equity + FY-end market value (B/M only)
    data/vietnam_sectors.csv          ticker -> sector
    data/vietnam_benchmarks.csv.gz    VN30 and VNINDEX levels
    data/vietnam_exclusions.csv       what the source holds vs what is scored

Locations come from `paths.py`; override them there (or with its environment
variables) rather than by passing paths in here.

Run:  python src/fscore_vietnam/schema_adapter_util.py
      python src/fscore_vietnam/schema_adapter_util.py --pipeline-results DIR
"""
from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]        # src/fscore_vietnam -> src -> repo
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd                                        # noqa: E402

from fscore_vietnam import schema_adapter as sa            # noqa: E402
# The only two things this package borrows from `fscore`, and both are
# contracts rather than logic: the column set a scored market's fundamentals
# must present, and the filename its exclusion report is looked up under.
# They live here, in the runner, so `schema_adapter` itself imports nothing
# from `fscore`.
from fscore.data.fs_clean import EXCLUSIONS                # noqa: E402
from fscore.data.loaders import SCORED_MARKET_FUND_COLS    # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pipeline-results", default=str(sa.PIPELINE_DIR),
                    help="where the pipeline wrote its panels")
    ap.add_argument("--db", default=str(sa.DB_PATH), help="path to fscore.db")
    ap.add_argument("--sectors", default=str(sa.SECTOR_MAP_CSV))
    ap.add_argument("--out", default=str(ROOT / "data"))
    args = ap.parse_args()

    # the adapter defaults to this name; passing it explicitly is what ties
    # the file it writes to the file `fs_clean.exclusion_report` reads
    exclusions_name = EXCLUSIONS.format(market="vietnam")
    assert exclusions_name == sa.EXCLUSIONS_FILENAME, (
        f"exclusion filename drifted: fscore expects {exclusions_name}, "
        f"schema_adapter defaults to {sa.EXCLUSIONS_FILENAME}")

    paths = sa.build_cache(out_dir=args.out,
                           pipeline_dir=args.pipeline_results,
                           db_path=args.db,
                           sector_csv=args.sectors,
                           exclusions_filename=exclusions_name)

    fund = pd.read_csv(paths["fundamentals"], parse_dates=["report_date"])
    missing = [c for c in SCORED_MARKET_FUND_COLS if c not in fund.columns]
    assert not missing, f"fundamentals missing canonical columns: {missing}"
    assert not fund.duplicated(["ticker", "fiscal_year"]).any(), \
        "fundamentals are not unique on (ticker, fiscal_year)"
    # Vietnamese signals belong to this package, not to `fscore.signal`. A
    # statement line reappearing here would mean something started
    # re-deriving them; fail loudly rather than let a second copy grow back.
    from fscore.signal.piotroski import SIGNALS                # noqa: E402
    strays = [c for c in ("total_assets", "net_income", "cfo", "revenue",
                          "cogs", "equity_issued", "long_term_debt")
              if c in fund.columns] + [c for c in SIGNALS if c in fund.columns]
    assert not strays, f"fundamentals must carry no signal inputs, found: {strays}"

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
          "portfolios are total-return. See schema_adapter.")


if __name__ == "__main__":
    main()
