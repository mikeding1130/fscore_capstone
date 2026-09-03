"""The optimisation-gain test, D = Sharpe(GMV) - Sharpe(EW), per market.

The reviewer's first priority was a *direct* test of whether portfolio
construction adds anything, rather than inferring it from two separately
reported Sharpe ratios. D answers it, and the test is paired: the random
distribution is of the same within-basket difference, because the optimised
Monte Carlo columns optimise the very baskets the equal-weight columns hold
equally, column for column. An unpaired comparison would mix the optimiser's
effect together with the luck of drawing different names.

Two experiment families report D, and they are not interchangeable:

  * **main study** — F-Score ranked inside the high-B/M subset, random baskets
    drawn from that same subset. Computed here by re-running each market.
  * **grid** — F-Score ranked across the whole scoreable universe, random
    baskets drawn from the whole universe. Read from the grid summaries, which
    already carry D and its p-value for all nine cells per market.

Vietnam's main study runs here too, with one difference: it reads a score
panel produced upstream instead of computing the nine signals, so the panel is
passed in rather than derived. It needs the Vietnamese caches that
`src/fscore_vietnam/schema_adapter_util.py` writes from the sibling `../thesis`
checkout (see VIETNAM_RERUN_NEEDED.md); where those are absent the market
falls back to its grid rows, as any market does.

Run:  python scripts/d_test.py            # every market it can compute
      python scripts/d_test.py us japan   # a subset
Writes results/d_test_by_market.csv
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fscore.pipeline import run_study  # noqa: E402

RESULTS = ROOT / "results"
YEARS = list(range(2012, 2025))
RUN_KW = dict(n_mc=1000, n_mc_opt=300, seed=42, detone=False, end_cap=None)
MAIN_STUDY = {"us": dict(lag_months=1), "japan": dict(lag_months=3),
              "vietnam": dict(lag_months=6)}


def load_market(market: str):
    """Fundamentals, prices, sectors, membership and — where the market is
    scored upstream — its score panel, from the source this market's main
    study reads. The panel is None for markets scored in this repository."""
    data = ROOT / "data"
    if market == "japan":
        from fscore.data.bbg_processed import constituents
        fund = pd.read_csv(data / "japan_bbg_fundamentals.csv",
                           parse_dates=["report_date"])
        prices = pd.read_csv(data / "japan_prices.csv.gz", parse_dates=["date"])
        sectors = pd.read_csv(data / "japan_sectors.csv").set_index("ticker")["sector"]
        return fund, prices, sectors, constituents(market, data, YEARS), None
    if market == "vietnam":
        # Read the three files directly rather than through `load_cached`,
        # which also demands the benchmark cache. Benchmarks carry no null and
        # take no part in D, so requiring them would block this test on a file
        # it never reads.
        from fscore.data.fs_clean import load_scores
        fund = pd.read_csv(data / "vietnam_fundamentals.csv",
                           parse_dates=["report_date"])
        prices = pd.read_csv(data / "vietnam_prices.csv.gz", parse_dates=["date"])
        sectors = pd.read_csv(data / "vietnam_sectors.csv").set_index("ticker")["sector"]
        # Scored upstream by `src/fscore_vietnam`: the study reads that panel
        # and does not recompute a signal. No index membership applies.
        return fund, prices, sectors, None, load_scores(market, data)
    from fscore.data.edgar import load_membership
    from fscore.data.yahoo import load_cached
    fund, prices, sectors, _ = load_cached(market, data)
    return fund, prices, sectors, load_membership(data), None


def main_study_d(market: str) -> list[dict]:
    fund, prices, sectors, membership, scores = load_market(market)
    study = run_study(market, fund, prices, sectors, YEARS,
                      membership=membership, scores=scores,
                      **MAIN_STUDY[market], **RUN_KW)
    rows = []
    for construction, r in study.synergy_table().iterrows():
        rows.append({"market": market, "family": "main study (high-B/M subset)",
                     "spec": f"k=30, {construction}",
                     "D": round(float(r.D_fscore), 4),
                     "D_random_mean": round(float(r.D_random_mean), 4),
                     "percentile": round(float(r.percentile), 4),
                     "p_value": round(float(r.p_value), 4),
                     "significant_5pct": bool(r.significant),
                     "n_draws": int(r.n)})
    return rows


def grid_d(market: str) -> list[dict]:
    p = RESULTS / "grid" / f"{market}_grid_summary.csv"
    if not p.exists():
        return []
    g = pd.read_csv(p)
    rows = []
    # D does not vary with N at a given k - the same seed draws the same first
    # 1,000 baskets whether the run asks for 1,000 or 5,000 - so one row per k.
    for k, sub in g.groupby("k"):
        r = sub.iloc[0]
        rows.append({"market": market, "family": "grid (full universe)",
                     "spec": f"k={k}, GMV",
                     "D": round(float(r.D), 4), "D_random_mean": None,
                     "percentile": None, "p_value": round(float(r.D_p), 4),
                     "significant_5pct": bool(r.D_p < 0.05), "n_draws": 300})
    return rows


def merge_into_shared(fresh: pd.DataFrame,
                      previous: pd.DataFrame | None
                      ) -> tuple[pd.DataFrame, list[str]]:
    """Fold this run's rows into the shared table without truncating it.

    A subset run must not shorten the file. Running `d_test.py us` after a full
    pass used to overwrite it with US-only rows — no error raised, just a
    quietly incomplete table, which is how the committed version lost Japan.
    Rows for the markets recomputed here are replaced; every other market's are
    carried through, and the caller is told which.
    """
    fresh = fresh.sort_values(["market", "family", "spec"])
    if previous is None or previous.empty:
        return fresh.reset_index(drop=True), []
    keep = previous[~previous.market.isin(set(fresh.market.unique()))]
    merged = (pd.concat([keep, fresh], ignore_index=True)
                .sort_values(["market", "family", "spec"])
                .reset_index(drop=True))
    return merged, sorted(keep.market.unique())


def main() -> None:
    wanted = sys.argv[1:] or ["us", "japan", "vietnam"]
    rows = []
    for m in wanted:
        rows.extend(grid_d(m))
        if m in MAIN_STUDY:
            print(f"running the {m} main study for its D test ...", flush=True)
            try:
                rows.extend(main_study_d(m))
            except FileNotFoundError as exc:
                print(f"  {m}: main study not runnable here ({exc})")
        else:
            print(f"{m}: no main study configured here - grid rows only")

    if not rows:
        print("nothing to report")
        return
    dest = RESULTS / "d_test_by_market.csv"
    prev = pd.read_csv(dest) if dest.exists() else None
    out, kept = merge_into_shared(pd.DataFrame(rows), prev)
    if kept:
        print(f"\nkeeping the rows already on file for {kept}")
    out.to_csv(dest, index=False)
    print("\n" + out.to_string(index=False))
    neg = int((out.D < 0).sum())
    sig = int(out.significant_5pct.sum())
    print(f"\nD is negative in {neg} of {len(out)} specifications; {sig} clear 5%.")

    # A significant D can still be a negative D, and the two readings are
    # opposite. Say which one applies rather than leaving "significant" to be
    # read as "the optimiser helps".
    trap = out[(out.D < 0) & out.significant_5pct]
    for _, r in trap.iterrows():
        print(f"\nNote — {r.market} {r.spec} ({r.family}): D = {r.D:+.4f} is "
              f"**negative** yet significant (p = {r.p_value}). The optimiser "
              f"lowers this basket's Sharpe; it clears 5% only because it "
              f"lowers a random basket's Sharpe by more "
              f"(random mean D = {r.D_random_mean:+.4f}). Read as 'hurts "
              f"less than it hurts chance', not as 'adds value'.")
    print(f"saved -> {RESULTS / 'd_test_by_market.csv'}")


if __name__ == "__main__":
    main()
