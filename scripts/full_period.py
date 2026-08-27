"""Within-country robustness: re-run each market over its own full data span.

The headline results run on the common 2012-2024 formation window so the
markets are compared on the same calendar. That window is a compromise in
both directions: it discards US formations the data would support, and it is
unreachable for Japan, whose fundamentals begin far later. A common window
that no market's data actually chose is worth checking against each market's
own maximum span, which is what this produces - not a competing headline, but
a statement of how much the reported numbers depend on the window.

The span is measured, not assumed. A formation year is feasible when the
point-in-time snapshot yields a high-B/M subset at least as large as the
basket, and when its full twelve-month holding year finishes inside the price
sample. Everything else - lag, membership, seed, covariance window, Monte
Carlo draws - is held identical to the headline run, so the window is the
only thing that differs.

Run:  python scripts/full_period.py            # every market with a cache
      python scripts/full_period.py us         # one market

Writes, per market:
    results/{market}_fullperiod_summary.csv
    results/{market}_fullperiod_placement.csv
    results/{market}_fullperiod_diagnostics.csv
    results/figures/{market}_fullperiod_nav.png
and one combined comparison:
    results/robustness_full_period.csv
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fscore.data.loaders import high_bm_subset            # noqa: E402
from fscore.data.yahoo import load_cached                 # noqa: E402
from fscore.pipeline import (build_universe, formation_date,  # noqa: E402
                             pit_snapshot, run_study)
from fscore.plotting import save_fig, setup_plots         # noqa: E402
from fscore.selection.baskets import BASKET_SIZE          # noqa: E402

RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"

# Held identical to the headline notebooks; only the year range differs.
COMMON_YEARS = list(range(2012, 2025))
RUN_KW = dict(n_mc=1000, n_mc_opt=300, seed=42, detone=False, end_cap=None)

# The proposal evaluates through 2025, so a holding year may not run past June
# 2025 — the July-2024 formation is the last one, in every market and in this
# robustness check too. The price cache now reaches August 2026 and would
# support a July-2025 formation, but taking it would move the goalposts: the
# window would no longer be the one the study was specified on. "Full
# available period" therefore means each market's own earliest feasible
# formation forward to the common end, not forward into 2026.
EVAL_END = pd.Timestamp("2025-06-30")

MARKETS = {
    "us":    dict(lag_months=1, membership=True,
                  bench={"SPY (S&P 500)": "SPY", "VTV (US value ETF)": "VTV"}),
    "japan": dict(lag_months=3, membership=False,
                  bench={"1306.T (TOPIX ETF, JPY)": "1306.T"}),
    # Vietnam's cache is built by src/fscore_vietnam/schema_adapter_util.py. Its lag is 6
    # months rather than 3: report_date is the 31 December fiscal year end and
    # +6 months lands on 30 June, the last day before formation.
    "vietnam": dict(lag_months=6, membership=False,
                    bench={"VN30 (VN30 index, VND)": "VN30"}),
}
PROBE_FROM, PROBE_TO = 2005, 2027


def feasible_years(market: str, fund, prices, membership, lag_months: int,
                   k: int = BASKET_SIZE) -> tuple[list[int], pd.DataFrame]:
    """The formation years this market's data can actually support, within the
    study's fixed evaluation end (see EVAL_END)."""
    price_end = min(prices.date.max(), EVAL_END)
    rows = []
    for y in range(PROBE_FROM, PROBE_TO):
        snap = pit_snapshot(fund, y, lag_months=lag_months)
        if snap.empty:
            rows.append({"year": y, "value_names": 0, "usable": False,
                         "reason": "no point-in-time statements"})
            continue
        uni = build_universe(prices, snap, y, n=150,
                             members=membership.get(y) if membership else None)
        vs = high_bm_subset(uni, quantile=0.4) if len(uni) else uni
        end = formation_date(y) + pd.DateOffset(years=1) - pd.Timedelta(days=1)
        if len(vs) < k:
            reason = f"high-B/M subset {len(vs)} < basket {k}"
        elif end > price_end:
            reason = (f"holding year ends {end:%Y-%m-%d}, past the "
                      f"{price_end:%Y-%m-%d} evaluation end")
        else:
            reason = ""
        rows.append({"year": y, "universe": len(uni), "value_names": len(vs),
                     "usable": not reason, "reason": reason})
    tbl = pd.DataFrame(rows)
    return tbl[tbl.usable].year.tolist(), tbl


PLACEMENT_STRATEGIES = [("fscore_EW", "EW"), ("fscore_GMV", "GMV"),
                        ("fscore_GMVsec", "GMVsec")]


def headline_row(market: str) -> pd.DataFrame | None:
    """The already-published common-window numbers, for the comparison."""
    p = RESULTS / f"{market}_summary.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, index_col=0)


def placement_table(study) -> pd.DataFrame:
    """Strategy x metric, laid out exactly like the headline placement CSV so
    the two can be read side by side."""
    frames = []
    for strat, how in PLACEMENT_STRATEGIES:
        if strat not in study.daily.columns or how not in study.mc_daily:
            continue
        pl = study.placement(strategy=strat, construction=how)
        pl.index = pd.MultiIndex.from_product([[strat], pl.index])
        frames.append(pl)
    return pd.concat(frames) if frames else pd.DataFrame()


def headline_placement(market: str) -> pd.DataFrame | None:
    """Published placement CSV: two unnamed index columns (strategy, metric)."""
    p = RESULTS / f"{market}_mc_placement.csv"
    if not p.exists():
        return None
    h = pd.read_csv(p)
    cols = list(h.columns)
    return h.rename(columns={cols[0]: "strategy", cols[1]: "metric"})


def run_market(market: str, cfg: dict) -> dict | None:
    fund, prices, sectors, bench = load_cached(market, ROOT / "data")
    membership = None
    if cfg["membership"]:
        from fscore.data.edgar import load_membership
        membership = load_membership(ROOT / "data")

    years, probe = feasible_years(market, fund, prices, membership,
                                  cfg["lag_months"])
    probe.to_csv(RESULTS / f"{market}_fullperiod_feasibility.csv", index=False)
    if not years:
        print(f"  {market}: no feasible formation year - skipped")
        return None

    span_common = [y for y in COMMON_YEARS if y in years]
    print(f"  {market}: full span {years[0]}-{years[-1]} "
          f"({len(years)} formations); headline window covers "
          f"{len(span_common)} of them")
    if years == span_common:
        print(f"  {market}: full span already equals the headline window - "
              f"the robustness run is the same study, reported for the record")

    study = run_study(market, fund, prices, sectors, years,
                      lag_months=cfg["lag_months"], membership=membership,
                      **RUN_KW)

    summ = study.summary()
    summ.to_csv(RESULTS / f"{market}_fullperiod_summary.csv")
    place = placement_table(study)
    place.to_csv(RESULTS / f"{market}_fullperiod_placement.csv")
    diag = pd.DataFrame([{"year": yr.year, **yr.diagnostics}
                         for yr in study.yearly]).set_index("year")
    diag.to_csv(RESULTS / f"{market}_fullperiod_diagnostics.csv")

    # NAV over the full span, with the headline window shaded so the extra
    # years are visually separable from the ones already reported.
    start, end = study.daily.index.min(), study.daily.index.max()
    nav = (1 + study.daily.fillna(0)).cumprod()
    fig, ax = plt.subplots(figsize=(10, 5))
    for s in study.daily.columns:
        ls = "-" if s.startswith("fscore") else "--"
        lw = 2.2 if s == "fscore_EW" else 1.2
        ax.plot(nav.index, nav[s], ls, lw=lw, label=s)
    if span_common:
        ax.axvspan(formation_date(span_common[0]),
                   formation_date(span_common[-1]) + pd.DateOffset(years=1),
                   color="0.85", zorder=0, label="headline window")
    ax.set_yscale("log"); ax.set_ylabel("growth of 1 (log)")
    ax.set_title(f"{market.upper()} - full available period, "
                 f"{start:%b %Y} - {end:%b %Y} "
                 f"({len(years)} formations)")
    ax.legend(fontsize=8, ncol=2); plt.tight_layout()
    save_fig(f"{market}_fullperiod_nav", directory=FIGS)
    plt.close(fig)

    # Side-by-side rows: one per strategy, headline window against full span.
    head, hplace = headline_row(market), headline_placement(market)
    rows = []
    for strat, _ in PLACEMENT_STRATEGIES:
        if strat not in summ.index:
            continue
        r = {"market": market, "strategy": strat,
             "full_first_formation": years[0], "full_last_formation": years[-1],
             "full_formations": len(years),
             "full_span": f"{start:%Y-%m} to {end:%Y-%m}",
             "headline_formations": len(span_common),
             "extra_formations": len(years) - len(span_common),
             "full_sharpe": round(float(summ.loc[strat, "sharpe"]), 3),
             "full_ann_return": round(float(summ.loc[strat, "ann_return"]), 4)}
        if head is not None and strat in head.index:
            r["headline_sharpe"] = round(float(head.loc[strat, "sharpe"]), 3)
            r["headline_ann_return"] = round(float(head.loc[strat, "ann_return"]), 4)
            r["sharpe_delta"] = round(r["full_sharpe"] - r["headline_sharpe"], 3)
        if len(place) and (strat, "sharpe") in place.index:
            r["full_percentile"] = round(float(place.loc[(strat, "sharpe"), "percentile"]), 3)
            r["full_p_value"] = round(float(place.loc[(strat, "sharpe"), "p_value"]), 4)
        if hplace is not None:
            m = hplace[(hplace.strategy == strat) & (hplace.metric == "sharpe")]
            if len(m):
                r["headline_percentile"] = round(float(m.percentile.iloc[0]), 3)
                r["headline_p_value"] = round(float(m.p_value.iloc[0]), 4)
        rows.append(r)
    return rows


def main() -> None:
    setup_plots()
    FIGS.mkdir(parents=True, exist_ok=True)
    wanted = sys.argv[1:] or list(MARKETS)
    rows = []
    for m in wanted:
        if m not in MARKETS:
            print(f"  unknown market {m!r} - skipped")
            continue
        print(f"\n=== {m} ===")
        try:
            r = run_market(m, MARKETS[m])
        except FileNotFoundError as exc:
            print(f"  {m}: no cache ({exc}) - skipped")
            continue
        if r:
            rows.extend(r)
    if rows:
        cmp = pd.DataFrame(rows).set_index(["market", "strategy"])
        # Running one market must not delete the others. The combined file is
        # the union of what is on disk and what this run produced, with this
        # run winning where they overlap — otherwise
        # `full_period.py vietnam` would silently discard the US and Japan
        # rows, which is exactly what it used to do.
        out = RESULTS / "robustness_full_period.csv"
        if out.exists():
            prev = pd.read_csv(out).set_index(["market", "strategy"])
            keep = prev[~prev.index.isin(cmp.index)]
            cmp = (pd.concat([keep, cmp])
                     .reindex(columns=cmp.columns.union(prev.columns,
                                                        sort=False))
                     .sort_index())
        cmp.to_csv(out)
        print("\n" + cmp.to_string())
        print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
