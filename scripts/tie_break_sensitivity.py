"""How much of a grid result is the tie-break draw rather than the F-Score?

The F-Score is an integer 0-9, so the top-k cut almost never falls cleanly.
`rank_by_fscore` breaks the tie at random, seeded per formation year, and the
grid diagnostics already report how many of the k slots that decides
(`tie_break_slots`). In Vietnam that number reaches k itself in several
formation years: every name in the "F-Score basket" that year is a coin flip
among firms sharing the same score.

When that happens, the reported basket is one draw from a large set of
equally F-Score-justified baskets, and its Sharpe carries the sampling noise
of that draw on top of everything else. A single seed cannot show this. This
script re-runs one grid cell at several seeds and reports the spread of the
headline numbers across them — the honest error bar around a result the
notebook prints as one number.

The random Monte-Carlo draws move with the seed as well, so what varies here
is the whole (basket, control) pair, exactly as it would if the study had
been specified with a different arbitrary seed on day one.

Run:  python scripts/tie_break_sensitivity.py vietnam
      python scripts/tie_break_sensitivity.py vietnam --k 25 --n-mc 1000 \\
             --seeds 42 43 44 45 46
Writes results/{market}_tiebreak_sensitivity.csv
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                                          # noqa: E402
import pandas as pd                                         # noqa: E402

from fscore.data.fs_clean import load_scores                # noqa: E402
from fscore.data.team_scores import sectors_from_scores     # noqa: E402
from fscore.grid import run_grid                            # noqa: E402

RESULTS = ROOT / "results"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("market")
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--n-mc", type=int, default=1000)
    ap.add_argument("--n-gmv", type=int, default=300)
    ap.add_argument("--first-year", type=int, default=2012)
    ap.add_argument("--last-year", type=int, default=2024)
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 43, 44, 45, 46, 47, 48, 49])
    args = ap.parse_args()

    market = args.market.lower()
    scores = load_scores(market, ROOT / "data")
    prices = pd.read_csv(ROOT / "data" / f"{market}_prices.csv.gz",
                         parse_dates=["date"])
    sectors = sectors_from_scores(scores)
    years = list(range(args.first_year, args.last_year + 1))

    rows = []
    for seed in args.seeds:
        t0 = time.time()
        study = run_grid(market, scores, prices, sectors, years,
                         k=args.k, n_mc=args.n_mc, n_gmv=args.n_gmv, seed=seed)
        summ = study.summary()
        pl = study.placement("fscore_EW", "mc_ew", "sharpe")
        pl_nf = (study.placement("fscore_EW", "mc_nonf_ew", "sharpe")
                 if study.mc_nonf_ew.shape[1] else {})
        syn = study.synergy()
        diag = study.diagnostics()
        rows.append({
            "seed": seed,
            "fscore_EW_sharpe": round(float(summ.loc["fscore_EW", "sharpe"]), 4),
            "fscore_EW_ann_return": round(float(summ.loc["fscore_EW", "ann_return"]), 4),
            "fscore_GMV_sharpe": round(float(summ.loc["fscore_GMV", "sharpe"]), 4),
            "universe_EW_sharpe": round(float(summ.loc["universe_EW", "sharpe"]), 4),
            "percentile": round(float(pl["percentile"]), 4),
            "p_value": round(float(pl["p_value"]), 4),
            "significant": bool(pl["significant"]),
            "p_value_nonf": (round(float(pl_nf["p_value"]), 4) if pl_nf else np.nan),
            "D_fscore": round(float(syn["D_fscore"]), 4),
            "D_p_value": round(float(syn["p_value"]), 4),
            "mean_tie_break_slots": round(float(diag.tie_break_slots.mean()), 2),
            "years_fully_tie_broken": int((diag.tie_break_slots >= args.k).sum()),
        })
        print(f"  seed {seed}: EW Sharpe {rows[-1]['fscore_EW_sharpe']:.3f}, "
              f"p {rows[-1]['p_value']:.3f}  ({time.time() - t0:.0f}s)", flush=True)

    out = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{market}_tiebreak_sensitivity.csv"
    out.to_csv(path, index=False)

    num = out[["fscore_EW_sharpe", "fscore_EW_ann_return", "fscore_GMV_sharpe",
               "percentile", "p_value", "D_fscore", "D_p_value"]]
    print(f"\n{market}: k={args.k}, N={args.n_mc}, formations "
          f"{years[0]}-{years[-1]}, {len(args.seeds)} seeds")
    print(out.to_string(index=False))
    print("\nspread across seeds")
    print(num.agg(["min", "median", "max", "std"]).round(4).to_string())
    print(f"\nsignificant at 5% in {int(out.significant.sum())} of "
          f"{len(out)} seeds")
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
