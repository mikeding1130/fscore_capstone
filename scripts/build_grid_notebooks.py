"""Generate the reviewer-suggested notebook grid: one .ipynb per
(country x basket size x random-sample size) = 2 x 3 x 3 = 18 notebooks
under notebooks/grid/, every figure saved at 300 dpi via
fscore.plotting.save_fig.

Run:  python scripts/build_grid_notebooks.py          # build
      python scripts/build_grid_notebooks.py execute  # build + run all 18
"""
import pathlib
import sys

import nbformat as nbf

ROOT = pathlib.Path(__file__).resolve().parents[1]
GRID_DIR = ROOT / "notebooks" / "grid"

KS = [20, 25, 30]
MCS = [1000, 2000, 5000]
# Study window agreed for both markets: formations July 2012 .. July 2025
# (14 chained holding years), evaluation capped at 2025-12-31. The panel is
# scoreable from 2002, but the sample is held to the common window so the
# two markets are compared over identical calendar time.
MARKETS = {"us": ("United States", "list(range(2012, 2026))"),
           "japan": ("Japan", "list(range(2012, 2026))")}


def cells(market: str, k: int, n_mc: int):
    title, years = MARKETS[market]
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    tag = f"{market}_k{k}_mc{n_mc}"
    return [
        md(f"""# {title} — F-Score grid study: basket k = {k}, random draws = {n_mc}

One cell of the reviewer-suggested 3 x 3 x 2 grid (basket sizes 20/25/30 x
random-sample sizes 1,000/2,000/5,000 x two markets). Signals come from the
team-computed workbook (`data/processed/`, exact Piotroski conventions,
financial firms removed); prices from the cached Yahoo data. Formations are
July 1 of **2012-2025** (14 chained holding years) using score year T-1 —
one conservative timing rule for both markets, over identical calendar time
so the two are comparable. The evaluation window ends 2025-12-31.
Covariances are estimated on 36 months of daily returns ending the day
before formation. Note the panel resolves to *currently listed* symbols, so
the universe is survivorship-tilted — disclosed as a data limitation. The value control falls back to the universe
in years before B/M coverage begins (flagged in the diagnostics).

Peer-review design points (see `src/fscore/grid.py` docstring): explicit
random basis = full eligible universe with fresh draws each year and reported
overlap; a non-F-Score random control; strict F ≥ 8 portfolio; universe EW
and plain universe minimum-variance controls; a dollar-neutral long-short
book (long top-k scores, short bottom-k, charged both legs' trading costs
plus a stock-borrow fee) wherever shorting is available — Vietnam runs
long-only, so `fscore_LS` is simply absent there; denoised
(not detoned) GMV;
primary measure fixed in advance = net-of-cost Sharpe (20 bp per side,
rf = 0); and the synergy test D = Sharpe(GMV) − Sharpe(EW) computed per
basket. All figures are saved at dpi = 300."""),
        code(f"""import sys, pathlib
ROOT = pathlib.Path.cwd().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from fscore.data.team_scores import (load_team_scores, sectors_from_scores,
                                     exclusion_report)
from fscore.grid import run_grid
from fscore.plotting import setup_plots, save_fig

setup_plots()          # study-wide figure defaults; every saved chart is 300 dpi
MARKET, K, N_MC = "{market}", {k}, {n_mc}
YEARS = {years}
FIG = ROOT / "results" / "figures"
OUT = ROOT / "results" / "grid"; OUT.mkdir(parents=True, exist_ok=True)
TAG = "{tag}"

scores = load_team_scores(MARKET, ROOT / "data")
prices = pd.read_csv(ROOT / "data" / f"{{MARKET}}_prices.csv.gz", parse_dates=["date"])
sectors = sectors_from_scores(scores)
study = run_grid(MARKET, scores, prices, sectors, YEARS, k=K, n_mc=N_MC,
                 n_gmv=300, seed=42)
diag = study.diagnostics()
diag.round(3)"""),
        md("""### 0. Data discarded before any test

A firm-year enters the study only with a **complete nine-signal F-Score**.
Partial scores are dropped rather than summed over whatever is available —
an incomplete score is not a low score, and keeping them would push those
firms towards the bottom of the ranking and into the short leg. The table
below is the full accounting; `dropped_no_price` in the diagnostics above
counts the further names removed for insufficient price history."""),
        code("""drops_total = exclusion_report(MARKET, ROOT / "data")
drops_year = exclusion_report(MARKET, ROOT / "data", by_year=True)
drops_total.to_csv(OUT / f"{TAG}_exclusions.csv")
print(f"{MARKET.upper()}: {int(drops_total.dropped_incomplete_signals.iloc[0])} of "
      f"{int(drops_total.rows.iloc[0])} firm-years dropped for incomplete signals "
      f"({drops_total.pct_dropped_incomplete.iloc[0]:.1f}%); "
      f"{drops_total.pct_kept.iloc[0]:.1f}% kept")
drops_total"""),
        md("""### 1. Summary (primary measure: net-of-cost Sharpe; rf = 0)

Costs are 20 bp per side charged on each strategy's **own** one-way turnover,
computed from its actual weights — so the optimised portfolios pay for weight
drift and the near-static universe control pays almost nothing. The
long-short book (`fscore_LS`, present only where shorting is available) pays
both legs plus a 100 bp annual stock-borrow fee on its short notional."""),
        code("""summary = study.summary()
summary[["ann_return", "ann_vol", "sharpe", "max_drawdown",
         "turnover", "cost_drag", "net_ann_return", "net_sharpe"]].round(3)"""),
        md("### 2. Yearly returns (July–June holding years; final year ends Dec 2025)"),
        code("""yearly = study.yearly_returns()
(yearly * 100).round(1)"""),
        md("""### 3. Placement vs the random distributions

Reported gross and net of costs. The random control is redrawn every year, so
it carries its own turnover (~1 − k/|universe|) — charging both sides is the
like-for-like comparison; the gross rows show costs are not driving it.

Significance is judged at **one level fixed in advance: 5%** (`significant`
column = p < 0.05). There is no 1% or 10% tier: p = 0.06 is not
significant."""),
        code("""rows = {}
for pool, label in [("mc_ew", "random (full universe)"),
                    ("mc_nonf_ew", "random (non-F-Score names)")]:
    frame = getattr(study, pool)
    if frame.shape[1]:
        for stat in ["sharpe", "ann_return"]:
            for net in (False, True):
                basis = "net" if net else "gross"
                rows[(label, stat, basis)] = study.placement(
                    "fscore_EW", pool, stat, net=net)
placement = pd.DataFrame(rows).T
print(f"turnover — F-Score EW {study.strategy_turnover('fscore_EW'):.3f} "
      f"vs random basket {study.mc_turnover():.3f} (one-way, per year)")
placement.round(3)"""),
        code("""fig, ax = plt.subplots(figsize=(7, 3.5))
sh = study.mc_metric(study.mc_ew, "sharpe")
ax.hist(sh, bins=40, alpha=0.75, label=f"{N_MC} random baskets (EW)")
# the placement index is (pool, statistic, basis); the histogram shows gross
fs = placement.loc[("random (full universe)", "sharpe", "gross"), "fscore"]
ax.axvline(fs, color="crimson", lw=2, label=f"F-Score EW = {fs:.2f}")
ax.set_xlabel("Sharpe (chained, gross)"); ax.set_ylabel("baskets"); ax.legend(fontsize=8)
ax.set_title(f"{MARKET.upper()} k={K}: F-Score vs {N_MC} random baskets")
plt.tight_layout()
save_fig(f"{TAG}_mc_hist", directory=FIG)
plt.show()"""),
        md("""### 4. Synergy test: per-basket optimisation gain
D = Sharpe(GMV) − Sharpe(EW), judged at the 5% level."""),
        code("""syn = study.synergy()
d_rand = (study.mc_metric(study.mc_gmv, "sharpe")
          - study.mc_metric(study.mc_ew[list(range(study.mc_gmv.shape[1]))], "sharpe")).dropna()
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.hist(d_rand, bins=30, alpha=0.75, label=f"D over {len(d_rand)} random baskets")
ax.axvline(syn["D_fscore"], color="crimson", lw=2,
           label=f"D(F-Score) = {syn['D_fscore']:.2f}")
ax.set_xlabel("D = Sharpe(GMV) - Sharpe(EW)"); ax.set_ylabel("baskets"); ax.legend(fontsize=8)
ax.set_title(f"{MARKET.upper()} k={K}: optimisation gain, F-Score basket vs random")
plt.tight_layout()
save_fig(f"{TAG}_synergy_hist", directory=FIG)
plt.show()
pd.Series(syn).round(3)"""),
        md("### 5. Track record and outputs"),
        code("""nav = (1 + study.daily.fillna(0)).cumprod()
fig, ax = plt.subplots(figsize=(9, 4.5))
for c in nav.columns:
    ax.plot(nav.index, nav[c], lw=1.6 if c.startswith("fscore") else 1.0,
            ls="-" if c.startswith("fscore") else "--", label=c)
ax.set_yscale("log"); ax.set_ylabel("growth of 1 (log)"); ax.legend(fontsize=7, ncol=2)
ax.set_title(f"{MARKET.upper()} k={K}, {N_MC} draws — strategies and controls")
plt.tight_layout()
save_fig(f"{TAG}_nav", directory=FIG)
plt.show()

summary.to_csv(OUT / f"{TAG}_summary.csv")
placement.to_csv(OUT / f"{TAG}_placement.csv")
pd.Series(syn).to_frame("value").to_csv(OUT / f"{TAG}_synergy.csv")
yearly.to_csv(OUT / f"{TAG}_yearly_returns.csv")
diag.to_csv(OUT / f"{TAG}_diagnostics.csv")
print("saved", TAG)"""),
    ]


def build_all():
    GRID_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for market in MARKETS:
        for k in KS:
            for n_mc in MCS:
                nb = nbf.v4.new_notebook(metadata={
                    "kernelspec": {"display_name": "Python 3",
                                   "language": "python", "name": "python3"},
                    "language_info": {"name": "python"}})
                nb.cells = cells(market, k, n_mc)
                p = GRID_DIR / f"{market}_k{k}_mc{n_mc}.ipynb"
                nbf.write(nb, p)
                paths.append(p)
    print(f"built {len(paths)} notebooks in {GRID_DIR}")
    return paths


def execute_all(paths):
    from nbclient import NotebookClient
    import time
    for p in paths:
        t0 = time.time()
        nb = nbf.read(p, as_version=4)
        NotebookClient(nb, timeout=3600,
                       resources={"metadata": {"path": str(GRID_DIR)}}).execute()
        nbf.write(nb, p)
        print(f"{p.name}: OK in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ps = build_all()
    if len(sys.argv) > 1 and sys.argv[1] == "execute":
        execute_all(ps)
