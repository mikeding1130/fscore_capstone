"""Generate the reviewer-suggested notebook grid: one .ipynb per
(country x basket size x random-sample size) = 2 x 3 x 3 = 18 notebooks
under notebooks/grid/, all figures saved via plt.savefig(..., dpi=300).

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
MARKETS = {"us": ("United States", "list(range(2021, 2026))"),
           "japan": ("Japan", "list(range(2022, 2026))")}


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
July 1 using score year T-1 (one conservative timing rule for both markets);
the evaluation window ends 2025-12-31.

Peer-review design points (see `src/fscore/grid.py` docstring): explicit
random basis = full eligible universe with fresh draws each year and reported
overlap; a non-F-Score random control; strict F ≥ 8 portfolio; universe EW
and plain universe minimum-variance controls; denoised (not detoned) GMV;
primary measure fixed in advance = net-of-cost Sharpe (20 bp per side,
rf = 0); and the synergy test D = Sharpe(GMV) − Sharpe(EW) computed per
basket. All figures are saved at dpi = 300."""),
        code(f"""import sys, pathlib
ROOT = pathlib.Path.cwd().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from fscore.data.team_scores import load_team_scores, sectors_from_scores
from fscore.grid import run_grid

MARKET, K, N_MC = "{market}", {k}, {n_mc}
YEARS = {years}
FIG = ROOT / "results" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "results" / "grid"; OUT.mkdir(parents=True, exist_ok=True)
TAG = "{tag}"

scores = load_team_scores(MARKET, ROOT / "data")
prices = pd.read_csv(ROOT / "data" / f"{{MARKET}}_prices.csv.gz", parse_dates=["date"])
sectors = sectors_from_scores(scores)
study = run_grid(MARKET, scores, prices, sectors, YEARS, k=K, n_mc=N_MC,
                 n_gmv=300, seed=42)
diag = study.diagnostics()
diag.round(3)"""),
        md("### 1. Summary (primary measure: net-of-cost Sharpe; rf = 0, "
           "costs 20 bp per side on one-way turnover)"),
        code("""summary = study.summary()
summary.round(3)"""),
        md("### 2. Yearly returns (July–June holding years; final year ends Dec 2025)"),
        code("""yearly = study.yearly_returns()
(yearly * 100).round(1)"""),
        md("### 3. Placement vs the random distributions"),
        code("""rows = {}
for pool, label in [("mc_ew", "random (full universe)"),
                    ("mc_nonf_ew", "random (non-F-Score names)")]:
    frame = getattr(study, pool)
    if frame.shape[1]:
        for stat in ["sharpe", "ann_return"]:
            rows[(label, stat)] = study.placement("fscore_EW", pool, stat)
placement = pd.DataFrame(rows).T
placement.round(3)"""),
        code("""fig, ax = plt.subplots(figsize=(7, 3.5))
sh = study.mc_metric(study.mc_ew, "sharpe")
ax.hist(sh, bins=40, alpha=0.75, label=f"{N_MC} random baskets (EW)")
fs = placement.loc[("random (full universe)", "sharpe"), "fscore"]
ax.axvline(fs, color="crimson", lw=2, label=f"F-Score EW = {fs:.2f}")
ax.set_xlabel("Sharpe (chained, gross)"); ax.set_ylabel("baskets"); ax.legend(fontsize=8)
ax.set_title(f"{MARKET.upper()} k={K}: F-Score vs {N_MC} random baskets")
plt.tight_layout()
plt.savefig(FIG / f"{TAG}_mc_hist.png", dpi=300)
plt.show()"""),
        md("### 4. Synergy test (reviewer priority 1): per-basket "
           "optimisation gain D = Sharpe(GMV) − Sharpe(EW)"),
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
plt.savefig(FIG / f"{TAG}_synergy_hist.png", dpi=300)
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
plt.savefig(FIG / f"{TAG}_nav.png", dpi=300)
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
