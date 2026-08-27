"""Generate the reviewer-suggested grid study: one notebook per market, each
sweeping (basket size k) x (random-sample size N) internally.

The grid is 3 x 3 x 2 = 18 cells, as the review asked. It used to be 18
notebooks, one per cell, which meant eighteen copies of the same prose and the
same loading code, and a methodology fix had to be made eighteen times. The
sweep now runs inside two notebooks - `notebooks/grid/us_grid.ipynb` and
`notebooks/grid/japan_grid.ipynb` - and writes exactly the same per-cell
outputs under the same `{market}_k{k}_mc{N}_*` names.

Results are unchanged by the merge, and that is a property of the seeding
rather than a hope: every random draw in `fscore.grid` and
`fscore.selection.baskets` comes from a locally constructed
`np.random.default_rng(seed + year)`, never from the global NumPy state, so a
cell's output depends only on (market, k, N, seed) and not on what ran before
it in the same process. `scripts/verify_grid_merge.py` checks this against the
saved baseline rather than taking it on trust.

Run:  python scripts/build_grid_notebooks.py          # build
      python scripts/build_grid_notebooks.py execute  # build + run both
"""
import pathlib
import sys

import nbformat as nbf

ROOT = pathlib.Path(__file__).resolve().parents[1]
GRID_DIR = ROOT / "notebooks" / "grid"

KS = [20, 25, 30]
MCS = [1000, 2000, 5000]
# Formations July 2012 .. July 2024 — thirteen chained holding years, every
# one of them complete. July 2024 is the last formation whose full year ends
# inside the sample (June 2025); adding July 2025 would contribute a half
# year and mix an incomplete window in with the complete ones.
MARKETS = {"us": ("United States", "list(range(2012, 2025))"),
           "japan": ("Japan", "list(range(2012, 2025))")}

# Each market's grid now reads the same statements its main study reads, so
# the grid answers a robustness question about the reported dataset rather
# than about the team workbook it used to sit on.
LOADERS = {
    "us": '''from fscore.data.edgar import load_membership
fund = pd.read_csv(ROOT / "data" / "us_fundamentals.csv", parse_dates=["report_date"])
sectors_map = pd.read_csv(ROOT / "data" / "us_sectors.csv").set_index("ticker")["sector"]
membership = load_membership(ROOT / "data")''',
    "japan": '''from fscore.data.bbg_processed import constituents
fund = pd.read_csv(ROOT / "data" / "japan_bbg_fundamentals.csv", parse_dates=["report_date"])
sectors_map = pd.read_csv(ROOT / "data" / "japan_sectors.csv").set_index("ticker")["sector"]
membership = constituents(MARKET, ROOT / "data", YEARS)''',
}
SOURCE_NOTE = {
    "us": ("SEC EDGAR XBRL statements with true 10-K filing dates, screened to "
           "the S&P 500 constituent list as of each formation."),
    "japan": ("Bloomberg statements from `data/processed/Japan/`, screened to "
              "the TPX100 constituent list as of each formation. The universe "
              "is ~100 names against the US 500, so the high-B/M subset is "
              "roughly 35 and **k = 30 covers 83–91% of it** — the random "
              "basket and the F-Score basket then share most of their names, "
              "and the comparison has little left to detect. **k = 20 is the "
              "interpretable cell for Japan.**"),
}


def cells(market: str):
    title, years = MARKETS[market]
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    return [
        md(f"""# {title} — F-Score grid study: k ∈ {{20, 25, 30}} × N ∈ {{1000, 2000, 5000}}

The {title} half of the reviewer-suggested 3 × 3 × 2 grid (basket sizes ×
random-sample sizes × two markets). All nine cells run in this one notebook;
each writes `results/grid/{market}_k{{k}}_mc{{N}}_*.csv` and its four figures
exactly as before, so the per-cell outputs are unchanged.

Signals come from the team's FS_clean statements, scored by this repository's
own signal code (`fscore.signal.piotroski`, unit-tested, beginning-of-year
asset scaling on both sides of every delta); prices from the cached Yahoo
data. Formations are July 1 of **2012–2024** (13 chained holding years, each a
full twelve months) using score year T−1 — one conservative timing rule for
both markets, over identical calendar time so the two are comparable. July
2024 is the last formation whose complete year finishes inside the sample.
Covariances are estimated on 36 months of daily returns ending the day before
formation. Note the panel resolves to *currently listed* symbols, so the
universe is survivorship-tilted — disclosed as a data limitation. The value
control falls back to the universe in years before B/M coverage begins
(flagged in the diagnostics).

Peer-review design points (see `src/fscore/grid.py` docstring): explicit
random basis = full eligible universe with fresh draws each year and reported
overlap; a non-F-Score random control; strict F ≥ 8 portfolio; universe EW and
plain universe minimum-variance controls; a dollar-neutral long-short book
(long top-k scores, short bottom-k, charged both legs' trading costs plus a
stock-borrow fee) wherever shorting is available — Vietnam runs long-only, so
`fscore_LS` is simply absent there; denoised (not detoned) GMV; primary
measure fixed in advance = the GROSS Sharpe ratio (rf = 0), with turnover and
net-of-cost figures reported separately; and the synergy test
D = Sharpe(GMV) − Sharpe(EW) computed per basket. All figures are saved at
dpi = 300.

**On seeding.** Every cell calls `run_grid(..., seed=42)`, and every random
draw inside it is made by a locally constructed
`np.random.default_rng(seed + year)` — the global NumPy generator is never
touched. A cell's result is therefore a function of (market, k, N, seed)
alone, which is what lets all nine share a process without affecting each
other."""),
        code(f"""import sys, pathlib, time
ROOT = pathlib.Path.cwd().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from fscore.data.score_panel import build_score_panel
from fscore.data.team_scores import sectors_from_scores
from fscore.grid import run_grid
from fscore.plotting import setup_plots, save_fig

setup_plots()          # study-wide figure defaults; every saved chart is 300 dpi
MARKET = "{market}"
KS, MCS = {KS}, {MCS}
YEARS = {years}
FIG = ROOT / "results" / "figures"
OUT = ROOT / "results" / "grid"; OUT.mkdir(parents=True, exist_ok=True)

# Loaded once for all nine cells rather than once per notebook. The source is
# the same one this market's main study uses, so the grid varies k and N
# against the reported dataset instead of a different one.
{LOADERS[market]}
prices = pd.read_csv(ROOT / "data" / f"{{MARKET}}_prices.csv.gz", parse_dates=["date"])
scores = build_score_panel(fund, [y - 1 for y in YEARS],
                           sectors=sectors_map, membership=membership)
per_year = scores.attrs["per_year"]
sectors = sectors_from_scores(scores)
print(f"{{MARKET.upper()}}: {{scores.ticker.nunique()}} tickers, "
      f"{{len(scores)}} scored firm-years, formations {{YEARS[0]}}–{{YEARS[-1]}}")"""),
        md(f"""### 0. Data discarded, and how EQ_OFFER was measured

**Source.** {SOURCE_NOTE[market]}

A firm-year enters the study only with a **complete nine-signal F-Score**.
Partial scores are dropped rather than summed over whatever is available —
an incomplete score is not a low score, and keeping them would push those
firms towards the bottom of the ranking and into the short leg.

`no_tm2_assets` counts firm-years scored without a t−2 statement, where the
prior-year ratio falls back to period-end assets; they are scored, not
dropped, and counted so the fallback is visible.

**EQ_OFFER.** The faithful measure is the cash-flow statement's
equity-issuance line; the substitute asks whether the share count rose. They
disagree on 38.2% of US firm-years, and not symmetrically — share counts miss
issuance that buybacks net away, so the substitute scores firms **more
generously** (mean F 6.08 against 5.76, placement flattered by ~14 percentile
points; see `results/eq_offer_sensitivity.csv`). The split below shows which
measure each firm-year actually used, so a market running entirely on the
substitute is read with that bias in mind.

This accounting is a property of the source data, so it is identical across
all nine cells — computed once here, then written under every cell's tag so
each cell's output set stays self-contained."""),
        code("""tot = per_year.sum(numeric_only=True)
print(f"{MARKET.upper()}: {int(tot.scored)} scored firm-years, "
      f"{int(tot.dropped_incomplete_signals)} dropped for incomplete signals "
      f"({100 * tot.dropped_incomplete_signals / (tot.scored + tot.dropped_incomplete_signals):.1f}%)")
print(f"  EQ_OFFER from the cash-flow line : {int(tot.eq_offer_from_cashflow)}")
print(f"  EQ_OFFER from the share count    : {int(tot.eq_offer_from_shares)}")
print(f"  scored without a t-2 statement   : {int(tot.no_tm2_assets)}")
drops_total = per_year.set_index("score_year")
drops_total"""),
        md("""### 1. The sweep

Nine cells, each the full study at one (k, N). Every cell writes the same five
CSVs and four figures it wrote when it was its own notebook. The tables and
charts for a chosen reference cell are shown afterwards; the consolidated grid
follows at the end."""),
        code("""def run_cell(k, n_mc):
    \"\"\"One grid cell: run, save the five CSVs and four figures, return the
    pieces the consolidated table needs.\"\"\"
    tag = f"{MARKET}_k{k}_mc{n_mc}"
    study = run_grid(MARKET, scores, prices, sectors, YEARS, k=k, n_mc=n_mc,
                     n_gmv=300, seed=42)
    diag = study.diagnostics()
    summary = study.summary()
    yearly = study.yearly_returns()

    rows = {}
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
    syn = study.synergy()

    # --- figures: same four, same names, same 300 dpi ---
    # left: what the source holds vs what is scored; right: which EQ_OFFER
    # measure each firm-year used, since the substitute is the generous one
    b = per_year.set_index("score_year")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    axes[0].bar(b.index, b.scored, label="scored", color="tab:blue")
    axes[0].bar(b.index, b.dropped_incomplete_signals, bottom=b.scored,
                label="incomplete nine signals", color="tab:grey")
    axes[0].set_xlabel("score year"); axes[0].set_ylabel("firm-years")
    axes[0].set_title(f"{MARKET.upper()}: held vs scored")
    axes[0].legend(fontsize=7)
    axes[1].bar(b.index, b.eq_offer_from_cashflow, label="cash-flow line",
                color="tab:green")
    axes[1].bar(b.index, b.eq_offer_from_shares, bottom=b.eq_offer_from_cashflow,
                label="share count (generous)", color="tab:orange")
    axes[1].set_xlabel("score year"); axes[1].set_ylabel("firm-years")
    axes[1].set_title(f"{MARKET.upper()}: EQ_OFFER measure used")
    axes[1].legend(fontsize=7)
    plt.tight_layout(); save_fig(f"{tag}_exclusions", directory=FIG); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    sh = study.mc_metric(study.mc_ew, "sharpe")
    ax.hist(sh, bins=40, alpha=0.75, label=f"{n_mc} random baskets (EW)")
    # the placement index is (pool, statistic, basis); the histogram shows gross
    fs = placement.loc[("random (full universe)", "sharpe", "gross"), "fscore"]
    ax.axvline(fs, color="crimson", lw=2, label=f"F-Score EW = {fs:.2f}")
    ax.set_xlabel("Sharpe (chained, gross)"); ax.set_ylabel("baskets"); ax.legend(fontsize=8)
    ax.set_title(f"{MARKET.upper()} k={k}: F-Score vs {n_mc} random baskets")
    plt.tight_layout(); save_fig(f"{tag}_mc_hist", directory=FIG); plt.close(fig)

    d_rand = (study.mc_metric(study.mc_gmv, "sharpe")
              - study.mc_metric(study.mc_ew[list(range(study.mc_gmv.shape[1]))],
                                "sharpe")).dropna()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(d_rand, bins=30, alpha=0.75, label=f"D over {len(d_rand)} random baskets")
    ax.axvline(syn["D_fscore"], color="crimson", lw=2,
               label=f"D(F-Score) = {syn['D_fscore']:.2f}")
    ax.set_xlabel("D = Sharpe(GMV) - Sharpe(EW)"); ax.set_ylabel("baskets")
    ax.legend(fontsize=8)
    ax.set_title(f"{MARKET.upper()} k={k}: optimisation gain, F-Score basket vs random")
    plt.tight_layout(); save_fig(f"{tag}_synergy_hist", directory=FIG); plt.close(fig)

    nav = (1 + study.daily.fillna(0)).cumprod()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for c in nav.columns:
        ax.plot(nav.index, nav[c], lw=1.6 if c.startswith("fscore") else 1.0,
                ls="-" if c.startswith("fscore") else "--", label=c)
    ax.set_yscale("log"); ax.set_ylabel("growth of 1 (log)"); ax.legend(fontsize=7, ncol=2)
    ax.set_title(f"{MARKET.upper()} k={k}, {n_mc} draws — strategies and controls")
    plt.tight_layout(); save_fig(f"{tag}_nav", directory=FIG); plt.close(fig)

    # --- the same five CSVs, under the same names ---
    drops_total.to_csv(OUT / f"{tag}_exclusions.csv")
    summary.to_csv(OUT / f"{tag}_summary.csv")
    placement.to_csv(OUT / f"{tag}_placement.csv")
    pd.Series(syn).to_frame("value").to_csv(OUT / f"{tag}_synergy.csv")
    yearly.to_csv(OUT / f"{tag}_yearly_returns.csv")
    diag.to_csv(OUT / f"{tag}_diagnostics.csv")
    return study, summary, placement, syn, yearly, diag"""),
        code("""cells_out = {}
for k in KS:
    for n_mc in MCS:
        t0 = time.time()
        cells_out[(k, n_mc)] = run_cell(k, n_mc)
        print(f"  k={k:>2} N={n_mc:>4}  done in {time.time()-t0:5.0f}s", flush=True)
print(f"\\n{len(cells_out)} cells complete")"""),
        md("""### 2. Reference cell — k = 30, N = 5000

The largest basket and the finest random distribution, shown in full. Every
other cell wrote the same tables and figures to disk; the consolidated grid in
section 4 is where they are compared.

**Primary measure: gross Sharpe; rf = 0.** Gross performance is the
cross-country convention — cost models differ by market and would otherwise
confound the comparison. Turnover is reported beside it, and the net-of-cost
columns follow as a sensitivity.

`nominal_k` is the basket size; `effective_n` is 1/Σw², the number of holdings
the weights actually amount to. They are not the same quantity: an optimised
or sector-capped book concentrates, so its effective N sits below the names
nominally held."""),
        code("""REF = (30, 5000)
study, summary, placement, syn, yearly, diag = cells_out[REF]
diag.round(3)"""),
        code("""summary[["ann_return", "ann_vol", "sharpe", "max_drawdown",
         "nominal_k", "effective_n"]].round(3)"""),
        md("""#### Turnover and the net-of-cost sensitivity (reported separately)"""),
        code("""summary[["turnover", "cost_drag", "net_ann_return", "net_sharpe"]].round(4)"""),
        md("### Yearly returns (each formation held July–June, a full twelve months)"),
        code("""(yearly * 100).round(1)"""),
        md("""### 3. Placement vs the random distributions

Reported gross and net of costs. The random control is redrawn every year, so
it carries its own turnover (~1 − k/|universe|) — charging both sides is the
like-for-like comparison; the gross rows show costs are not driving it.

Significance is judged at **one level fixed in advance: 5%** (`significant`
column = p < 0.05). There is no 1% or 10% tier: p = 0.06 is not
significant."""),
        code("""print(f"turnover — F-Score EW {study.strategy_turnover('fscore_EW'):.3f} "
      f"vs random basket {study.mc_turnover():.3f} (one-way, per year)")
placement.round(3)"""),
        code("""from IPython.display import Image, display
k, n = REF
for suffix in ["mc_hist", "synergy_hist", "nav"]:
    display(Image(filename=str(FIG / f"{MARKET}_k{k}_mc{n}_{suffix}.png"), width=640))"""),
        md("""### 4. The consolidated grid

One row per cell. This is what the grid was built to show: whether the
conclusion moves with the basket size or with the number of random draws.

`N` should not move anything but the resolution of the p-value — the same
seed draws the same first 1,000 baskets whether the run asks for 1,000 or
5,000 — so any drift down an `N` block is Monte-Carlo noise, not a finding.
Movement across `k` blocks is the real sensitivity."""),
        code("""rows = []
for (k, n_mc), (st, summ, place, sy, yr, dg) in cells_out.items():
    pl = place.loc[("random (full universe)", "sharpe", "gross")]
    rows.append({
        "mkt": MARKET, "k": k, "N": n_mc,
        "EW": round(float(summ.loc["fscore_EW", "sharpe"]), 4),
        "uniEW": round(float(summ.loc["universe_EW", "sharpe"]), 4)
                 if "universe_EW" in summ.index else np.nan,
        "value": round(float(summ.loc["value_EW", "sharpe"]), 4)
                 if "value_EW" in summ.index else np.nan,
        "to_EW": round(float(summ.loc["fscore_EW", "turnover"]), 4),
        "effN_GMV": round(float(summ.loc["fscore_GMV", "effective_n"]), 4)
                    if "fscore_GMV" in summ.index else np.nan,
        "pct": round(float(pl["percentile"]), 4),
        "p": round(float(pl["p_value"]), 4),
        "sig": bool(pl["significant"]),
        "D": round(float(sy["D_fscore"]), 4),
        "D_p": round(float(sy["p_value"]), 4),
    })
grid = pd.DataFrame(rows).sort_values(["k", "N"]).reset_index(drop=True)
grid.to_csv(OUT / f"{MARKET}_grid_summary.csv", index=False)
grid"""),
        code("""fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for n_mc, mark in zip(MCS, ["o", "s", "^"]):
    sub = grid[grid.N == n_mc]
    axes[0].plot(sub.k, sub.EW, mark + "-", label=f"N={n_mc}")
    axes[1].plot(sub.k, sub.pct, mark + "-", label=f"N={n_mc}")
axes[0].set_xlabel("basket size k"); axes[0].set_ylabel("gross Sharpe (F-Score EW)")
axes[0].set_title(f"{MARKET.upper()}: Sharpe across the grid")
axes[1].axhline(0.95, color="crimson", ls=":", lw=1.2, label="95th pct")
axes[1].set_xlabel("basket size k"); axes[1].set_ylabel("percentile vs random")
axes[1].set_title(f"{MARKET.upper()}: placement across the grid")
for a in axes:
    a.set_xticks(KS); a.legend(fontsize=8)
plt.tight_layout(); save_fig(f"{MARKET}_grid_sensitivity", directory=FIG); plt.show()"""),
        md("""### 5. Outputs

Each cell's five CSVs and four figures are already on disk under its own
`{market}_k{k}_mc{N}_*` tag — unchanged from when every cell was its own
notebook. This notebook adds one file the eighteen never produced: the
consolidated `{market}_grid_summary.csv` above, which used to be assembled by
hand."""),
        code("""print(f"per-cell outputs : results/grid/{MARKET}_k*_mc*_*.csv  "
      f"({len(cells_out) * 6} files)")
print(f"per-cell figures : results/figures/{MARKET}_k*_mc*_*.png  "
      f"({len(cells_out) * 4} files)")
print(f"consolidated     : results/grid/{MARKET}_grid_summary.csv")"""),
    ]


def build_all():
    GRID_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for market in MARKETS:
        nb = nbf.v4.new_notebook(metadata={
            "kernelspec": {"display_name": "Python 3",
                           "language": "python", "name": "python3"},
            "language_info": {"name": "python"}})
        nb.cells = cells(market)
        p = GRID_DIR / f"{market}_grid.ipynb"
        nbf.write(nb, p)
        paths.append(p)
    print(f"built {len(paths)} notebooks in {GRID_DIR} "
          f"({len(KS) * len(MCS)} grid cells each)")
    return paths


def execute_all(paths):
    from nbclient import NotebookClient
    import time
    for p in paths:
        t0 = time.time()
        nb = nbf.read(p, as_version=4)
        NotebookClient(nb, timeout=14400,
                       resources={"metadata": {"path": str(GRID_DIR)}}).execute()
        nbf.write(nb, p)
        print(f"{p.name}: OK in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ps = build_all()
    if len(sys.argv) > 1 and sys.argv[1] == "execute":
        execute_all(ps)
