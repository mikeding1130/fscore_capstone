"""Generate the Japan single-year demo (02) and the US/Japan full-study
notebooks (03, 04). Idempotent; execution happens separately.

Run:  python scripts/build_notebooks.py
"""
import json
import pathlib

import nbformat as nbf

ROOT = pathlib.Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


# ----------------------------------------------------------------------
# 02: Japan single-year demo — mirror of 01 on synthetic data
# ----------------------------------------------------------------------

def build_japan_demo():
    src = json.loads((NB_DIR / "01_us_fscore_single_year.ipynb").read_text(encoding="utf-8"))
    nb = nbf.v4.new_notebook(metadata=src["metadata"])
    for cell in src["cells"]:
        text = "".join(cell["source"])
        text = (text.replace("United States", "Japan")
                    .replace('prefix="US"', 'prefix="JP"')
                    .replace("SEED = \"Japan\", 2023, 11", "SEED = \"Japan\", 2023, 23"))
        c = (nbf.v4.new_markdown_cell(text) if cell["cell_type"] == "markdown"
             else nbf.v4.new_code_cell(text))
        nb.cells.append(c)
    nbf.write(nb, NB_DIR / "02_japan_fscore_single_year.ipynb")


# ----------------------------------------------------------------------
# 03 / 04: full studies on real (cached) data
# ----------------------------------------------------------------------

CFG = {
    "us": {
        "num": "03", "title": "United States",
        "bench": {"SPY (S&P 500)": "SPY", "VTV (US value ETF)": "VTV"},
        "bench_note": "Both benchmarks are USD, like the portfolios.",
        "ff": "us", "usd_convert": False,
        "universe_note": "current S&P 500 members",
    },
    "japan": {
        "num": "04", "title": "Japan",
        "bench": {"1306.T (TOPIX ETF, JPY)": "1306.T",
                  "EWJV (MSCI Japan Value ETF, USD)": "EWJV"},
        "bench_note": ("1306.T is JPY like the portfolios; EWJV is USD, so its "
                       "row mixes in currency effects and is indicative only."),
        "ff": "japan", "usd_convert": True,
        "universe_note": "current Nikkei 225 members",
    },
}


def full_study_cells(m: dict) -> list:
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = []

    cells.append(md(f"""# {m['title']} — Full Piotroski F-Score Study (formations 2023–2025)

The proposal's complete loop on **real data**: point-in-time universe →
high-B/M value subset → 9-signal F-Score → fixed-basket selection (F-Score vs
value / market-cap / liquidity-matched / random Monte-Carlo) → EW / GMV /
sector-capped GMV on RMT-cleaned covariances → annual-rebalance backtest →
placement in the random distribution → investable benchmarks → Fama-French
three-factor regression.

**Data & scope.** Yahoo Finance serves ~5 annual statement periods per name,
which under the 5-month reporting lag supports July-1 formations in
**2023, 2024 and 2025** (three holding years, annually rebalanced). Universe
membership is {m['universe_note']} — a survivorship caveat documented in
`src/fscore/data/universe.py`; the proposal's 2000–2025 sample needs the
commercial point-in-time source listed as the gating item in `data/README.md`
and plugs into the same canonical schemas.

Run `python scripts/fetch_us_japan.py` once before this notebook (builds the
git-ignored cache under `data/`)."""))

    cells.append(code(f"""import sys, pathlib
ROOT = pathlib.Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from fscore.data.yahoo import load_cached
from fscore.pipeline import run_study
from fscore.evaluation import (metrics, benchmark_returns, fetch_ff_factors,
                               factor_regression, to_usd)

MARKET, YEARS = "{m['num'] == '03' and 'us' or 'japan'}", [2023, 2024, 2025]
fund, prices, sectors, bench = load_cached(MARKET, ROOT / "data")
print(f"fundamentals: {{fund.ticker.nunique()}} tickers, "
      f"FY{{fund.fiscal_year.min()}}–FY{{fund.fiscal_year.max()}}")
print(f"prices: {{prices.ticker.nunique()}} tickers, "
      f"{{prices.date.min():%Y-%m-%d}} → {{prices.date.max():%Y-%m-%d}}")
fund.groupby("fiscal_year").size().rename("statements")"""))

    cells.append(md("""### 1. Run the multi-year study

Per formation year: universe = top-150 names by median dollar volume with
continuous listing and complete, published statements; value subset = top 40%
by B/M (~60 names); baskets of 30; 1,000 random baskets under EW (300 pushed
through the GMV / sector-GMV pipeline)."""))

    cells.append(code("""study = run_study(MARKET, fund, prices, sectors, YEARS,
                  n_mc=1000, n_mc_opt=300, seed=42)
diag = pd.DataFrame([{"year": yr.year, **yr.diagnostics} for yr in study.yearly])
diag.set_index("year")"""))

    cells.append(md("### 2. F-Score distribution within the value universe, per formation"))

    cells.append(code("""fig, axes = plt.subplots(1, len(YEARS), figsize=(11, 3), sharey=True)
for ax, yr in zip(axes, study.yearly):
    counts = yr.scored.fscore.value_counts().sort_index()
    ax.bar(counts.index, counts.values)
    ax.set_title(f"{yr.year} formation"); ax.set_xlabel("F-Score"); ax.set_xticks(range(10))
axes[0].set_ylabel("firms")
fig.suptitle(f"{MARKET.upper()} — F-Score distribution (high-B/M universe)")
plt.tight_layout(); plt.show()

pd.DataFrame({yr.year: yr.scored.fscore.describe() for yr in study.yearly}).round(2)"""))

    bench_map = ",\n         ".join(f'"{k}": "{v}"' for k, v in m["bench"].items())
    cells.append(md(f"""### 3. Chained track record vs investable benchmarks

{m['bench_note']}"""))

    cells.append(code(f"""BENCH = {{{bench_map}}}
start, end = study.daily.index.min(), study.daily.index.max()
bench_rets = {{name: benchmark_returns(bench, tk, start, end)
              for name, tk in BENCH.items()}}

nav = (1 + study.daily.fillna(0)).cumprod()
fig, ax = plt.subplots(figsize=(10, 5))
styles = {{"fscore_EW": ("-", 2.2), "fscore_GMV": ("-", 1.4), "fscore_GMVsec": ("-", 1.4),
          "value_EW": ("--", 1.2), "mktcap_EW": ("--", 1.2), "liquidity_EW": ("--", 1.2)}}
for s in study.daily.columns:
    ls, lw = styles[s]
    ax.plot(nav.index, nav[s], ls, lw=lw, label=s)
for name, r in bench_rets.items():
    ax.plot((1 + r).cumprod(), ":", lw=1.6, label=name)
ax.set_ylabel("growth of 1 (log)"); ax.set_yscale("log")
ax.set_title(f"{{MARKET.upper()}} — F-Score strategies vs controls and benchmarks, "
             f"{{start:%b %Y}} – {{end:%b %Y}}")
ax.legend(fontsize=8, ncol=2); plt.tight_layout(); plt.show()"""))

    cells.append(code("""tbl = study.summary()
for name, r in bench_rets.items():
    tbl.loc[name] = metrics(r)
tbl.round(3)"""))

    cells.append(md("""### 4. Statistical control — placement in the Monte-Carlo random distribution

Same universe, same basket size, same construction pipeline; the only
difference is *which* 30 names. Percentile = share of random baskets the
F-Score portfolio beats; p-value is one-sided."""))

    cells.append(code("""pairs = [("EW", "fscore_EW"), ("GMV", "fscore_GMV"), ("GMVsec", "fscore_GMVsec")]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=False)
for ax, (how, strat) in zip(axes, pairs):
    sharpes = study.mc_summary(how)["sharpe"].dropna()
    fs = metrics(study.daily[strat].dropna())["sharpe"]
    ax.hist(sharpes, bins=30, alpha=0.75)
    ax.axvline(fs, color="crimson", lw=2)
    ax.set_title(f"{how}: F-Score={fs:.2f}", fontsize=10)
    ax.set_xlabel("Sharpe (chained 3y)")
axes[0].set_ylabel("random baskets")
fig.suptitle(f"{MARKET.upper()} — F-Score basket vs random distribution, by construction")
plt.tight_layout(); plt.show()

placements = pd.concat({strat: study.placement(strat, how) for how, strat in pairs})
placements.round(3)"""))

    cells.append(md("""### 5. Turnover and implied trading cost

One-way turnover between consecutive annual rebalances; cost drag assumes
20 bp per side (2 x turnover x 20 bp per year)."""))

    cells.append(code("""to = study.turnover_table()
to.loc["mean"] = to.mean()
cost = (2 * to.loc["mean"] * 0.0020).rename("annual cost drag")
pd.concat([to.round(3).T, cost.round(4)], axis=1)"""))

    ff_lines = f'''ff = fetch_ff_factors("{m['ff']}")'''
    if m["usd_convert"]:
        ff_lines += '''
# Ken French Japan factors are USD-denominated -> convert JPY returns to USD
fx = bench[bench.ticker == "JPY=X"].set_index("date")["adj_close"]
series = {s: to_usd(study.daily[s].dropna(), fx)
          for s in ["fscore_EW", "fscore_GMV", "value_EW"]}'''
    else:
        ff_lines += '''
series = {s: study.daily[s].dropna()
          for s in ["fscore_EW", "fscore_GMV", "value_EW"]}'''

    cells.append(md("""### 6. Does alpha survive the factor exposures?

Daily excess returns regressed on the Fama-French three factors (market,
size, value) with Newey-West standard errors."""
                    + (" Japan portfolio returns are converted to USD to match "
                       "the USD-denominated Japan factor set." if m["usd_convert"] else "")))

    cells.append(code(f"""{ff_lines}
reg = pd.DataFrame({{s: factor_regression(r, ff) for s, r in series.items()}}).T
reg.round(4)"""))

    cells.append(code("""RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
tbl.to_csv(RESULTS / f"{MARKET}_summary.csv")
placements.to_csv(RESULTS / f"{MARKET}_mc_placement.csv")
to.to_csv(RESULTS / f"{MARKET}_turnover.csv")
reg.to_csv(RESULTS / f"{MARKET}_factor_regression.csv")
diag.to_csv(RESULTS / f"{MARKET}_diagnostics.csv", index=False)
print("saved to", RESULTS)"""))

    cells.append(md("### 7. Reading the results\n\n*(filled in after execution)*"))
    return cells


def build_full(market: str):
    m = CFG[market]
    nb = nbf.v4.new_notebook(metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python"},
    })
    nb.cells = full_study_cells(m)
    path = NB_DIR / f"{m['num']}_{market}_full_study.ipynb"
    nbf.write(nb, path)
    print("wrote", path)


if __name__ == "__main__":
    build_japan_demo()
    print("wrote", NB_DIR / "02_japan_fscore_single_year.ipynb")
    build_full("us")
    build_full("japan")
