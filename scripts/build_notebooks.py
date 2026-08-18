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
        "years": "list(range(2012, 2026))", "lag": 1, "membership": True,
        "data_note": (
            "SEC EDGAR XBRL fundamentals (FY2009 onward — XBRL was mandated "
            "2009-2011, which is why the proposal's 2000 start is not reachable "
            "from free filings) with **true 10-K filing dates** as "
            "`report_date`, so a 1-month buffer replaces the blanket 5-month "
            "lag. Universe membership is the S&P 500 list **as of each "
            "formation date** (historical constituents dataset), removing "
            "index-inclusion look-ahead; names whose price history has "
            "vanished from Yahoo (many delistings) still drop out — the "
            "residual survivorship is documented in the README. Formations "
            "run **July 2012 .. July 2025** — the window agreed with Japan so "
            "the two markets cover identical calendar time — and the "
            "evaluation window is capped at **2025-12-31** (the final "
            "formation contributes a half holding year). Covariances are "
            "estimated on 36 months of daily returns ending the day before "
            "formation."),
    },
    "japan": {
        "num": "04", "title": "Japan",
        "bench": {"1306.T (TOPIX ETF, JPY)": "1306.T",
                  "EWJV (MSCI Japan Value ETF, USD)": "EWJV"},
        "bench_note": ("1306.T is JPY like the portfolios; EWJV is USD, so its "
                       "row mixes in currency effects and is indicative only."),
        "ff": "japan", "usd_convert": True,
        "years": "[2023, 2024, 2025]", "lag": 3, "membership": False,
        "data_note": (
            "Yahoo Finance serves ~5 annual statement periods per name, which "
            "supports July-1 formations in **2023, 2024 and 2025** only; the "
            "evaluation window is capped at **2025-12-31** to stay inside the "
            "proposal's 2000–2025 sample (the final formation contributes a "
            "half holding year). "
            "Report dates are fiscal period ends; the reporting lag is 3 "
            "months — the statutory deadline for the securities report "
            "(yukashoken hokokusho), so a March fiscal year is public by "
            "end-June, just before formation. No free source provides 2000-2025 "
            "Japanese fundamentals (J-Quants free tier is 2 years; EDINET XBRL "
            "starts 2008 and needs the Japanese-GAAP taxonomy); extending "
            "Japan to the proposal window requires a licensed source "
            "(J-Quants paid tier / Refinitiv), which plugs into the same "
            "canonical schemas. Universe membership is current Nikkei 225 "
            "members — a survivorship caveat documented in "
            "`src/fscore/data/universe.py`."),
    },
}


def full_study_cells(m: dict) -> list:
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = []

    cells.append(md(f"""# {m['title']} — Full Piotroski F-Score Study (formations 2023–2025)

The proposal's complete loop on **real data**: point-in-time universe →
high-B/M value subset → 9-signal F-Score → fixed-basket selection (F-Score vs
value / market-cap / liquidity-matched / random Monte-Carlo) → EW / GMV /
sector-capped GMV on RMT-cleaned covariances, plus a dollar-neutral
long-short book (long top-k scores, short bottom-k) where shorting is
available → annual-rebalance backtest →
placement in the random distribution → investable benchmarks → Fama-French
three-factor regression.

**Data & scope.** {m['data_note']}

Run `python scripts/{'fetch_us_edgar.py' if m['membership'] else 'fetch_us_japan.py'}`
once before this notebook (builds the git-ignored cache under `data/`)."""))

    membership_load = ("""
from fscore.data.edgar import load_membership
membership = load_membership(ROOT / "data")""" if m["membership"] else """
membership = None""")

    cells.append(code(f"""import sys, pathlib
ROOT = pathlib.Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from fscore.plotting import setup_plots, save_fig
from fscore.data.yahoo import load_cached
from fscore.pipeline import run_study
from fscore.evaluation import (metrics, benchmark_returns, fetch_ff_factors,
                               factor_regression, to_usd)

setup_plots()      # study-wide figure defaults; every saved chart is 300 dpi
RESULTS_FIG = ROOT / "results" / "figures"
MARKET = "{'us' if m['membership'] else 'japan'}"
YEARS = {m['years']}
LAG_MONTHS = {m['lag']}  # {'report_date = true 10-K filing date' if m['lag'] == 1 else 'report_date = fiscal period end (no filing dates on Yahoo)'}
END_CAP = pd.Timestamp("2025-12-31")  # proposal sample ends in 2025
fund, prices, sectors, bench = load_cached(MARKET, ROOT / "data"){membership_load}
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

    cells.append(md("""**Data discarded before any test.** A firm-year is scored
only when all nine signals are computable from the fiscal T-1 and T-2
statements; partial scores are dropped rather than summed over whatever is
available (an incomplete score is not a low score). The per-formation count
is `dropped_incomplete_signals` in the diagnostics below."""))

    cells.append(code("""study = run_study(MARKET, fund, prices, sectors, YEARS,
                  n_mc=1000, n_mc_opt=300, lag_months=LAG_MONTHS,
                  membership=membership, end_cap=END_CAP, seed=42,
                  detone=False)   # RMT denoise only; see section 4b
diag = pd.DataFrame([{"year": yr.year, **yr.diagnostics} for yr in study.yearly])
diag.set_index("year")"""))

    cells.append(md("### 2. F-Score distribution within the value universe, per formation"))

    cells.append(code("""ncol = min(5, len(YEARS))
nrow = -(-len(YEARS) // ncol)
fig, axes = plt.subplots(nrow, ncol, figsize=(2.3 * ncol, 2.5 * nrow),
                         sharey=True, squeeze=False)
for ax, yr in zip(axes.flat, study.yearly):
    counts = yr.scored.fscore.value_counts().sort_index()
    ax.bar(counts.index, counts.values)
    ax.set_title(f"{yr.year}", fontsize=9); ax.set_xticks(range(0, 10, 3))
for ax in axes.flat[len(study.yearly):]:
    ax.axis("off")
fig.suptitle(f"{MARKET.upper()} — F-Score distribution within the high-B/M universe, by formation year")
plt.tight_layout(); save_fig(f"{MARKET}_fscore_distribution", directory=RESULTS_FIG); plt.show()

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
ax.legend(fontsize=8, ncol=2); plt.tight_layout()
save_fig(f"{{MARKET}}_nav_vs_benchmarks", directory=RESULTS_FIG); plt.show()"""))

    cells.append(code("""tbl = study.summary()      # incl. turnover / cost_drag / net_*
for name, r in bench_rets.items():
    tbl.loc[name] = metrics(r)   # benchmarks are buy-and-hold: no turnover
tbl[["ann_return", "ann_vol", "sharpe", "max_drawdown",
     "turnover", "cost_drag", "net_ann_return", "net_sharpe"]].round(3)"""))

    cells.append(md("""### 4. Statistical control — placement in the Monte-Carlo random distribution

Same universe, same basket size, same construction pipeline; the only
difference is *which* 30 names. Percentile = share of random baskets the
F-Score portfolio beats; the p-value is one-sided and judged at the study's
single significance level, **5%** (`significant` column). No 1% or 10% tier
is used, so p = 0.06 counts as not significant."""))

    cells.append(code("""pairs = [("EW", "fscore_EW"), ("GMV", "fscore_GMV"), ("GMVsec", "fscore_GMVsec")]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=False)
for ax, (how, strat) in zip(axes, pairs):
    sharpes = study.mc_summary(how)["sharpe"].dropna()
    fs = metrics(study.daily[strat].dropna())["sharpe"]
    ax.hist(sharpes, bins=30, alpha=0.75)
    ax.axvline(fs, color="crimson", lw=2)
    ax.set_title(f"{how}: F-Score={fs:.2f}", fontsize=10)
    ax.set_xlabel(f"Sharpe (chained {len(YEARS)}y)")
axes[0].set_ylabel("random baskets")
fig.suptitle(f"{MARKET.upper()} — F-Score basket vs random distribution, by construction")
plt.tight_layout(); save_fig(f"{MARKET}_mc_placement", directory=RESULTS_FIG); plt.show()

placements = pd.concat({strat: study.placement(strat, how) for how, strat in pairs})
placements.round(3)"""))

    if not m["usd_convert"]:   # US only — the RMT detoning diagnostic
        cells.append(md("""### 4b. RMT covariance: denoise only vs denoise + detone

The covariance fed to the minimum-variance solve is RMT-denoised
(Marchenko–Pastur noise band flattened). **Detoning** — additionally removing
the dominant market eigenmode — is off by default because it leaves the
matrix singular, so inverting it optimises residual risk only. This section
runs the detoned variant side by side to quantify what that choice costs."""))

        cells.append(code("""study_dt = run_study(MARKET, fund, prices, sectors, YEARS,
                     n_mc=1000, n_mc_opt=300, lag_months=LAG_MONTHS,
                     membership=membership, end_cap=END_CAP, seed=42,
                     detone=True)

def _diag(st, label):
    yr = st.yearly[0]
    w = yr.weights["fscore_GMV"]
    cov = clean_rmt(holding_returns(prices, list(w.index),
                                    formation_date(yr.year) - pd.DateOffset(years=1),
                                    formation_date(yr.year) - pd.Timedelta(days=1))[list(w.index)],
                    detone=(label == "denoise+detone"))
    ev = np.linalg.eigvalsh(cov)
    m_ = metrics(st.daily["fscore_GMV"].dropna())
    return {"construction": label,
            "min eigenvalue": ev.min(),
            "condition number": ev.max() / max(ev.min(), 1e-300),
            "max weight": w.max(),
            "effective N": 1 / (w ** 2).sum(),
            "predicted ann vol": np.sqrt(w.values @ cov @ w.values * 252),
            "realised ann vol": m_["ann_vol"],
            "ann return": m_["ann_return"],
            "sharpe": m_["sharpe"]}

from fscore.construction import clean_rmt
from fscore.pipeline import holding_returns, formation_date
cmp = pd.DataFrame([_diag(study, "denoise only"), _diag(study_dt, "denoise+detone")])
cmp.set_index("construction").T"""))

        cells.append(code("""rows = {}
for label, st in [("denoise only", study), ("denoise+detone", study_dt)]:
    for strat in ["fscore_GMV", "fscore_GMVsec"]:
        pl = st.placement(strat, "GMV" if strat == "fscore_GMV" else "GMVsec")
        rows[(label, strat)] = pl.loc["sharpe"]
detone_cmp = pd.DataFrame(rows).T
detone_cmp.round(3)"""))

        cells.append(code("""fig, ax = plt.subplots(figsize=(9, 4.5))
for label, st, ls in [("denoise only", study, "-"), ("denoise+detone", study_dt, "--")]:
    for strat, c in [("fscore_EW", "tab:blue"), ("fscore_GMV", "tab:red")]:
        nav_ = (1 + st.daily[strat].fillna(0)).cumprod()
        ax.plot(nav_.index, nav_, ls, color=c, lw=1.5, label=f"{strat} ({label})")
ax.set_yscale("log"); ax.set_ylabel("growth of 1 (log)")
ax.set_title(f"{MARKET.upper()} — effect of RMT detoning on the GMV track record")
ax.legend(fontsize=8); plt.tight_layout()
save_fig("us_detone_comparison", directory=RESULTS_FIG)
plt.show()"""))

    cells.append(md("""### 5. Turnover and implied trading cost

One-way turnover per rebalance, computed on each strategy's actual weights
(so GMV pays for weight drift, not just for name changes); cost drag =
2 x turnover x 20 bp per side, already applied in the `net_*` columns above.
For reference, the random control is redrawn every year and turns over
roughly `1 - k/|universe|`, i.e. in the same range as the F-Score basket."""))

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
size, value) with Newey-West standard errors. The `alpha_significant` column
is the verdict at the study's single level, 5%."""
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
