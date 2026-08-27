"""Generate the Japan single-year demo (02) and the full-study notebooks
(03 US, 04 Japan, 05 Vietnam). Idempotent; execution happens separately.

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
                    # the figure name is derived from the market too: without
                    # this the Japan demo overwrites the US demo's PNG
                    .replace("demo_us_", "demo_japan_")
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
        "years": "list(range(2012, 2025))", "lag": 1, "membership": True,
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
            "run **July 2012 .. July 2024** — thirteen chained holding years, "
            "each a full twelve months. July 2024 is the last formation whose "
            "complete year finishes inside the sample; taking July 2025 as "
            "well would mix a half-year window in with the complete ones. "
            "Covariances are estimated on 36 months of daily returns ending "
            "the day before formation."),
    },
    "japan": {
        "num": "04", "title": "Japan",
        "bench": {"1306.T (TOPIX ETF, JPY)": "1306.T",
                  "EWJV (MSCI Japan Value ETF, USD)": "EWJV"},
        "bench_note": ("1306.T is JPY like the portfolios; EWJV is USD, so its "
                       "row mixes in currency effects and is indicative only."),
        "ff": "japan", "usd_convert": True,
        "years": "list(range(2012, 2025))", "lag": 3, "membership": True,
        "loader": """from fscore.data.bbg_processed import constituents
fund = pd.read_csv(ROOT / "data" / "japan_bbg_fundamentals.csv",
                   parse_dates=["report_date"])
prices = pd.read_csv(ROOT / "data" / "japan_prices.csv.gz", parse_dates=["date"])
sectors = pd.read_csv(ROOT / "data" / "japan_sectors.csv").set_index("ticker")["sector"]
bench = pd.read_csv(ROOT / "data" / "japan_benchmarks.csv.gz", parse_dates=["date"])
membership = constituents(MARKET, ROOT / "data", YEARS)""",
        "data_note": (
            "Bloomberg statements from `data/processed/Japan/`, which carry "
            "fiscal years back to 2000 and so lift Japan onto the **same "
            "July 2012 – July 2024 window as the US** — thirteen chained "
            "holding years, replacing the two formations the old Yahoo cache "
            "allowed. Book equity comes from `Common_Shareholders_Equity`; "
            "the vendor's own `Book_Value`, `Historical_Market_Cap` and "
            "`Proceeds_Issuance_Common_Stock` columns are **empty in every "
            "sheet**, so market cap is rebuilt as raw close × shares "
            "outstanding and EQ_OFFER falls back to the share count — the "
            "generous measure (see `results/eq_offer_sensitivity.csv`). "
            "The universe is the **TPX100** constituent list as of each "
            "formation date, so index-inclusion look-ahead is removed — but "
            "it is only ~100 names, and the high-B/M subset is correspondingly "
            "small: a 30-name basket is 83–91% of it, which leaves the "
            "random-basket comparison little room to separate anything. The "
            "grid's **k = 20** cell is the interpretable one for Japan; k = 30 "
            "is kept here for continuity with the US. "
            "Report dates are fiscal period ends; the reporting lag is 3 "
            "months — the statutory deadline for the securities report "
            "(yukashoken hokokusho), so a March fiscal year is public by "
            "end-June, just before formation. "
            "**Prices are not from this vendor tree**: its price workbook is "
            "empty in both markets (its own `price_coverage.xlsx` records "
            "`Has_Any_Adjusted_Price = False` for every name), so prices come "
            "from the Yahoo cache. That covers 97–100% of each formation's "
            "constituents; the names it cannot serve — Toshiba, Bank of "
            "Yokohama, NTT Docomo — are delisted or merged, which is the "
            "residual survivorship bias stated plainly."),
    },
    "vietnam": {
        "num": "05", "title": "Vietnam",
        "bench": {"VN30 (VN30 index, VND)": "VN30",
                  "VNINDEX (all-share index, VND)": "VNINDEX"},
        "bench_note": (
            "Both indices are VND, like the portfolios — no currency effect "
            "enters this row. They are, however, **capital indices**: neither "
            "reinvests cash dividends, while the portfolios are built on "
            "dividend-adjusted closes. The gap is the index dividend yield, "
            "historically around 1.5-2% a year, and it flatters every "
            "portfolio against these two rows by that much. No total-return "
            "version of either index exists in the source database, so this "
            "is disclosed rather than corrected."),
        "ff": "vietnam", "usd_convert": False,
        "crosscheck_panel": True,
        "years": "list(range(2012, 2025))", "lag": 6, "membership": False,
        "data_note": (
            "Statements come from the team's own preprocessing repository "
            "(`../thesis`), which crawls FireAnt, CafeF and TCBS into "
            "`fscore.db`, reconciles the three, applies accounting checks and "
            "writes a per-firm-year panel — FY2009 onward, the span the "
            "Vietnamese sources cover. `scripts/build_vietnam_data.py` maps "
            "that panel into this study's canonical schema. `report_date` is "
            "the 31 December fiscal year end (the panel carries no filing "
            "date), and the reporting lag is **6 months**: 31 December + 6 "
            "months = 30 June, the last day before a 1 July formation. That "
            "is the most conservative rule that still admits the prior fiscal "
            "year, and it is the same screening date the sibling pipeline "
            "uses, so both branches select on the same information. "
            "Formations run **July 2012 .. July 2024** — thirteen chained "
            "holding years, the same calendar as the US and Japan. The price "
            "cache reaches August 2026 and would support a July 2025 "
            "formation, but taking it would move the goalposts relative to "
            "the other two markets. Survivorship is **partial rather than "
            "total**: the price panel does retain lines that died inside the "
            "sample — 125 of its 1,371 tickers print for the last time before "
            "2026, spread across 2012-2025 — but names the vendor no longer "
            "resolves at all leave no trace to count, so the residual is "
            "unquantified. Covariances are estimated on 36 months of daily "
            "returns ending the day before formation."),
    },
}


def full_study_cells(m: dict) -> list:
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = []

    span = (m['years'].replace('list(range(', '').replace('))', '')
            .replace(', ', '–') if 'range' in m['years'] else m['years'])
    cells.append(md(f"""# {m['title']} — Full Piotroski F-Score Study

The proposal's complete loop on **real data**: point-in-time universe →
high-B/M value subset → 9-signal F-Score → fixed-basket selection (F-Score vs
value / market-cap / liquidity-matched / random Monte-Carlo) → EW / GMV /
sector-capped GMV on RMT-cleaned covariances, plus a dollar-neutral
long-short book (long top-k scores, short bottom-k) where shorting is
available → annual-rebalance backtest →
placement in the random distribution → investable benchmarks → Fama-French
three-factor regression.

**Data & scope.** {m['data_note']}

Run `python scripts/{m['fetch_script']}`
once before this notebook (builds the git-ignored cache under `data/`)."""))

    membership_load = ("""
from fscore.data.edgar import load_membership
membership = load_membership(ROOT / "data")""" if m["membership"] else """
membership = None""")
    # Most markets load through the shared Yahoo-cache reader; Japan's
    # statements come from the Bloomberg tree instead, so it overrides.
    data_load = m.get("loader") or (
        'fund, prices, sectors, bench = load_cached(MARKET, ROOT / "data")'
        + membership_load)

    cells.append(code(f"""import sys, pathlib
ROOT = pathlib.Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from fscore.plotting import setup_plots, save_fig
from fscore.data.yahoo import load_cached
from fscore.pipeline import run_study
from fscore.evaluation import (metrics, benchmark_returns, fetch_ff_factors,
                               factor_regression, to_usd){m['extra_imports']}

setup_plots()      # study-wide figure defaults; every saved chart is 300 dpi
RESULTS_FIG = ROOT / "results" / "figures"
MARKET = "{m['market']}"
YEARS = {m['years']}
LAG_MONTHS = {m['lag']}  # {m['lag_note']}
END_CAP = None   # every holding year is complete; nothing is truncated
{data_load}
print(f"fundamentals: {{fund.ticker.nunique()}} tickers, "
      f"FY{{fund.fiscal_year.min()}}–FY{{fund.fiscal_year.max()}}")
print(f"prices: {{prices.ticker.nunique()}} tickers, "
      f"{{prices.date.min():%Y-%m-%d}} → {{prices.date.max():%Y-%m-%d}}")
fund.groupby("fiscal_year").size().rename("statements")"""))

    if m.get("crosscheck_panel"):
        # Vietnam is the one market whose signals arrive already computed, by
        # a different codebase. Recomputing them here from the canonical
        # statement lines is what lets its results sit beside the other two.
        cells.append(md("""### 0. Cross-validation of the signal layer

The Vietnamese nine signals arrive from the team's preprocessing repository
already computed. The US and Japanese ones are computed by
`fscore.signal.piotroski` in this repository. Putting the three markets in one
table is only legitimate if they rest on the same scoring rules, so this
section recomputes every Vietnamese firm-year with *this* repository's signal
code, from the canonical statement lines `scripts/build_vietnam_data.py`
writes, and compares flag by flag against the shipped panel.

Two conventions had to be matched for this to be a real test rather than a
tautology: `net_income` is the parent-company share, and `cogs` is derived as
`net_sales - gross_profit` so that `(revenue - cogs) / revenue` here is the
same quantity as the `gross_profit / net_sales` margin the sibling pipeline
scored. Everything else — beginning-of-year asset scaling on both sides of
every delta, the cash-flow equity-issuance measure, dropping firm-years whose
nine signals are not all computable — is this repository's own rule set."""))

        cells.append(code("""from fscore.data.fs_clean import score_panel_path
from fscore.signal.piotroski import piotroski_signals

shipped = pd.read_csv(score_panel_path(MARKET, ROOT / "data"))
rows = []
for y in range(int(shipped.score_year.min()), int(shipped.score_year.max()) + 1):
    snap = fund[fund.fiscal_year.isin([y, y - 1, y - 2])]
    scored_y = piotroski_signals(snap, year=y)
    if len(scored_y):
        rows.append(scored_y.assign(score_year=y))
ours = pd.concat(rows, ignore_index=True)

PAIRS = [("roa_pos", "f_roa"), ("cfo_pos", "f_cfo"), ("delta_roa_pos", "f_droa"),
         ("accruals_ok", "f_accrual"), ("delta_leverage_down", "f_dlever"),
         ("delta_liquidity_up", "f_dliquid"), ("no_issuance", "f_eq_offer"),
         ("delta_margin_up", "f_dmargin"), ("delta_turnover_up", "f_dturn")]
both = ours.merge(shipped, on=["ticker", "score_year"], suffixes=("_ours", "_theirs"))
agree = pd.Series({"composite F-Score": (both.fscore_ours == both.fscore_theirs).mean(),
                   **{ours_c: (both[ours_c] == both[theirs_c]).mean()
                      for ours_c, theirs_c in PAIRS}}, name="agreement")
print(f"{len(both):,} of {len(shipped):,} shipped firm-years matched on (ticker, score_year)")
print(f"mean absolute F-Score difference: {(both.fscore_ours - both.fscore_theirs).abs().mean():.4f}")
(100 * agree).round(2).to_frame("% agreement")"""))

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
                  detone=False)   # RMT denoise only — detoning is
                                  # out of scope for this study
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
# solid for the F-Score variants, dashed for the controls; a strategy not
# named here still plots, so adding one never breaks the figure
styles = {{"fscore_EW": ("-", 2.2), "fscore_GMV": ("-", 1.4), "fscore_GMVsec": ("-", 1.4),
          "fscore_LS": ("-", 1.4),
          "value_EW": ("--", 1.2), "mktcap_EW": ("--", 1.2), "liquidity_EW": ("--", 1.2)}}
for s in study.daily.columns:
    ls, lw = styles.get(s, ("-" if s.startswith("fscore") else "-.", 1.2))
    ax.plot(nav.index, nav[s], ls, lw=lw, label=s)
for name, r in bench_rets.items():
    ax.plot((1 + r).cumprod(), ":", lw=1.6, label=name)
ax.set_ylabel("growth of 1 (log)"); ax.set_yscale("log")
ax.set_title(f"{{MARKET.upper()}} — F-Score strategies vs controls and benchmarks, "
             f"{{start:%b %Y}} – {{end:%b %Y}}")
ax.legend(fontsize=8, ncol=2); plt.tight_layout()
save_fig(f"{{MARKET}}_nav_vs_benchmarks", directory=RESULTS_FIG); plt.show()"""))

    cells.append(code("""tbl = study.summary()      # gross first; costs follow separately
for name, r in bench_rets.items():
    tbl.loc[name] = metrics(r)   # benchmarks are buy-and-hold: no turnover
tbl[["ann_return", "ann_vol", "sharpe", "max_drawdown",
     "nominal_k", "effective_n"]].round(3)"""))

    cells.append(md("""`nominal_k` is the basket size, `effective_n` is 1/Σw² —
an optimised or sector-capped book concentrates, so the two are not the same
number of holdings. Turnover and the net-of-cost sensitivity follow below."""))

    cells.append(code("""tbl[["turnover", "cost_drag",
     "net_ann_return", "net_sharpe"]].round(4)"""))

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

    if m["ff"] == "vietnam":
        # Ken French covers the US, Japan and the developed regions — not
        # Vietnam, and no free substitute does. The factors are built from
        # this study's own panel instead; see `local_ff3_factors`.
        ff_lines = ('ff = local_ff3_factors(fund, prices, YEARS, '
                    'lag_months=LAG_MONTHS)\n'
                    'series = {s: study.daily[s].dropna()\n'
                    '          for s in ["fscore_EW", "fscore_GMV", "value_EW"]}')
        cells.append(md("""### 6. Does alpha survive the factor exposures?

Daily excess returns regressed on three factors (market, size, value) with
Newey-West standard errors. The `alpha_significant` column is the verdict at
the study's single level, 5%.

**The factors are local, not Ken French's.** The library covers the US, Japan
and the developed regions; it does not cover Vietnam, and no free substitute
does. Skipping the regression would leave the study's only emerging market
without the one test that strips style exposure out of the result, so the
factors are built here from the same point-in-time panel the study already
uses — a 2x3 size / book-to-market sort, formed 1 July on fiscal T-1
statements, value-weighted and held for twelve months
(`fscore.evaluation.local_ff3_factors`). Two consequences: the market leg is
this panel's eligible universe rather than the whole exchange, and market cap
is measured at the fiscal year end rather than at formation. The alphas below
are therefore alphas against a *local approximation* of the three factors,
and are not comparable coefficient-for-coefficient with the US and Japan
tables above."""))
        cells.append(code(f"""{ff_lines}
reg = pd.DataFrame({{s: factor_regression(r, ff) for s, r in series.items()}}).T
reg.round(4)"""))
        _skip_ff = True
    else:
        _skip_ff = False
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

    if not _skip_ff:
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


# Fields derived from the config rather than repeated in it: the cache-building
# script, the market key the notebook passes to `load_cached`, and the one-line
# explanation of what `report_date` means for that market.
FETCH_SCRIPT = {"us": "fetch_us_edgar.py", "japan": "build_japan_bbg.py",
                "vietnam": "build_vietnam_data.py"}
EXTRA_IMPORTS = {
    "us": "", "japan": "",
    # Ken French has no Vietnamese factor set; the notebook builds one
    "vietnam": "\nfrom fscore.evaluation import local_ff3_factors",
}
LAG_NOTE = {
    "us": "report_date = true 10-K filing date",
    "japan": "report_date = fiscal period end; the lag is applied on top",
    "vietnam": "report_date = 31 Dec fiscal year end; +6m = 30 Jun, the day before formation",
}


def build_full(market: str):
    m = dict(CFG[market], market=market,
             fetch_script=FETCH_SCRIPT[market], lag_note=LAG_NOTE[market],
             extra_imports=EXTRA_IMPORTS[market])
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
    build_full("vietnam")
