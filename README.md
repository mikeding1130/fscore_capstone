# Piotroski F-Score Across Developed and Emerging Markets

MScFE Capstone — Quantitative Fundamentals (Topic 1)
**Team:** Zu Yao Teoh · Zhicheng Ding · Ta Tan Phat

We test whether the Piotroski F-Score is a practical stock-selection methodology
across three markets — **US & Japan (developed)** vs **Vietnam (emerging)** —
and whether quantitative portfolio construction
(Equal-Weight, GMV, sector-constrained GMV, with RMT covariance cleaning)
improves the resulting portfolios. Following Piotroski (2000), the F-Score is
computed **within the high book-to-market (value) universe** only.

## Pipeline

```
universe (point-in-time, survivorship-safe, ~100-150 names)
   └─> high-B/M value subset                      src/fscore/data/
         └─> 9-signal F-Score (0–9)               src/fscore/signal/
               └─> SELECTION @ fixed basket       src/fscore/selection/
                   fscore | random-MC | liquidity | value | mkt-cap
                     └─> CONSTRUCTION              src/fscore/construction/
                         EW | GMV | sector-GMV  (+ RMT denoise/detone)
                           └─> BACKTEST + METRICS  src/fscore/evaluation/
                               vs random distribution (percentile, p-value)
                               vs investable benchmarks + FF3 regression
```

Selection and construction are deliberately **decoupled**: all selection
methods are first compared under equal weighting (isolating stock-picking),
then optimization is layered on the same baskets (isolating construction).
The multi-year loop (annual July-1 formations, chained holding years) lives in
`src/fscore/pipeline.py`.

## Repository layout

| Path | Contents |
|---|---|
| `src/fscore/data/` | loaders, point-in-time alignment, high-B/M subset; `universe.py` (S&P 500 / Nikkei 225 constituents + sectors), `yahoo.py` (real-data adapter + cache) |
| `src/fscore/signal/` | the nine Piotroski signals and composite score |
| `src/fscore/selection/` | fixed-size baskets: F-Score top-k, Monte-Carlo random, liquidity-matched, value, market-cap |
| `src/fscore/construction/` | EW / long-only GMV / sector-capped GMV (exact SLSQP solve); RMT covariance cleaning |
| `src/fscore/evaluation/` | backtest metrics, random-distribution placement; `benchmarks.py` (investable ETFs, Fama-French 3-factor regression) |
| `src/fscore/pipeline.py` | full multi-year study runner (PIT snapshot → universe → score → baskets → weights → chained backtest) |
| `scripts/` | `fetch_us_japan.py` (build the data cache), `build_notebooks.py` |
| `notebooks/` | `01`/`02` single-year synthetic demos (US, Japan); `03`/`04` **full real-data studies** (US, Japan) |
| `data/` | git-ignored cache; see `data/README.md` for schemas and gating checks |
| `results/` | CSV outputs of the full studies (summary, MC placement, turnover, factor regressions) |
| `tests/` | signal smoke test + construction/PIT unit tests |

## Quick start

```bash
pip install -r requirements.txt
python tests/test_signals.py && python tests/test_pipeline.py

# developed pair, real data (one-shot download, ~30 min):
python scripts/fetch_us_japan.py
jupyter lab notebooks/03_us_full_study.ipynb      # or 04_japan_full_study
```

Notebooks `01`/`02` run end-to-end on **clearly-labeled synthetic demo data**
(no network needed); `03`/`04` run the full study on the cached real data.

## Peer-review grid study (notebooks/grid/)

Following the M5 peer review, `src/fscore/grid.py` re-runs the developed pair
on the team-computed F-Score workbooks (`data/processed/*_Fscores_nonfinancial.xlsx`,
exact Piotroski conventions, financial firms removed) across the reviewer's
grid — basket size k ∈ {20, 25, 30} × random draws N ∈ {1000, 2000, 5000} ×
{US, Japan} = **18 executed notebooks**, figures saved at dpi = 300.
Design answers the review's four priorities: explicit random basis (full
eligible universe, fresh yearly draws, overlap reported), the **direct synergy
test** D = Sharpe(GMV) − Sharpe(EW) per basket, a strict F ≥ 8 portfolio, and
one pre-registered primary measure (net-of-cost Sharpe) tested at one
pre-registered significance level (**5%**, `fscore.evaluation.ALPHA`).
Formations run
**July 2003 – July 2025** (23 chained holding years; the workbooks' first
scoreable year is 2002, so 2000–2002 formations cannot exist). Aggregated
outputs in `results/grid/`; build/execute via
`python scripts/build_grid_notebooks.py execute`.

Headline (full window): percentiles are insensitive to N (1,000 draws
suffice). On this survivor-tilted blue-chip panel the screen alone does not
beat same-universe random selection (US ≈ 16th pct at k = 25; Japan ≈ 80th
pct, n.s.), and the optimisation gain D is statistically indistinguishable
from the random-basket distribution in both markets. This contrasts with the
high-B/M PIT design (main study: 93–97th pct under GMV) and the 2021–2025
sub-window (US D ≈ +0.12, p ≈ 0.06) — the selection × construction synergy
is **conditional** on universe design and estimation window, not universal.

## Headline results — developed pair

Full tables and charts live in notebooks `03`/`04` (executed) and `results/`.

All verdicts below are at the single pre-registered level, **α = 5%**.

| | US (15 formations, Jul 2011 – Dec 2025) | Japan (3 formations, Jul 2023 – Dec 2025) |
|---|---|---|
| F-Score EW vs random | 64–69th pct, p ≈ 0.31–0.36 — n.s. | 66–70th pct, p ≈ 0.30–0.34 — n.s. |
| F-Score + GMV vs random-GMV | 93–94th pct, p ≈ 0.06 — **n.s.** | 69th pct — n.s. |
| F-Score + sector-GMV | 97th pct, p ≈ 0.03 — **significant** | 71st pct — n.s. |
| vs market ETF | 17.0% vs 13.9% p.a. (GMV vs SPY) | 26.9% vs 20.0% p.a. (EW vs TOPIX) |
| FF3 alpha (best variant) | 2.9% p.a., t = 1.83, p ≈ 0.07 — n.s. | 7.0% p.a., t = 1.00 — n.s. |

Reading: selection alone is not significant in either market. Only one test
in the whole developed pair clears 5% — the sector-capped GMV placement in
the US (p ≈ 0.03) — and it sits alongside a GMV placement that does not
(p ≈ 0.06), so it is a single result, not a pattern. Note also that these
figures predate two corrections (RMT detoning switched off, per-strategy
trading costs); notebooks 03/04 need a re-run to refresh them.

## Data status

| Market | Prices | Fundamentals | Universe membership | Backtest window |
|---|---|---|---|---|
| US | ✅ Yahoo (daily, 2008–) | ✅ SEC EDGAR XBRL (FY2009–, true 10-K filing dates) | ✅ historical S&P 500 members per formation date | **formations 2011–2025** (15 years) |
| Japan | ✅ Yahoo (daily, 2022–) | ✅ Yahoo (5 annual periods) | ⚠️ current Nikkei 225 members | formations 2023–2025 |
| Vietnam | TODO | TODO | TODO | team-maintained database, 10y |

**Why the proposal's 2000 start is not fully reachable from free sources.**
US: EDGAR XBRL begins with the 2009–2011 mandate, so machine-readable
point-in-time statements start at FY2009 (pre-2009 requires CRSP/Compustat).
Japan: no free source serves long-history fundamentals at all (J-Quants free
tier is 2 years; EDINET XBRL starts 2008 and needs the Japanese-GAAP
taxonomy) — extending Japan needs a licensed source (J-Quants paid tier /
Refinitiv). Both plug into the same canonical schemas without touching
downstream code.

**Remaining caveats.** US: membership is point-in-time (historical
constituents list) and EDGAR keeps filings of delisted firms, but names whose
price history has vanished from Yahoo still drop out of the tradable
universe — a residual survivorship tilt. Japan: current-constituent universe,
~5 statement periods, period-end report dates with a conservative 5-month
lag (the two gating checks in `data/README.md` remain open for Japan).
