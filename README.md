# Piotroski F-Score Across Developed and Emerging Markets

MScFE Capstone — Quantitative Fundamentals (Topic 1)
**Team:** Zu Yao Teoh · Zhicheng Ding · Ta Tan Phat

We test whether the Piotroski F-Score is a practical stock-selection methodology
across two matched market pairs — **US & Japan (developed)** vs **Malaysia &
Vietnam (emerging)** — and whether quantitative portfolio construction
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

## Data status

| Market | Prices | Fundamentals | Universe membership | Backtest window |
|---|---|---|---|---|
| US | ✅ Yahoo (daily, 2008–) | ✅ SEC EDGAR XBRL (FY2009–, true 10-K filing dates) | ✅ historical S&P 500 members per formation date | **formations 2011–2025** (15 years) |
| Japan | ✅ Yahoo (daily, 2022–) | ✅ Yahoo (5 annual periods) | ⚠️ current Nikkei 225 members | formations 2023–2025 |
| Malaysia | TODO | TODO | TODO | 2019-12 – 2025-12 (current access) |
| Vietnam | TODO | TODO | TODO | ~2000–2025 |

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
