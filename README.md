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
universe (point-in-time, survivorship-safe, ~100 names)
   └─> high-B/M value subset                      src/fscore/data/
         └─> 9-signal F-Score (0–9)               src/fscore/signal/
               └─> SELECTION @ fixed basket       src/fscore/selection/
                   fscore | random-MC | liquidity | value | mkt-cap
                     └─> CONSTRUCTION              src/fscore/construction/
                         EW | GMV | sector-GMV  (+ RMT denoise/detone)
                           └─> BACKTEST + METRICS  src/fscore/evaluation/
                               vs random distribution (percentile, p-value)
```

Selection and construction are deliberately **decoupled**: all selection
methods are first compared under equal weighting (isolating stock-picking),
then optimization is layered on the same baskets (isolating construction).

## Repository layout

| Path | Contents |
|---|---|
| `src/fscore/data/` | loaders, point-in-time alignment, universe construction, high-B/M subset |
| `src/fscore/signal/` | the nine Piotroski signals and composite score |
| `src/fscore/selection/` | fixed-size baskets: F-Score top-k, Monte-Carlo random, liquidity-matched, value, market-cap |
| `src/fscore/construction/` | EW / GMV / sector-constrained GMV weights; RMT covariance cleaning |
| `src/fscore/evaluation/` | annual-rebalance backtest, performance metrics, random-distribution comparison |
| `notebooks/` | `01_us_fscore_single_year` and `02_japan_fscore_single_year` — runnable demos |
| `data/` | git-ignored; see `data/README.md` for expected file schemas |
| `tests/` | smoke tests for the signal layer |

## Quick start

```bash
pip install -r requirements.txt
jupyter lab notebooks/01_us_fscore_single_year.ipynb
```

Both notebooks run end-to-end on **clearly-labeled synthetic demo data** so the
pipeline is testable before real data lands; swap in real fundamentals/prices
via the loaders in `src/fscore/data/loaders.py` (interfaces documented there).

## Data status (gating item)

| Market | Prices | Fundamentals | Delisted coverage | Window |
|---|---|---|---|---|
| US | TODO | TODO | TODO | ~2000–2025 |
| Japan | TODO | TODO | TODO | ~2000–2025 |
| Malaysia | TODO | TODO | TODO | 2019-12 – 2025-12 (current access) |
| Vietnam | team DB ✅ | team DB ✅ | verify | per DB |

## References

Piotroski (2000); Markowitz (1952); Laloux et al. (1999); Marchenko–Pastur
(1967); López de Prado (2020); DeMiguel, Garlappi & Uppal (2009);
Fama & French (1992, 1993). Full MLA bibliography in the report.
