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
| `src/fscore/data/` | loaders and point-in-time alignment; `fs_clean.py` (statements → signals, the study's score panel), `edgar.py` (SEC filings), `yahoo.py` (prices), `universe.py` (constituents + sectors) |
| `src/fscore/signal/` | the nine Piotroski signals and composite score |
| `src/fscore/selection/` | fixed-size baskets: F-Score top-k, Monte-Carlo random, liquidity-matched, value, market-cap |
| `src/fscore/construction/` | EW / long-only GMV / sector-capped GMV (exact SLSQP solve); RMT covariance cleaning |
| `src/fscore/evaluation/` | backtest metrics, random-distribution placement; `benchmarks.py` (investable ETFs, Fama-French 3-factor regression) |
| `src/fscore/pipeline.py` | full multi-year study runner (PIT snapshot → universe → score → baskets → weights → chained backtest) |
| `src/fscore/markets.py` | per-market trading constraints (where a short leg is tradable) |
| `src/fscore/grid.py` | robustness grid runner (basket size × random-sample size × market) |
| `src/fscore/plotting.py` | figure defaults; every saved chart is 300 dpi |
| `scripts/` | data builders (`fetch_us_edgar.py`, `fetch_us_japan.py`), notebook generators, `export_panel.py` |
| `notebooks/` | `01`/`02` synthetic demos; `03`/`04` full studies; `grid/` 18 robustness notebooks |
| `data/` | git-ignored cache, rebuilt by the fetch scripts |
| `results/` | CSVs and 300-dpi figures behind every table and chart in the report |
| `tests/` | 19 tests, each pinning one design decision |

## Reproducing the results

Everything in the report comes from the notebooks in this repository. To
reproduce them from a clean clone:

**1. Install** (Python 3.11+):

```bash
pip install -r requirements.txt
```

**2. Check the install** — runs in seconds, no network, no data needed:

```bash
python tests/test_signals.py && python tests/test_pipeline.py
```

**3. Build the data caches.** They are git-ignored because they are large and
rebuildable; both scripts are resumable and safe to re-run.

```bash
python scripts/fetch_us_edgar.py
```

```bash
python scripts/fetch_us_japan.py
```

`fetch_us_edgar.py` pulls US fundamentals from SEC EDGAR (true 10-K filing
dates, FY2009+) and takes roughly an hour; `fetch_us_japan.py` pulls prices and
Japanese fundamentals from Yahoo Finance in about forty minutes. The signal
panel itself ships as a derived CSV — see `results/panel/PROVENANCE.md` for
what it contains and why the raw vendor fundamentals are not redistributed.

**4. Re-run the analysis.** Each command regenerates the notebooks from their
source templates and executes them, writing CSVs to `results/` and 300-dpi
figures to `results/figures/`:

```bash
python scripts/build_grid_notebooks.py execute
```

```bash
python scripts/build_notebooks.py
```

The grid takes about an hour (18 notebooks); the two main-study notebooks
about twenty minutes. Results are deterministic — every random draw is
seeded — so a clean run reproduces the reported figures exactly.

Notebooks `01`/`02` are self-contained demos on **clearly-labelled synthetic
data**; they need no network and are the fastest way to see the pipeline end
to end.

For a non-technical overview of what the system does and why, see
[SOLUTION_DESIGN.md](SOLUTION_DESIGN.md).

## Peer-review grid study (notebooks/grid/)

Following the M5 peer review, `src/fscore/grid.py` re-runs the developed pair
across the reviewer's grid, scoring the team's FS_clean statements with this
repository's own signal code (`fscore.signal.piotroski`) so that the grid and
the main study rest on one implementation — basket size k ∈ {20, 25, 30} × random draws N ∈ {1000, 2000, 5000} ×
{US, Japan} = **18 executed notebooks**, figures saved at dpi = 300.
Design answers the review's four priorities: explicit random basis (full
eligible universe, fresh yearly draws, overlap reported), the **direct synergy
test** D = Sharpe(GMV) − Sharpe(EW) per basket, a strict F ≥ 8 portfolio, and
one pre-registered primary measure (net-of-cost Sharpe) tested at one
pre-registered significance level (**5%**, `fscore.evaluation.ALPHA`).
Formations run
**July 2012 – July 2025** (14 chained holding years), the window shared with
the main study; covariances use 36 months of daily returns. Aggregated
outputs in `results/grid/` and `results/grid_summary_2012_2025.csv`;
build/execute with `python scripts/build_grid_notebooks.py execute`.

Headline: the percentile is insensitive to the number of random draws —
1,000 and 5,000 agree to within a percentage point — but **sensitive to
basket size**, which is itself a finding: k matters more than Monte-Carlo
precision, so the grid is read as a whole rather than at its best cell.

| Sharpe percentile vs random | k = 20 | k = 25 | k = 30 |
|---|---|---|---|
| US | 9% | 15% | 31% |
| Japan | 75% | 49% | 39% |

**No cell in either market clears 5%.** The US screen sits below the random
median at every basket size and below plain universe equal weight
(0.70–0.80 vs 0.85); Japan straddles the median (0.77–0.82 vs 0.80). The
optimisation gain D is negative throughout (−0.24 to −0.03), so under a
36-month covariance the minimum-variance step does not reward the F-Score
basket. The long-short book is negative in the US (−0.32) and barely
positive in Japan (+0.12).

## Headline results — main study

From the executed notebooks `03`/`04`; every verdict at the single
pre-registered level, **α = 5%**.

| | US (14 formations, Jul 2012 – Dec 2025) | Japan (3 formations, Jul 2023 – Dec 2025) |
|---|---|---|
| F-Score EW | 16.0% p.a., Sharpe 0.80 | 27.0% p.a., Sharpe 1.23 |
| vs random baskets | 48th pct, p = 0.52 — n.s. | 68th pct, p = 0.33 — n.s. |
| + GMV | 93rd pct, p = 0.067 — n.s. | 57th pct — n.s. |
| + sector-GMV | 97th pct, p = 0.027 — **significant** | 60th pct — n.s. |
| vs market ETF | 0.80 vs SPY 0.88 | 1.23 vs TOPIX 0.98 |
| vs pure value screen | 0.80 vs 0.85 | 1.23 vs 1.28 |
| Long-short (high − low) | Sharpe −0.13, −0.31 net | −1.11 |
| FF3 alpha | 1.1% p.a., t = 0.62 — n.s. | 7.2% p.a., t = 1.01 — n.s. |

**One test out of the twenty-odd run clears 5%** — the sector-capped GMV
placement in the US (p = 0.027). It should not be read as evidence that the
screen picks stocks, for three reasons: its equal-weight counterpart sits at
the 48th percentile, i.e. indistinguishable from random selection; the plain
GMV placement beside it does not clear (p = 0.067); and the grid study finds
the optimisation gain negative in all nine US cells. A single significant
cell among many tests is what multiple testing produces.

The substantive findings are the negative ones. After fourteen years the US
portfolio trails both SPY (0.80 vs 0.88) and the pure book-to-market screen
that precedes the nine signals (0.85), so the signals subtract from the
value sort rather than adding to it. Japan beats its index over three years
but on far too little data to conclude, and there too the value screen is
ahead. The long-short variant loses money in both markets.

### Measurement sensitivity

`scripts/eq_offer_sensitivity.py` scores the US sample under both
equity-issuance measures, since Japan can only use the share-count
substitute: they disagree on **38.2%** of firm-years (mean F-Score 5.59 vs
5.91, because buybacks mask issuance), yet move Sharpe by 0.013 and the
percentile by 6 points, leaving every verdict unchanged. That bounds what
the Japanese substitute can be costing.

### Cross-validation of the signal layer

Signals are computed by `fscore.signal.piotroski` from the team's FS_clean
statements. Scoring the same firm-years independently and comparing against
the collaborators' own implementation agrees on **100%** of US firm-years and
**99.8%** of Japanese ones (3,369 in total; the residue is four cases at the
share-count boundary). That comparison is what surfaced a defect in this
code: the year-on-year deltas had scaled the current year by average assets
and the prior year by period-end assets, putting a change of denominator
convention inside every difference. Agreement before the fix was 61% / 66%.

## Data status

| Market | Prices | Fundamentals | Universe membership | Backtest window |
|---|---|---|---|---|
| US (main study) | ✅ Yahoo (daily, 2002–) | ✅ SEC EDGAR XBRL (FY2009–, true 10-K filing dates, incl. equity-issuance cash flow) | ✅ historical S&P 500 members per formation date | **formations 2012–2025** (14 years) |
| Japan (main study) | ✅ Yahoo (daily, 2002–) | ⚠️ Yahoo (≈5 annual periods) | ⚠️ current Nikkei 225 members | formations 2023–2025 |
| US / Japan (grid) | ✅ Yahoo (daily, 2002–) | ✅ derived panel, `results/panel/` | ⚠️ currently listed symbols | **formations 2012–2025** (14 years) |
| Vietnam | TODO | TODO | TODO | team-maintained database, 10y |

**Why the sample starts in 2012.** EDGAR XBRL begins with the 2009–2011
mandate, so machine-readable point-in-time US statements start at FY2009,
and a 36-month covariance window needs three further years before the first
formation. For Japan, no free source serves long-history fundamentals:
J-Quants' free tier was measured at roughly two years of daily bars with no
statements and no historical listing snapshot (`scripts/check_jquants.py`
re-runs that check), and EDINET requires parsing two Japanese XBRL
taxonomies. 2012 is therefore the earliest start both markets share.

**Remaining caveats.** US: membership is point-in-time (historical
constituents list) and EDGAR keeps filings of delisted firms, but names whose
price history has vanished from Yahoo still drop out of the tradable
universe — a residual survivorship tilt. Japan: current-constituent universe,
~5 statement periods, period-end report dates with a conservative 5-month
lag (the two gating checks in `data/README.md` remain open for Japan).
