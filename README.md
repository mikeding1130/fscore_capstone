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
                         EW | GMV | sector-GMV  (+ RMT denoise)
                           └─> BACKTEST + METRICS  src/fscore/evaluation/
                               vs random distribution (percentile, p-value)
                               vs investable benchmarks + FF3 regression
```

Selection and construction are deliberately **decoupled**: all selection
methods are first compared under equal weighting (isolating stock-picking),
then optimization is layered on the same baskets (isolating construction).
The multi-year loop (annual July-1 formations, chained holding years) lives in
`src/fscore/pipeline.py`.

**Rebalancing is annual, and the cost model matches it.** A book is bought at
formation and left to drift for twelve months; the only trade is at the next
formation, and turnover is measured from the weights the book has drifted to
by then. Applying a fixed weight vector to daily returns — the obvious
shortcut — silently rebalances every day instead: on the 2023 US basket that
implies 1.19 of one-way turnover over the year against 0.11 for buy-and-hold,
an order of magnitude of trading that no cost was ever charged for.

## Repository layout

| Path | Contents |
|---|---|
| `src/fscore/data/` | loaders and point-in-time alignment; `fs_clean.py` (statements → signals, the study's score panel), `edgar.py` (SEC filings), `yahoo.py` (prices), `vietnam.py` (the team pipeline → canonical frames, VN30/VNINDEX), `universe.py` (constituents + sectors) |
| `src/fscore/signal/` | the nine Piotroski signals and composite score |
| `src/fscore/selection/` | fixed-size baskets: F-Score top-k, Monte-Carlo random, liquidity-matched, value, market-cap |
| `src/fscore/construction/` | EW / long-only GMV / sector-capped GMV (exact SLSQP solve); RMT covariance cleaning |
| `src/fscore/evaluation/` | backtest metrics, random-distribution placement; `benchmarks.py` (investable ETFs, Fama-French 3-factor regression, and a locally built factor set for markets Ken French does not cover) |
| `src/fscore/pipeline.py` | full multi-year study runner (PIT snapshot → universe → score → baskets → weights → chained backtest) |
| `src/fscore/markets.py` | per-market trading constraints (where a short leg is tradable) |
| `src/fscore/grid.py` | robustness grid runner (basket size × random-sample size × market) |
| `src/fscore/plotting.py` | figure defaults; every saved chart is 300 dpi |
| `src/fscore_vietnam/` | the Vietnamese pipeline: crawl → checked, scored panel, plus `schema_adapter.py` mapping it into the canonical schemas. Self-contained; see its README |
| `scripts/` | data builders (`fetch_us_edgar.py`, `fetch_us_japan.py`), notebook generators, `export_panel.py`, `tie_break_sensitivity.py`, `reconcile_report.py` |
| `notebooks/` | `01`/`02` synthetic demos; `03`/`04`/`05` full studies (US / Japan / Vietnam); `grid/` three robustness notebooks, nine cells each |
| `data/` | git-ignored cache, rebuilt by the fetch scripts |
| `results/` | CSVs and 300-dpi figures behind every table and chart in the report |
| `tests/` | 22 tests, each pinning one design decision |

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

```bash
python src/fscore_vietnam/schema_adapter_util.py
```

`fetch_us_edgar.py` pulls US fundamentals from SEC EDGAR (true 10-K filing
dates, FY2009+) and takes roughly an hour; `fetch_us_japan.py` pulls prices and
Japanese fundamentals from Yahoo Finance in about forty minutes. The signal
panel itself ships as a derived CSV — see `results/panel/PROVENANCE.md` for
what it contains and why the raw vendor fundamentals are not redistributed.

Vietnam has no vendor API here. Its pipeline lives in this repository, at
`src/fscore_vietnam` — it crawls FireAnt, CafeF and TCBS into `fscore.db`,
reconciles them, checks the accounting, scores the nine signals and writes the
panels. `schema_adapter_util.py` in that same package maps the checked panel
into this study's canonical schemas and pulls the VN30 and VNINDEX levels for
the benchmark comparison. It runs in
seconds and needs no network — see `data/README.md` for the three conventions
that differ from the developed pair.

**4. Re-run the analysis.** Each command regenerates the notebooks from their
source templates and executes them, writing CSVs to `results/` and 300-dpi
figures to `results/figures/`:

```bash
python scripts/build_grid_notebooks.py execute
```

```bash
python scripts/build_notebooks.py
```

The grid takes a little over an hour (three notebooks, nine cells each); the
three main-study notebooks about twenty-five minutes. Results are
deterministic — every random draw is seeded — so a clean run reproduces the
reported figures exactly. `python scripts/build_grid_notebooks.py execute vietnam`
runs one market's grid on its own.

Notebooks `01`/`02` are self-contained demos on **clearly-labelled synthetic
data**; they need no network and are the fastest way to see the pipeline end
to end.

For a non-technical overview of what the system does and why, see
[SOLUTION_DESIGN.md](SOLUTION_DESIGN.md).

## Peer-review grid study (notebooks/grid/)

Following the M5 peer review, `src/fscore/grid.py` re-runs every market across
the reviewer's grid, scoring the statement panels with this repository's own
signal code (`fscore.signal.piotroski`) so that the grid and the main study
rest on one implementation — basket size k ∈ {20, 25, 30} × random draws
N ∈ {1000, 2000, 5000} × {US, Japan, Vietnam} = **27 grid cells**, figures
saved at dpi = 300. The cells run as a sweep inside **three notebooks**,
`us_grid.ipynb`, `japan_grid.ipynb` and `vietnam_grid.ipynb`, rather than the
separate files they started as; each cell still writes its own
`{market}_k{k}_mc{N}_*` outputs, and the merge was checked value-by-value
against the pre-merge results (`scripts/verify_grid_merge.py`, 108/108 CSVs
identical for the developed pair).
Design answers the review's four priorities: explicit random basis (full
eligible universe, fresh yearly draws, overlap reported), the **direct synergy
test** D = Sharpe(GMV) − Sharpe(EW) per basket, a strict F ≥ 8 portfolio, and
one pre-registered primary measure (the gross Sharpe ratio, with turnover
reported separately) tested at one pre-registered significance level (**5%**,
`fscore.evaluation.ALPHA`). Formations run **July 2012 – July 2024** —
thirteen chained holding years, each a full twelve months, the window shared
with the main study; covariances use 36 months of daily returns. Aggregated
outputs in `results/grid/`, consolidated per market in
`results/grid/{market}_grid_summary.csv` (the notebooks now build this table
themselves; it reproduces the hand-assembled
`results/grid_summary_2012_2024.csv` exactly);
build/execute with `python scripts/build_grid_notebooks.py execute`.

Headline: the percentile is insensitive to the number of random draws —
1,000 and 5,000 agree to within two percentage points — but **sensitive to
basket size**, which is itself a finding: k matters more than Monte-Carlo
precision, so the grid is read whole rather than at its best cell. The
consolidated table makes the reason plain: within a given k, every column but
the percentile and its p-value is *bit-identical* across N, because the same
seed draws the same first 1,000 baskets whether the run asks for 1,000 or
5,000. N buys resolution on the p-value, nothing else — the grid is three
independent settings tested at three precisions, not nine settings.

| Sharpe percentile vs random (N = 5000) | k = 20 | k = 25 | k = 30 |
|---|---|---|---|
| US | 16% | 19% | 36% |
| Japan | 74% | 53% | 46% |
| Vietnam | **98%** | **99%** | 90% |

**No cell in the developed pair clears 5%; six of Vietnam's nine do.** The US
screen sits below the random median at every basket size, below plain universe
equal weight (0.73–0.81 vs 0.86) and below the value screen alone (0.81);
Japan straddles the median (0.70–0.74 vs 0.72). Vietnam clears 5% at k = 20
(p = 0.016) and k = 25 (p = 0.015) at every N, and misses at k = 30
(p = 0.105) — so the emerging-market result is real against a random basket
but **not stable across basket size**, exactly the sensitivity the grid was
built to expose.

**The Vietnamese edge does not survive the simplest control.** Its universe
equal-weight portfolio — no screen at all, just hold everything — returns
Sharpe **1.285**, against the F-Score basket's 1.286 / 1.318 / 1.184 at
k = 20 / 25 / 30. So the F-Score basket beats a *random 25-name basket*
comfortably and the *whole universe held equally* not at all. Whole-universe
minimum variance does better still (2.49), on a covariance whose conditioning
is far weaker than the developed markets' — see `REVIEW_FINDINGS.md` §B6.

**The optimisation gain D is not significant anywhere.** It is negative almost
throughout the developed pair (−0.22 to +0.01) and positive throughout Vietnam
(+0.23 to +0.45), but no cell in any market reaches p < 0.05 (best: Vietnam
k = 25, p = 0.14). Whatever minimum variance does to an F-Score basket, it does
about as much to a random one.

**And it does not survive the tie-break.** The F-Score is an integer, so the
top-k cut rarely falls cleanly; the remaining slots go to a seeded random draw
among firms sharing the cut-off score — on Vietnam at k = 25 that is 14 of 25
slots on average, and all 25 in two of the thirteen formation years.
`scripts/tie_break_sensitivity.py` re-runs the cell at eight seeds
(`results/vietnam_tiebreak_sensitivity.csv`): Sharpe spans 1.211–1.356 and the
p-value 0.010–0.059, with **six of eight seeds clearing 5% and two not**. The
seed the study uses, 42, produces the lowest p-value of the eight. The
synergy statistic is equally seed-dependent in level (D 0.170–0.451) but not
in verdict — its p-value never falls below 0.14. Quote the seed distribution,
not one draw.

**Basket size is not the number of holdings.** At k = 25 the equal-weight
book holds 25 names by construction, but the minimum-variance book has an
effective N (1/Σw²) of **5.3 (Japan), 6.6 (US), 11.5 (Vietnam)**. The optimiser
concentrates into a handful of names; k is an upper bound, and the two
quantities sit side by side in every summary table.

## Headline results — main study

From the executed notebooks `03`/`04`. The headline convention is **gross**
performance (cost models differ by market and would confound a cross-country
comparison); turnover and net-of-cost figures are reported separately.
Every verdict is at the single pre-registered level, **α = 5%**.

| | US (13 formations, Jul 2012 – Jun 2025) | Japan (2 formations, Jul 2023 – Jun 2025) | Vietnam (13 formations, Jul 2012 – Jun 2025) |
|---|---|---|---|
| F-Score EW | 15.9% p.a., Sharpe 0.81 | 18.0% p.a., Sharpe 0.77 | 11.4% p.a., Sharpe 0.44 |
| vs random baskets | 47th pct, p = 0.53 — n.s. | 75th pct, p = 0.25 — n.s. | 92nd pct, p = 0.085 — n.s. |
| + GMV | 97th pct, p = 0.030 — **significant** | 67th pct — n.s. | 95th pct, p = 0.050 — n.s. |
| + sector-GMV | 99.7th pct, p = 0.003 — **significant** | 81st pct — n.s. | 55th pct, p = 0.45 — n.s. |
| vs market index | 0.81 vs SPY 0.85 | 0.77 vs TOPIX 0.65 | 0.44 vs VN30 0.47, VNINDEX 0.54 |
| vs pure value screen | 0.81 vs 0.86 | 0.77 vs 0.76 | 0.44 vs 0.20 |
| Long-short (high − low) | Sharpe −0.11 | −1.32 | not tradable (long-only market) |
| Effective N (k = 30) | EW 30, GMV 5.6, sector-GMV 8.8 | — | EW 30, GMV 7.0, sector-GMV 8.9 |
| Turnover (EW) | 0.61 one-way per year | 0.97 | 0.70 |

**Selection does nothing; the two significant results come from the
optimiser, and they concentrate.** The equal-weight F-Score basket sits at
the 47th percentile — indistinguishable from drawing 30 names at random from
the same universe — and trails both SPY and the plain book-to-market screen
that precedes the nine signals. What clears 5% is GMV (p = 0.030) and
sector-capped GMV (p = 0.003) applied to that same basket. Two cautions
belong with those numbers: the grid study, which ranks inside the full
universe rather than a value subset, finds the optimisation gain **negative**
in all nine US cells; and the winning books hold an effective 5.6 and 8.8
names against a nominal 30, so their Sharpe is earned by concentration, not
by the screen.

Japan clears nothing. Its main study now spans two formations — `FS_clean`
carries no equity line, so the high-B/M universe depends on the fundamentals
cache, which for Japan begins at FY2021 — and the 2023 formation scores only
nine names. Those figures are reported for completeness, not as evidence.

Vietnam clears nothing either, **in this experiment family**, and this is the
one place where the two families must not be confused. Inside the high-B/M
subset of the 150 most liquid names, the F-Score basket returns 11.4% a year
at Sharpe 0.44 — below both the VN30 (0.47) and the all-share VNINDEX (0.54),
though comfortably above the pure book-to-market screen (0.20), which in this
market is where the damage is: Vietnamese deep value returned 5.7% a year with
a 77% drawdown. The *grid* study, which ranks the same score across the whole
scoreable universe instead of a value subset, reaches Sharpe 1.32 and p = 0.015
at k = 25. Both numbers are correct; they answer different questions, and the
report has to name which. Two further cautions belong with the Vietnamese
figures: one extra formation year moves the main-study p-value from 0.085 to
0.042 (`results/robustness_full_period.csv`), and the benchmark rows are
capital indices while the portfolios are total-return. `REVIEW_FINDINGS.md`
lists both, and the rest.

### Measurement sensitivity

`scripts/eq_offer_sensitivity.py` scores the US sample under both
equity-issuance measures, since Japan can only use the share-count
substitute: they disagree on **38.2%** of firm-years (mean F-Score 5.59 vs
5.91, because buybacks mask issuance), yet move Sharpe by 0.013 and the
percentile by 6 points, leaving every verdict unchanged. That bounds what
the Japanese substitute can be costing.

### Cross-validation of the signal layer

US and Japanese signals are computed by `fscore.signal.piotroski` from the
team's FS_clean statements. Scoring the same firm-years independently and
comparing against the collaborators' own implementation agrees on **100%** of
US firm-years and **99.8%** of Japanese ones (3,369 in total; the residue is
four cases at the share-count boundary). That comparison is what surfaced a
defect in this code: the year-on-year deltas had scaled the current year by
average assets and the prior year by period-end assets, putting a change of
denominator convention inside every difference. Agreement before the fix was
61% / 66%.

**Vietnam is not scored by `fscore.signal`.** The pipeline in
`src/fscore_vietnam` is the source of record for its nine signals, and both studies
read that panel — `run_study` takes `scores=`, the grid always did. An
earlier version recomputed the Vietnamese signals here from rebuilt statement
lines and agreed with the source on **100%** of firm-years and all nine
flags; since it never produced a different number, what it actually
contributed was a second copy that could drift, and it had drifted twice (it
read the extract *before* the accounting checks, and the main study fed it
only two fiscal years, silencing the beginning-of-year asset scaling in ΔROA
and Δturnover). It is gone; `src/fscore/data/vietnam.py` now supplies only
book equity, market value, sectors and the two index series.

## Data status

| Market | Prices | Fundamentals | Universe membership | Backtest window |
|---|---|---|---|---|
| US (main study) | ✅ Yahoo (daily, 2002–) | ✅ SEC EDGAR XBRL (FY2009–, true 10-K filing dates, incl. equity-issuance cash flow) | ✅ historical S&P 500 members per formation date | **formations 2012–2024** (13 full years) |
| Japan (main study) | ✅ Yahoo (daily, 2002–) | ⚠️ Yahoo (≈5 annual periods; no equity line, so the high-B/M universe starts FY2021) | ⚠️ current Nikkei 225 members | formations 2023–2024 (2 years) |
| US / Japan (grid) | ✅ Yahoo (daily, 2002–) | ✅ derived panel, `results/panel/` | ⚠️ currently listed symbols | **formations 2012–2024** (13 full years) |
| Vietnam (main study) | ✅ FireAnt, dividend-adjusted daily, 2009– | ✅ team pipeline over FireAnt/CafeF/TCBS (FY2009–, accounting-checked, **scored there**; period-end report dates, 6-month lag) | ⚠️ currently-resolvable symbols, partially survivorship-tilted (125 of 1,371 tickers stop printing before 2026) | **formations 2012–2024** (13 full years) |
| Vietnam (grid) | ✅ same panel | ✅ same panel, same scores | ⚠️ as above | **formations 2012–2024** (13 full years) |

**Why the sample runs 2012–2024.** It starts in 2012 for the reason below;
it ends with the July 2024 formation because that is the last one whose full
twelve-month holding year finishes inside the sample. Truncating a July 2025
formation at the sample end would mix a half-year window in with complete
ones, so it is excluded rather than shortened.

 EDGAR XBRL begins with the 2009–2011
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
