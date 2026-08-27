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
| `src/fscore/data/` | loaders and point-in-time alignment; `score_panel.py` (canonical statements → the study's score panel, shared by the grid and the main study), `bbg_processed.py` (the Bloomberg tree; Japan), `edgar.py` (SEC filings; US), `fs_clean.py` (the superseded team workbook), `yahoo.py` (prices), `universe.py` (constituents + sectors) |
| `src/fscore/signal/` | the nine Piotroski signals and composite score |
| `src/fscore/selection/` | fixed-size baskets: F-Score top-k, Monte-Carlo random, liquidity-matched, value, market-cap |
| `src/fscore/construction/` | EW / long-only GMV / sector-capped GMV (exact SLSQP solve); RMT covariance cleaning |
| `src/fscore/evaluation/` | backtest metrics, random-distribution placement; `benchmarks.py` (investable ETFs, Fama-French 3-factor regression) |
| `src/fscore/pipeline.py` | full multi-year study runner (PIT snapshot → universe → score → baskets → weights → chained backtest) |
| `src/fscore/markets.py` | per-market trading constraints (where a short leg is tradable) |
| `src/fscore/grid.py` | robustness grid runner (basket size × random-sample size × market) |
| `src/fscore/plotting.py` | figure defaults; every saved chart is 300 dpi |
| `scripts/` | data builders (`fetch_us_edgar.py`, `fetch_us_japan.py`, `build_japan_bbg.py`, `deepen_japan_prices.py`), notebook generators, `full_period.py`, `eq_offer_sensitivity.py`, `export_panel.py` |
| `notebooks/` | `01`/`02` synthetic demos; `03`/`04` full studies; `grid/` two notebooks sweeping 18 robustness cells |
| `data/` | git-ignored cache, rebuilt by the fetch scripts |
| `results/` | CSVs and 300-dpi figures behind every table and chart in the report |
| `tests/` | 20 tests, each pinning one design decision |

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

The grid takes a little over an hour (two notebooks, nine cells each); the two
main-study notebooks about twenty minutes. Results are deterministic — every
random draw is seeded — so a clean run reproduces the reported figures
exactly.

Notebooks `01`/`02` are self-contained demos on **clearly-labelled synthetic
data**; they need no network and are the fastest way to see the pipeline end
to end.

For a non-technical overview of what the system does and why, see
[SOLUTION_DESIGN.md](SOLUTION_DESIGN.md).

## Peer-review grid study (notebooks/grid/)

Following the M5 peer review, `src/fscore/grid.py` re-runs the developed pair
across the reviewer's grid. Each market's grid reads **the same statements its
main study reads** — SEC EDGAR for the US, the Bloomberg tree for Japan, via
`fscore.data.score_panel` — so the grid varies k and N against the reported
dataset rather than against a different one, and both rest on one signal
implementation (`fscore.signal.piotroski`) — basket size k ∈ {20, 25, 30} × random draws N ∈ {1000, 2000, 5000} ×
{US, Japan} = **18 grid cells**, figures saved at dpi = 300. The cells run as
a sweep inside **two notebooks**, `us_grid.ipynb` and `japan_grid.ipynb`,
rather than the eighteen separate files they started as; each cell still
writes its own `{market}_k{k}_mc{N}_*` outputs. That merge was checked
value-by-value against the eighteen-notebook results and reproduced all 108
CSVs exactly (`scripts/verify_grid_merge.py`); the numbers have since moved,
but from the later switch of data source, not from the merge.
Design answers the review's four priorities: explicit random basis (full
eligible universe, fresh yearly draws, overlap reported), the **direct synergy
test** D = Sharpe(GMV) − Sharpe(EW) per basket, a strict F ≥ 8 portfolio, and
one pre-registered primary measure (the gross Sharpe ratio, with turnover
reported separately) tested at one pre-registered significance level (**5%**,
`fscore.evaluation.ALPHA`). Formations run **July 2012 – July 2024** —
thirteen chained holding years, each a full twelve months, the window shared
with the main study; covariances use 36 months of daily returns. Aggregated
outputs in `results/grid/`, consolidated per market in
`results/grid/{market}_grid_summary.csv`, which the notebooks build
themselves — it used to be assembled by hand;
build/execute with `python scripts/build_grid_notebooks.py execute`.

Headline: the percentile is insensitive to the number of random draws —
1,000 and 5,000 agree to within two percentage points. The consolidated table
shows why: within a given k, every column but the percentile and its p-value
is *bit-identical* across N, because the same seed draws the same first 1,000
baskets whether the run asks for 1,000 or 5,000. N buys resolution on the
p-value, nothing else — the grid is three independent settings tested at three
precisions, not nine settings.

| Sharpe percentile vs random | k = 20 | k = 25 | k = 30 |
|---|---|---|---|
| US | 32% | 21% | 47% |
| Japan | 68% | 69% | 63% |

**No cell in either market clears 5%**, and neither market's F-Score screen
beats plain universe equal weight. In the US the value screen alone is the
strongest thing on the table (0.90–0.99 against 0.81 for the universe), and
ranking *within* it by F-Score gives back most of that edge (0.70–0.77). In
Japan the value screen is a drag (0.68–0.70 against 0.75), and the F-Score
roughly offsets it (0.76). Neither market shows the screen and the score
working in the same direction.

The grid draws its random baskets from the **full eligible universe**, which
is a different null from the main study's (the high-B/M subset). For Japan the
two disagree — 62.5% here against 96.8% there — and the section above sets out
why both are correct. The optimisation gain D is negative in all eighteen
cells (−0.21 to −0.05).

**Basket size is not the number of holdings.** At k = 25 the equal-weight
book holds 25 names by construction, but the minimum-variance book has an
effective N (1/Σw²) of **5.1–5.9**. The optimiser concentrates into a handful
of names; k is an upper bound, and the two quantities sit side by side in
every summary table.

## Headline results — main study

From the executed notebooks `03`/`04`. The headline convention is **gross**
performance (cost models differ by market and would confound a cross-country
comparison); turnover and net-of-cost figures are reported separately.
Every verdict is at the single pre-registered level, **α = 5%**.

Both markets now span the same thirteen formations, **Jul 2012 – Jun 2025**.

| | US (S&P 500, SEC EDGAR) | Japan (TPX100, Bloomberg) |
|---|---|---|
| F-Score EW | 15.9% p.a., Sharpe 0.81 | 15.6% p.a., Sharpe 0.76 |
| vs random baskets | 47th pct, p = 0.53 — n.s. | 96.8th pct, p = 0.032 — **significant** |
| + GMV | 97th pct, p = 0.030 — **significant** | 81st pct, p = 0.19 — n.s. |
| + sector-GMV | 99.7th pct, p = 0.003 — **significant** | 94th pct, p = 0.060 — n.s. |
| vs market ETF | 0.81 vs SPY 0.85 | 0.76 vs TOPIX 0.68 |
| vs pure value screen | 0.81 vs 0.86 | 0.76 vs 0.69 |
| Long-short (high − low) | Sharpe −0.11 | 0.35 |
| Effective N (k = 30) | EW 30, GMV 5.6, sector-GMV 8.8 | EW 30, GMV 5.0, sector-GMV 7.1 |
| Turnover (EW) | 0.61 one-way per year | 0.31 |

The two markets answer in mirror image: in the US selection does nothing and
the optimiser clears 5%; in Japan selection clears 5% and the optimiser does
not. Neither pattern survives the grid's wider random basis (below), and
Japan's figure carries the EQ_OFFER caveat.

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

Japan now runs the same thirteen formations, on the Bloomberg statements in
`data/processed/Japan/` (fiscal years from 2000, book equity from
`Common_Shareholders_Equity`), and its answer depends on **which random basis
the question is asked against** — a distinction worth stating carefully,
because the two numbers look like a contradiction and are not:

| random basis | random mean | F-Score EW | percentile | p |
|---|---|---|---|---|
| high-B/M subset (~35 names, main study) | 0.721 | 0.758 | 96.8% | **0.032** |
| full eligible universe (~85 names, grid) | 0.739 | 0.756 | 62.5% | 0.375 |

Both are correct answers to different questions. Japan's value screen is a
**drag**: `value_EW` 0.69 against `universe_EW` 0.75. Inside the high-B/M
subset the F-Score does pick better than chance; but the screen-plus-score
combination does not beat picking at random from the whole universe. The
F-Score is largely filling in the hole the value screen digs. Quoting either
figure alone misrepresents the result.

The US grid tells the reverse story: the value screen there is strongly
positive (`value_EW` 0.90–0.99 against `universe_EW` 0.81), yet ranking within
it by F-Score **loses** at every basket size (0.70–0.77), and no cell of
either market's grid clears 5%. The optimisation gain D = Sharpe(GMV) −
Sharpe(EW) is negative in all eighteen cells.

### Measurement sensitivity

`scripts/eq_offer_sensitivity.py` scores the US sample under both
equity-issuance measures, since Japan can only use the share-count substitute
— the Bloomberg tree ships a `Proceeds_Issuance_Common_Stock` column but it is
empty in every sheet of both markets. They disagree on **38.2%** of firm-years,
and the disagreement has a direction: share counts miss issuance that buybacks
net away, so the substitute is the **generous** measure (mean F-Score 6.08
against 5.76). It moves Sharpe by 0.025 and the percentile by 14 points,
leaving the US verdict unchanged — but Japan sits on the generous side of that
gap, so its percentiles should be read with the bias in mind rather than
quoted flat.

### Cross-validation of the signal layer

Signals are computed by `fscore.signal.piotroski`. Scoring the team's FS_clean
firm-years independently and comparing against
the collaborators' own implementation agrees on **100%** of US firm-years and
**99.8%** of Japanese ones (3,369 in total; the residue is four cases at the
share-count boundary). That comparison is what surfaced a defect in this
code: the year-on-year deltas had scaled the current year by average assets
and the prior year by period-end assets, putting a change of denominator
convention inside every difference. Agreement before the fix was 61% / 66%.

## Data status

| Market | Prices | Fundamentals | Universe membership | Backtest window |
|---|---|---|---|---|
| US (main study) | ✅ Yahoo (daily, 2002–) | ✅ SEC EDGAR XBRL (FY2009–, true 10-K filing dates, incl. equity-issuance cash flow) | ✅ historical S&P 500 members per formation date | **formations 2012–2024** (13 full years) |
| Japan (main study) | ✅ Yahoo (daily, 2002–) | ⚠️ Yahoo (≈5 annual periods; no equity line, so the high-B/M universe starts FY2021) | ⚠️ current Nikkei 225 members | formations 2023–2024 (2 years) |
| US / Japan (grid) | ✅ Yahoo (daily, 2002–) | ✅ derived panel, `results/panel/` | ⚠️ currently listed symbols | **formations 2012–2024** (13 full years) |
| Vietnam | TODO | TODO | TODO | team-maintained database, 10y |

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
