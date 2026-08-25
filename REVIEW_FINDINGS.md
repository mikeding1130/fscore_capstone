# Review findings — edge cases, failures, bugs and misreadings

Written while wiring Vietnam into the same two experiment families the US and
Japan already run. Everything below is either a defect I could reproduce, or a
place where two numbers in the repository cannot both be read the way the
current write-up reads them. Items marked **fixed here** were changed in this
session; the rest are left alone deliberately, because fixing them changes
published numbers and that is the team's call, not mine.

Ordering is by how much a reader would be misled, not by how hard the fix is.

---

## A. Defects that produce a wrong or misleading number

### A1. `full_period.py <one market>` deleted the other markets' results — **fixed here**

`main()` collected the rows for the markets it ran and wrote
`results/robustness_full_period.csv` from scratch. Running it for one market
therefore replaced a three-market file with a one-market file, with no warning.
Reproduced: `python scripts/full_period.py vietnam` took the file from six rows
(US, Japan) to three (Vietnam).

Fixed by merging: rows on disk that this run did not produce are kept, rows it
did produce win.

### A2. The RMT-detoning section overwrote the US figure — **section since removed**

`build_notebooks.py` gated the detoning diagnostic on `usd_convert` being
false, which is true for the US *and* for Vietnam, and the cell inside it
hard-coded `save_fig("us_detone_comparison")`. The first Vietnam run therefore
overwrote `results/figures/us_detone_comparison.png` with a Vietnamese chart
under an American name.

The team has since dropped detoning from the study, so section 4b is gone from
every notebook and the bug with it. The general lesson stands and applies to
other cells: **a figure name written as a literal rather than derived from
`MARKET` is a collision waiting for the second market to run.** Worth a grep
before the next market is added.

### A3. The Japan demo notebook wrote the US demo's figures — **fixed here**

`build_japan_demo()` rewrote the market name and the seed but not the figure
names, so regenerating notebook 02 produced a Japanese notebook that saves
`demo_us_fscore_distribution.png` and `demo_us_mc_placement.png`. The committed
notebook already carried the Japanese names, which means the notebook and its
generator had silently drifted apart — the committed file was not what the
script produces. Fixed with `.replace("demo_us_", "demo_japan_")`.

### A4. `notebooks/grid/vietnam_k25_mc1000.ipynb` wrote every output under the wrong tag

The notebook sets `K = 25` and `TAG = "vietnam_k30_mc1000"`. Every CSV and every
figure it writes lands under a `k30` name while the run is a `k=25` run. The
committed results are named `vietnam_k25_*`, so the notebook in the repository
did **not** produce the results in the repository — they came from some other
version. That notebook is now fully superseded by `notebooks/grid/vietnam_grid.ipynb`,
which sweeps the same nine cells. **Deleted in this session** — the file is in
git history if it is ever wanted back, and leaving a notebook that mislabels
its own outputs in the tree is worse than losing it.

### A5. `nominal_k` is the first year, `effective_n` is the mean of all years

`GridStudy.summary()` and `StudyResult.summary()` both set
`nominal_k = len(self.yearly[0].weights[name])` — the first formation only —
and sit it next to `effective_n`, which averages over every formation. The two
columns describe different periods and are read as a pair.

It bites hardest where the basket size is not constant:

* `results/grid/japan_k30_mc5000_summary.csv` reports `value_EW` with
  `nominal_k = 78` and `effective_n = 66.7`. The 78 is 2012, when the value
  control had fallen back to the whole universe; by 2023-24 it was 30 names.
* `results/japan_summary.csv` reports `fscore_EW` with `nominal_k = 9` and
  `effective_n = 19.5`. The 9 is the 2023 formation (see A7); the 19.5 is the
  mean of 9 and 30.
* Clearest of all, `results/grid/vietnam_k20_mc5000_summary.csv` reports
  `universe_EW` with **`nominal_k = 303` and `effective_n = 586.8`** — an
  equal-weight portfolio whose effective number of holdings is nearly twice
  its nominal size. For a single portfolio that is arithmetically impossible;
  the 303 is the 2012 universe and the 586.8 is the mean across a universe
  that grows to 848 by 2024. The same row reports `fscore_high_EW` at
  `nominal_k = 25`, `effective_n = 65.5`.

Report the range or the mean, not year one.

### A6. "Yearly returns" are calendar years, but the holding years are July–June

`GridStudy.yearly_returns()` does `groupby(r.index.year)`, and the notebook
prints it under the heading *"Yearly returns (each formation held July–June, a
full twelve months)"*. Every row in that table mixes the second half of one
holding year with the first half of the next, and the first and last rows are
half-years outright.

`results/grid/vietnam_k25_mc5000_yearly_returns.csv` shows it: the run covers
formations 2012–2024, so the daily series starts 2 July 2012 and ends 30 June
2025 — yet the table has **fourteen** rows, 2012 through 2025. The 2012 row is
July–December only and the 2025 row is January–June only. Neither is a year,
and neither is a holding period. Either the aggregation should key on the
formation year, or the heading has to stop claiming it does.

### A7. The long-short book disappears silently when `k` equals the scored count

`run_year` takes `head(k_eff)` as the long leg and `tail(k_eff)` as the short
leg. When `k_eff == len(scored)` the two are the same names, `shorts` comes out
empty, `weights[LONG_SHORT]` is an empty Series and the strategy is filtered out
of that year — with no diagnostic.

That is exactly what happened to Japan's 2023 formation, where only 9 firm-years
scored, so `k_eff = 9`. The consequences are all in `results/japan_summary.csv`:
`fscore_LS` is a **one-year** track record chained as if it were two, its
`turnover` is `0.0` (there is no consecutive pair of years to measure a trade
between), and its cost drag is therefore the 100 bp borrow fee alone — an
untraded long-short book. The reported `fscore_LS` Sharpe of −1.32 for Japan is
one year of a nine-name basket, not a strategy result.

Minimum fix: refuse to build the long-short leg unless `len(scored) >= 2 * k`,
and record why in the diagnostics.

### A8. The Monte-Carlo p-value has no `+1` correction

`vs_random` uses `p = mean(random >= observed)`. The unbiased empirical p-value
is `(1 + #{random >= observed}) / (N + 1)`; as written, a portfolio that beats
every draw is reported at `p = 0.000`, which no finite simulation can support.
At N = 1000 the shift is about 0.001 — small, except that several headline
numbers in this study sit at 0.039, 0.042 and 0.047 against a 0.05 line.

---

## B. Design issues — the numbers are correct but cannot carry the weight put on them

### B1. The same Monte-Carlo test has completely different power in each market

The grid universe is **67–78 names in the US**, **73–84 in Japan**, and
**303–856 in Vietnam**. With `k = 30` drawn from ~70, a "random" basket already
shares about 40% of its names with the F-Score basket. The diagnostics say so
directly (`overlap_random_vs_fscore`): ≈ 0.38–0.45 in the US, ≈ 0.36–0.41 in
Japan, ≈ 0.03–0.08 in Vietnam.

A test whose control is 40% the same portfolio as the treatment cannot detect a
moderate effect. So "no cell is significant in the US or Japan, six of nine are
in Vietnam" is partly a statement about universe size, not about the F-Score.
This is the single most important caveat for the cross-country comparison, and
it is currently nowhere in the write-up.

The clean way to neutralise it is to hold the *ratio* k/|universe| fixed rather
than k, or to draw the developed-market random baskets from a universe as large
as Vietnam's. Neither is free — the US grid universe is 67–78 names because
that is what the panel resolves — but until one of them is done, the three
markets' p-values are not on the same scale and should not be tabulated as if
they were.

### B2. Most of the "F-Score basket" is decided by the tie-break, and only one draw is reported

The score is an integer 0–9, so the top-*k* cut almost never falls cleanly.
`rank_by_fscore` breaks the tie at random, seeded per formation year, and
`tie_break_slots` records how many slots that decides. For Vietnam at k = 25 the
sequence across formations is **19, 18, 11, 14, 15, 12, 10, 12, 8, 2, 25, 25,
14, 25** — in three years *every* name in the basket is a coin flip among firms
sharing the same score. The US at k = 30 ranges 5–19, Japan 2–18.

The reported basket is therefore one sample from a large set of equally
F-Score-justified baskets, and its Sharpe carries that sampling noise on top of
everything else. A single seed cannot show this, so
`scripts/tie_break_sensitivity.py` was added and run
(`results/vietnam_tiebreak_sensitivity.csv`, Vietnam k = 25, N = 1000,
formations 2012–2024, eight seeds). **The result does not survive it:**

| seed | 42 | 43 | 44 | 45 | 46 | 47 | 48 | 49 |
|---|---|---|---|---|---|---|---|---|
| F-Score EW Sharpe | 1.318 | 1.303 | 1.356 | 1.270 | **1.211** | 1.261 | 1.334 | 1.244 |
| p vs random | 0.010 | 0.016 | 0.011 | 0.026 | **0.059** | 0.030 | 0.015 | **0.050** |
| significant at 5% | yes | yes | yes | yes | **no** | yes | yes | **no** |

Sharpe spans 1.211–1.356 (sd 0.049) and the p-value spans 0.010–0.059
(sd 0.018). **Six of the eight seeds clear 5%; two do not.** The seed the study
happens to use, 42, is the *most favourable of the eight* — it produces the
lowest p-value in the set. Nothing about the data changed between those runs;
only which of several equally high-scoring firms landed in the basket.

The same sweep puts an error bar on the synergy statistic: `D_fscore` ranges
0.170 to 0.451 with a median of 0.265, and the published 0.451 is again the
maximum of the eight. Its p-value never falls below 0.14 on any seed, so that
conclusion (no synergy) is the one robust result here.

**What to do about it.** Reporting one seed's Sharpe as the answer is not
defensible when the tie-break decides 14 of 25 slots on average and all 25 in
two of the thirteen formation years. Either report the seed distribution as the
result (median and range, as above), or replace the arbitrary tie-break with an
average over many draws — the basket becomes a portfolio of all equally-scored
candidates, which is what "the F-Score chose these" actually means.

### B3. Japan's `value_EW` is three different portfolios chained into one line

From `results/grid/japan_k30_mc5000_diagnostics.csv`: `bm_coverage = 0` for
formations 2012–2021, so `value_fallback` fired and the "value control" was the
entire universe (73–83 names). In 2022 coverage was 7 names — and
`value_fallback` did *not* fire, because the threshold is
`len(value_pool) < max(5, k // 4)`, which at k = 30 is 7. So 2022's value
control is a **seven-name portfolio**. Only 2023 and 2024 are a real 30-name
high-B/M basket.

That chained series is labelled `value_EW` and reported beside the US and
Vietnamese value controls as if it were the same object. It is not, and the
comparison should be withdrawn until Japan has B/M coverage.

### B4. The Japanese main study is two formation years, one of them nine names

`results/japan_summary.csv` covers formations 2023 and 2024 only, and 2023
scored 9 firm-years against a nominal basket of 30. Any US-vs-Japan-vs-Vietnam
sentence about the *main* study compares thirteen years against two. The grid
study does cover 2012–2024 for Japan; keep the two straight in the text.

### B5. Vietnam's two experiment families disagree, and both are in the repository

| | universe | formations | F-Score EW Sharpe | p vs random |
|---|---|---|---|---|
| grid, k = 25, N = 5000 (this run) | whole scoreable panel, 303–856 names | 2012–2024 | **1.318** | **0.015** |
| grid, k = 30, N = 5000 (this run) | same | 2012–2024 | 1.184 | 0.105 |
| main study | top-150 liquid ∩ top-40% B/M, ~60 names | 2012–2024 | **0.438** | 0.085 |
| grid, k = 25, N = 1000 (superseded) | same | 2012–**2025** | 1.702 | 0.039 |

All four are correct for the question they ask, and they are not the same
question. Three things to keep straight:

* The grid's Vietnamese result is significant at k = 20 and k = 25 and **not**
  at k = 30 — the sensitivity to basket size the grid exists to expose.
* The main-study Vietnamese F-Score portfolio sits **below both index rows**
  (VN30 0.474, VNINDEX 0.537).
* The previously committed cell used a fourteenth formation year (2025) that
  neither of the other markets has, which is most of the difference between
  1.702 and 1.318. It has been replaced by the nine-cell sweep on the common
  2012–2024 window.

A sentence like "the F-Score works in Vietnam" has to name which family, which
basket size, and which window it means.

### B6. In Vietnam the best portfolio is the one with no selection at all

Every Vietnamese grid cell reports `universe_GMV` at Sharpe **2.49** against
`fscore_GMV` 1.51–1.77 and `fscore_EW` 1.18–1.32. Minimum variance applied to
the *whole universe* — no F-Score, no screen of any kind — beats minimum
variance applied to the F-Score basket by a wide margin, in every cell. The
plain universe **equal weight** (1.285) also matches or beats the F-Score
basket at every k. So the Vietnamese screen beats a random 25-name basket and
does not beat simply holding everything. Any claim that the optimiser and the
score work well together has to explain both controls first.

One caveat about the control itself, from the sibling repository's own export
notebook: the whole-universe covariance runs at `q = n_assets / n_obs > 1` in
every formation year from 2013, because names that listed inside the 36-month
window truncate the estimation sample. Denoising keeps the matrix finite, but
this is a materially weaker estimate than the same control in the developed
markets. That is a reason to state the caveat next to the number, not a reason
to drop the number.

### B7. Vietnam's panel is liquidity-screened before the study sees it; the others are not

Full funnel, from `data/vietnam_exclusions.csv`: 23,493 firm-years in
the statement extract → **9,482 scored (40.4%)**. The largest single reason is
**5,296 firm-years removed by the June-turnover tradability gate**, followed by
2,388 with no prior year, 2,281 with an incomplete nine-signal score, 2,260
rejected by the accounting checks and 1,691 with no book-to-market.

The US scores 83.0% of its source rows and Japan 88.7%, with no liquidity gate
at all. So the Vietnamese universe has already had its illiquid tail removed
before the F-Score is applied, and the US and Japanese universes have not. That
is a plausible partial explanation for Vietnam looking better, and it must be
disclosed as a design difference rather than left inside a preprocessing repo.

### B8. `fscore_high_EW` (F ≥ 8) is a different animal in each market

`n_fscore_high` per formation: **25–132 in Vietnam**, **3–19 in the US**,
**1–22 in Japan**. The "strict Piotroski cutoff" line is therefore a ~5-name
concentrated book in the US and a ~90-name near-index portfolio in Vietnam.
Their Sharpes are not comparable, and in Vietnam the strict portfolio is barely
a screen at all.

### B9. The column called `sharpe` is not a Sharpe ratio

`metrics()` computes `(CAGR − rf) / (daily sd × √252)`. The numerator is a
compound growth rate, the denominator the standard deviation of arithmetic
returns. It is a CAGR-to-volatility ratio and is systematically below the
textbook Sharpe for volatile series. The handoff document says so; the CSV
column headers, the notebook tables and the figure titles all still say
"Sharpe". Rename the column, or ship a one-line note with the CSVs — otherwise
every downstream reader will take it at face value.

### B10. Vietnam's benchmark is a price index; its portfolios are total-return

VN30 and VNINDEX in `fscore.db` are capital indices — no cash dividends. The
portfolios are built on FireAnt's `adj_ratio`-adjusted closes, which *do* adjust
for dividends. The gap is the index dividend yield, historically ~1.5–2% a year,
and it is handed to every portfolio row for free. Either source a total-return
index (VN30-TRI) or add the yield back explicitly before quoting
"beats the benchmark".

### B11. One extra formation year flips Vietnam's main-study significance

From `results/robustness_full_period.csv`, same code and same seed:

| window | formations | Sharpe | percentile | p |
|---|---|---|---|---|
| 2011–2024 (Vietnam's own full span) | 14 | 0.446 | 0.958 | **0.042** |
| 2012–2024 (common headline window) | 13 | 0.438 | 0.915 | 0.085 |

A single formation year moves the result across the 5% line. That is not a
robust finding either way; report both windows side by side and say so.

### B12. The US and Japan exclusion tables report zero incomplete-signal drops — check it

`data/{market}_exclusions.csv` (as republished in every grid cell's
`*_exclusions.csv`) reports `dropped_incomplete_signals = 0` for both the US
and Japan, against 2,281 for Vietnam. Either the FS_clean workbooks are
pre-cleaned so that every surviving firm-year really does have all nine
signals computable — plausible — or the count is being lost before it is
written and those rows are silently absorbed into `dropped_no_prior_year`,
which `_write_exclusions` computes as the residual. I could not tell which,
because the workbooks are not on this machine. It is a one-line check on a
machine that has them, and it decides whether that column means anything.

### B13. Survivorship is partial, and the residual is not measurable from these panels

The Vietnamese price panel does retain names that stopped trading: 125 of its
1,371 tickers print for the last time before 2026, spread fairly evenly across
2012–2025. So it is not a purely surviving universe. But the panel is built from
symbols the vendor still resolves, and names removed from the vendor entirely
leave no trace to count. The same applies to Japan (current Nikkei 225 members)
and, more weakly, to the US (historical S&P 500 constituents, but names whose
price history has vanished from Yahoo still drop out). State it as "partial and
unquantified", not as "documented".

---

## C. Readings of the results that the numbers do not support

**C1. "GMV is significant versus random-GMV" is not "GMV improves the F-Score
basket."** Those are different statements and the study measures both. The
direct test is `D = Sharpe(GMV) − Sharpe(EW)` placed in the distribution of the
same statistic across random baskets — and `D` is negative at every `k` in the
US, and the synergy p-value never falls below 0.05 in any market or any cell. A
significant GMV-vs-random-GMV result can be produced by a good stock list alone.

**C2. `N ∈ {1000, 2000, 5000}` are not three replications.** They are nested
samples drawn from the same seed, so the first 1,000 baskets are identical in
all three. Only the resolution of the p-value changes. The grid tables show this
plainly — `EW` is byte-identical down each `N` block — so do not read the small
percentile drift as a finding.

**C3. "The random control is inside the value subset, so the test controls for
value."** True of the **main** study only. The grid's random control is drawn
from the full scoreable universe, so its null is "does the score beat a random
pick of *anything*", not "does the score add to value". Two different nulls,
two different sentences.

**C4. A 30-name GMV book is not a 30-name portfolio.** `effective_n` on the
grid's minimum-variance books is 4.7–7.5 against a nominal k of 20–30, and 6.97
on Vietnam's main-study `fscore_GMV`. Whatever the optimiser's Sharpe is, it is the Sharpe of a
seven-stock portfolio, with the concentration risk that implies.

---

## D. Concrete next steps

1. ~~Delete `notebooks/grid/vietnam_k25_mc1000.ipynb`~~ — done (A4);
   `vietnam_grid.ipynb` sweeps the same nine cells over the corrected window.
2. Run `scripts/tie_break_sensitivity.py` for the US and Japan too — Vietnam
   is done and the answer there was that two of eight seeds lose significance
   (B2). Report the seed spread beside every headline Sharpe and p-value, or
   average the tie-break away.
3. Fix A5–A8 in code. They are small, and each one currently prints a number
   that reads as something it is not.
4. Choose one Vietnamese formation window and use it in both experiment
   families (B5, B11). If the 2025 formation is admitted for Vietnam, say why
   the other two markets stop at 2024.
5. Either cap the Vietnamese universe by liquidity before `universe_GMV`, or
   print the `q = n/T` ratio next to that row every time (B6).
6. Add a Vietnam row to the README results table and a Vietnam section to
   `data/README.md` — both still say TODO.
7. Decide what to do about Japan: with two main-study formations and no B/M
   coverage before 2022, the Japanese main study cannot support a
   cross-country claim. The Japanese *grid* can.
