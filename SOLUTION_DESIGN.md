# Solution Design Document

**Project:** Does the Piotroski F-Score pick stocks, or does portfolio
construction do the work? Evidence from the United States, Japan and Vietnam.
**Team:** Zu Yao Teoh · Zhicheng Ding · Ta Tan Phat

## 1. Business Problem

The Piotroski F-Score is a widely used accounting screen: it poses nine
pass/fail tests against a company's annual report — is it profitable, is it
generating cash, has leverage fallen, and so on — and assigns a score from 0 to
9. Investors buy high-scoring stocks in the hope of beating the market.

That approach rests on two assumptions: first, that a high-scoring portfolio
beats a **randomly drawn portfolio from the same pool of stocks** — if it does
not, the screen is doing no work; and second, that the result holds outside the
United States. This project tests both assumptions and measures the
contributions of *stock selection* and *weighting* separately.

## 2. What the System Does

The system takes a market's annual filings and daily prices as input and
returns a verdict: how a portfolio built from this screen performs year by
year, and whether that performance can be distinguished from luck. It is
organised into five independent stages, any of which can be swapped out without
touching the others — which is what allows the same study to run across three
markets with very different data sources.

## 3. Code Structure: What Each Stage Solves

| Stage | Code location | What it solves |
|---|---|---|
| **1. Data** | `src/fscore/data/` | Maps filings and prices from three markets, each in a different format, onto a common schema. `edgar.py` reads SEC public filings, `bbg_processed.py` reads the Japanese terminal data, and `vietnam.py` reads the team's own database. Key constraint: only information that was public as of the formation date may be used. |
| **2. Scoring** | `src/fscore/signal/` | Computes the nine tests and a total score of 0–9. If any of the nine cannot be computed, the firm is left unscored for that year and recorded in the exclusion counts — an incomplete score is not treated as a low score. |
| **3. Selection** | `src/fscore/selection/` | Builds a fixed-size high-score basket alongside five control baskets, including 1,000–5,000 random baskets redrawn each year. The random basket is the single most important benchmark in this study: it answers the question "could luck have produced this result?" |
| **4. Weighting** | `src/fscore/construction/` | Applies three weighting schemes to the same basket: equal weight, plus two optimisers that use the historical **co-movement** between shares to reduce risk. Holding the basket fixed and varying only the weights is what separates stock-selection skill from weighting skill. |
| **5. Evaluation** | `src/fscore/evaluation/` | Rebalances annually and chain-links across years, reporting return, risk, trading costs, and where the portfolio sits within the random distribution, turning a track record into a statement about confidence. |
| **Main pipeline** | `src/fscore/pipeline.py`, `grid.py` | `pipeline.py` runs the five stages end to end to produce the main results; `grid.py` reruns them across 27 parameter combinations (3 basket sizes × 3 sample sizes × 3 markets) to check whether the conclusions depend on any single setting. |
| **Reproducibility safeguards** | `tests/` (22 tests), `scripts/` | Each test pins down a design decision so that later changes cannot silently alter the conclusions; `scripts/` rebuilds all data from scratch and exports the derived panels for external verification. |

## 4. Key Design Decisions

- **No information is used that was not public at the time.** Portfolios are
  formed on 1 July each year using only filings already published and prices
  already realised. Look-ahead bias is the most common way a backtest flatters
  itself.
- **Luck is measured, not assumed.** Every result is benchmarked against
  1,000–5,000 portfolios drawn at random from the same pool of stocks and
  redrawn each year; the significance level is pre-specified at 5%.
- **Trading costs are always charged.** Each strategy pays for its own
  turnover, and short strategies additionally pay stock borrow fees.
- **Strategies that cannot be traded are not reported.** Short selling is not
  permitted in Vietnam, so no long-short strategy appears in the Vietnamese
  results; the constraint is documented rather than assumed away.
- **Limitations are disclosed, not buried.** Firms with an incomplete set of
  nine signals are excluded and the exclusions are counted; companies that stop
  trading are held at their last traded price rather than quietly dropping out
  of the sample, which would let the portfolio escape delisting losses for
  free.

## 5. Deliverables and Cost

There are three deliverables: a **code repository** (five stages, 22 tests, and
scripts that rebuild the data from scratch); **eight notebooks** — six that run
end to end and generate every figure in the report (three main studies, one per
market, and three robustness sweeps of nine grid cells each), plus two
synthetic-data demonstrations that need no network; and a **derived data
panel** that lets others reproduce the results without buying licensed data
(see `results/panel/PROVENANCE.md`).

**Cost: no paid data or infrastructure is required.** The US portion rests
entirely on public filings and can be rebuilt end to end; Japan and Vietnam
rely on licensed or self-built data, and the raw data is not redistributed —
only irreversibly derived panels are shared. All computation runs on a single
laptop: rebuilding the data takes about two hours, and rerunning every notebook
takes about one more.
