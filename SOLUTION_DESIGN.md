# Solution Design Document

**Project:** Does the Piotroski F-Score pick stocks, or does portfolio
construction do the work? Evidence from the United States, Japan and Vietnam.
**Team:** Zu Yao Teoh · Zhicheng Ding · Ta Tan Phat

## The business question

A well-known accounting screen, the Piotroski F-Score, scores a company from
0 to 9 on nine yes/no health checks drawn from its annual report — is it
profitable, is it generating cash, is it becoming less indebted, and so on.
Investors buy the high scorers and expect to beat the market.

Two things are usually taken on faith. First, that a portfolio of high
scorers beats a portfolio of *randomly chosen* stocks from the same pool —
if it does not, the screen is decoration. Second, that the screen keeps
working outside the United States, where it was discovered.

This project tests both, and separates the contribution of **picking** the
stocks from the contribution of **weighting** them.

## What the solution does

The system takes annual financial statements and daily share prices for a
market, and produces a verdict: how a portfolio built from the screen would
have performed each year, and whether that performance is distinguishable
from luck.

It is built as five independent stages, so any one can be replaced without
disturbing the others — which is what allows the same study to run across
three markets with different data sources.

| Stage | What it does | Why it exists |
|---|---|---|
| **1. Data** | Assembles statements and prices per market, from official filings (SEC EDGAR), an exchange feed, or the team's own database. | Each market has a different source; downstream stages must not care which. |
| **2. Scoring** | Computes the nine checks and the 0–9 score for each company each year. | The screen itself. |
| **3. Selection** | Forms a fixed-size basket of the top scorers — plus five comparison baskets, including thousands of randomly drawn ones. | The random baskets are the honest yardstick: they answer "could luck have done this?" |
| **4. Construction** | Weights each basket three ways: equally, and two risk-minimising schemes that use the historical co-movement of the shares. | Separates skill at picking from skill at weighting. |
| **5. Evaluation** | Rebalances annually, chains the years together, and reports return, risk, trading costs, and where the screen's portfolio sits inside the random distribution. | Turns a track record into a verdict with a stated confidence level. |

## Design decisions a reviewer should know

- **Nothing is used before it was public.** Portfolios are formed each 1 July
  using only statements already filed and prices already observed. This is
  the single most common way backtests flatter themselves.
- **Luck is measured, not assumed.** Every result is placed against
  1,000–5,000 randomly drawn portfolios from the same pool, redrawn each
  year. One confidence level, 5%, is fixed in advance for every test.
- **Costs are charged.** Each strategy pays for its own trading, and a
  strategy that sells shares short also pays to borrow them.
- **Where a strategy cannot be traded, it is not reported.** Short selling is
  unavailable in Vietnam, so the long/short variant is absent from Vietnamese
  results rather than shown as a hypothetical.
- **Limits are disclosed, not hidden.** Companies that lack a complete set of
  nine checks are excluded and counted; companies that stopped trading are
  held at their last price rather than quietly dropped.

## What is delivered

- A **code repository** with the five stages, unit tests that pin each design
  decision, and scripts that rebuild the data from scratch.
- **Twenty notebooks** that run end to end and produce every figure and table
  in the report — eighteen covering a robustness grid (three basket sizes ×
  three sample sizes × two markets) and two covering the main study.
- A **derived data panel** so results can be verified without a paid data
  subscription (see `results/panel/PROVENANCE.md`).

## Cost and effort

No paid data or infrastructure. Public filings and free price feeds, run on a
laptop; a full rebuild of the data takes about two hours, and a full
re-execution of all notebooks about one.
