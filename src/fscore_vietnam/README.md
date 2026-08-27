# `fscore_vietnam` — the Vietnamese preprocessing pipeline

Crawl → reconcile → check → score → screen. This is the **source of record**
for every Vietnamese number in the study: `fscore.signal.piotroski` scores the
US and Japan, and `f_score_calculation.ipynb` here scores Vietnam. Nothing
downstream recomputes those flags.

It lived in a separate `thesis` repository until it was moved in here. Only
the paths changed — every notebook now gets its locations from `paths.py`
instead of writing `../fscore.db` and hoping the working directory matches.

## Layout

```
src/fscore_vietnam/          this package: notebooks + the shared modules
data/vietnam_pipeline/       DATA    — crawled universe files
data/vietnam_pipeline/results/   RESULTS — every panel the pipeline writes
<wherever you keep it>/fscore.db     the crawl, ~1.7 GB, NOT in the repository
```

`paths.py` resolves all four. Override any of them with `FSCORE_DB`,
`FSCORE_VN_DATA`, `FSCORE_VN_RESULTS`. The default for `fscore.db` is the old
`../thesis` checkout, so a machine that already has it keeps working with no
configuration; set `FSCORE_DB` once if you move it.

Run the notebooks **from this directory** — they assert on it, because
`import statement_fields` and `import paths` both resolve against the working
directory.

## Order

Each step reads what the previous ones wrote, so run them in this order after
a fresh crawl. Steps 1–4 need `fscore.db`; everything after reads CSVs only.

| # | notebook | writes |
|---|---|---|
| 1 | `simple_filter.ipynb` | `simple_filter_fa_price*.csv` — the coverage screen |
| 2 | `fields_extract.ipynb` | `f_score_fields_extract.csv` — canonical statement lines |
| 3 | `accounting_checks_and_corrections.ipynb` | `..._corrected.csv`, `accounting_clean_bol_matrix.csv` |
| 4 | `book_to_market_calculation.ipynb` | `book_to_market_panel.csv` |
| 5 | `f_score_calculation.ipynb` | `f_score_panel.csv` — **the nine signals** |
| 6 | `tradable_by_turnover_filter.ipynb` | `trade_turnover_panel.csv` |
| 7 | `price_matching_and_finalize.ipynb` | `final_panel.csv` |
| 8 | `run_grid_export.ipynb` | `vietnam_scores.csv`, `vietnam_prices.csv.gz` |

`share_count_diagnostics.ipynb` and `playground.ipynb` are diagnostics — they
write nothing the study depends on.

Step 8's two files are what both studies rank on. `schema_adapter_util.py`
then adds the four supporting caches that export leaves out:

```bash
python src/fscore_vietnam/schema_adapter_util.py
```

## Shared modules

* `statement_fields.py` — the canonical field layer. `FSCORE_INPUTS` is the
  contract that decides which statement lines get pulled, and
  `f_score_calculation.ipynb` asserts its signal set has not drifted from it.
* `schema_adapter.py` — maps the pipeline's panels (`symbol`, `period`,
  `book_equity`, …) into the canonical schemas `fscore` consumes (`ticker`,
  `fiscal_year`, `book_value`, …), plus the sector map, the VN30/VNINDEX
  series and the exclusion funnel. It computes no signal and imports nothing
  from `fscore`.
* `schema_adapter_util.py` — runs that adapter once, writes the four cache
  files, and asserts on what it wrote: canonical columns present, unique on
  (ticker, fiscal_year), and **no statement line or signal flag anywhere in
  it**. That last assert is the guard against Vietnam quietly being re-scored
  outside this package again.
* `accounting_checks_util.py` — the seven identity checks (balance identity,
  gross-profit decomposition, cash-flow roll, …). A firm-year has to pass all
  seven to reach `accounting_clean_bol_matrix.csv`, which is what every later
  step joins against.

## Data is not in git

`data/vietnam_pipeline/` is 73 MB of CSV and `fscore.db` is 1.7 GB; both are
git-ignored. Regenerate the panels by running the order above, or copy the
directory from a machine that has it.
