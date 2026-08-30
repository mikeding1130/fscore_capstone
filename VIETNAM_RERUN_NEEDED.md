# Vietnam: re-run done, and what it changed

This file used to say Vietnam's artefacts were `main`'s, restored unchanged
rather than regenerated, because the Vietnamese caches were not on the machine
that re-ran the US and Japan halves. **They have since been rebuilt and both
Vietnamese notebooks have been regenerated and executed**, so the pending
re-run this file described is complete. It is kept as the record of what that
re-run changed, since the changes are not all cosmetic.

## What was re-run

```bash
python scripts/build_notebooks.py        # regenerates 03, 04, 05
python scripts/build_grid_notebooks.py   # regenerates the three grid notebooks
```

then executed `notebooks/05_vietnam_full_study.ipynb` and
`notebooks/grid/vietnam_grid.ipynb`, followed by
`scripts/full_period.py vietnam`, `consolidated_report/consolidated_report.ipynb`
and `scripts/reconcile_report.py`.

The US and Japan notebooks were regenerated too — the generators are shared —
but only their **prose** changed, so their executed outputs were re-attached
rather than re-earned. Their code cells come out of the generator byte-identical
except for two that gained `SCORES = None` and `scores=SCORES`, and `None` is
already `run_study`'s default for that parameter, so the behaviour is unchanged
and the existing outputs remain valid for the code above them.

## What changed, and what did not

**Vietnam now runs LONG-SHORT.** `fscore.markets` used to mark it long-only and
drop `fscore_LS`. It now runs the book in all three markets, so the
high-minus-low spread is measured on one design everywhere instead of being
missing from the market where the long-only result is strongest. Short selling
of ordinary shares is still not available on HOSE/HNX, so the row is a
**hypothetical**: `markets.is_hypothetical_short("vietnam")` is `True`, and
every generated report prints that caveat beside the number.

**No pre-existing Vietnamese number moved.** Every non-`fscore_LS` row in every
grid cell and in the main study is byte-identical to what was committed before.
That is a property of the seeding, not luck: `run_grid_year` and `run_year`
construct `np.random.default_rng(seed + year)` after the weights are built, so
adding a strategy consumes no random state. The only additions are the
`fscore_LS` row, the `fscore_LS` column in `yearly_returns`, and two
diagnostics (`long_short_run`, `short_leg_names` / `short_leg_max_fscore`).

**The exclusion table changed shape**, exactly as this file predicted. It is now
written by summing the per-year frame, so it carries counts and no
`pct_scored`; the consolidated report and `reconcile_report.py` derive the
share from the counts instead. Vietnam's exclusion figure keeps its single
panel, since the EQ_OFFER split has no Vietnamese analogue.

**Two things were broken and are now fixed.** Neither was caused by the
long-short switch; both were exposed by re-running Vietnam.

* `scripts/full_period.py` handed Vietnam's fundamentals cache to the inline
  scorer. That cache carries book equity and market cap and nothing else — the
  nine signals were computed upstream — so it raised
  `KeyError: total_assets` and could not run at all. It now reads the shipped
  score panel, the same source `notebooks/05` reads. Vietnam's
  `*_fullperiod_*` numbers therefore changed: they now agree exactly with the
  headline, which is the correct answer, because Vietnam's feasible span *is*
  the headline window.
* `scripts/reconcile_report.py` required a `rows_in_source` column the grid
  stopped writing, so it raised before producing the `.docx`. It now derives
  the denominator from the counts present.

## Rebuilding the caches

`data/` tracks only `vietnam_prices.csv.gz` and `vietnam_scores.csv`.
`vietnam_fundamentals.csv`, `vietnam_sectors.csv`, `vietnam_benchmarks.csv.gz`
and `vietnam_exclusions.csv` are git-ignored, so a fresh clone still needs the
sibling repository to rebuild them before notebook `05` will run.

## Do not regenerate without re-running

`python scripts/build_grid_notebooks.py` (no `execute`) rewrites all three
notebooks and **strips their outputs**. Regenerating on a machine that cannot
execute a market turns that market's committed notebook into an empty shell.
Regenerate and execute together, or leave it alone.
