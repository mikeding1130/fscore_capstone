"""Export the submittable F-Score panel, so results can be verified.

The grading rubric asks that "the underlying data for the code should be
available for verification". The raw fundamentals behind the signals come
from a licensed terminal and are not ours to republish, so what ships is the
**derived** panel: the nine binary Piotroski flags and the 0-9 composite.

Why that distinction holds (see PROVENANCE.md for the per-column account):

  * each flag is the result of a comparison — `f_roa = 1` says ROA was
    positive, not what ROA was — so the mapping from statements to flags is
    many-to-one and cannot be inverted;
  * one firm-year emits nine bits from roughly fifteen continuous line items
    across three fiscal years, so the panel is no substitute for the source;
  * the facts underneath are public statutory filings (10-K, 有価証券報告書)
    that any reader can recompute from, independently of any vendor.

No continuous per-security value is shipped at all. `shares_outstanding` is a
raw vendor figure and nothing reads it — the EQ_OFFER flag was computed inside
the workbook from its own year-on-year share columns, so the flag survives its
removal intact. `bm` and `market_cap` are ours rather than the vendor's, but
they are still per-security numbers, so they are recomputed at load time from
the rebuildable caches instead of travelling with the panel.

Run:  python scripts/export_panel.py [us|japan|both]
Writes results/panel/{market}_fscore_panel.csv and results/panel/PROVENANCE.md
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fscore.data.team_scores import SIGNAL_COLS, load_team_scores  # noqa: E402

OUT = ROOT / "results" / "panel"

# Only irreversible content leaves the repository: identifiers, the nine
# one-bit flags and their sum. Every continuous per-security value is
# excluded and recomputed at load time from the freely rebuildable caches
# (see fscore.data.team_scores._attach_market_values), so nothing here is a
# data point that could stand in for the source.
KEEP = (["score_year", "fiscal_year", "ticker", "fscore"]
        + [c.lower() for c in SIGNAL_COLS] + ["sector"])
# Continuous values, deliberately not exported: shares_outstanding is a raw
# vendor figure; bm and market_cap are ours but are still per-security
# numbers, so they are regenerated rather than shipped.
DROP = ["shares_outstanding", "bm", "market_cap"]

PROVENANCE = """# Panel provenance

What each column is, where it came from, and why it can be published.

| Column | Source | Nature |
|---|---|---|
| `ticker`, `score_year`, `fiscal_year` | identifiers | not data |
| `f_roa` … `f_dturn` (nine flags) | derived from licensed-terminal fundamentals | **irreversible**: each is one bit, the outcome of a comparison (e.g. ROA > 0). The underlying value cannot be recovered. |
| `fscore` | sum of the nine flags | further aggregation, 0–9 |
| `sector` | Yahoo Finance classification | not vendor data |

**No continuous per-security value is exported.** Three columns that earlier
drafts carried are excluded:

- `shares_outstanding` — a raw vendor figure. Nothing reads it: the EQ_OFFER
  flag was computed inside the source workbook from its own year-on-year
  share columns, so the flag is unaffected by its removal.
- `bm`, `market_cap` — ours rather than the vendor's, but still per-security
  numbers. They are recomputed at load time from the caches that
  `scripts/fetch_us_edgar.py` and `scripts/fetch_us_japan.py` rebuild from
  public sources, so shipping them would add nothing but exposure.

What remains is identifiers plus nine bits and their sum.

## Information content

One firm-year emits nine bits, derived from roughly fifteen continuous line
items across three fiscal years. The mapping is many-to-one: infinitely many
statement sets produce the same nine flags. The panel therefore does not
substitute for, and cannot be used to reconstruct, the source data.

The facts the flags summarise — whether ROA was positive, whether gross
margin improved — are disclosed in public statutory filings (SEC 10-K,
Japanese 有価証券報告書). A reader can recompute every flag from those
filings without any commercial subscription.

## Reproducing the results from this panel

`results/panel/*.csv` plus the caches rebuilt by `scripts/fetch_us_edgar.py`
and `scripts/fetch_us_japan.py` are enough to re-run every notebook. The
study consumes `fscore` for ranking and `sector` for the sector constraint;
book-to-market and market capitalisation are joined from the rebuilt caches
when the panel is loaded, so no continuous value has to travel with it.

## Status of permission

The terminal is not accessible to the team, so a written confirmation cannot
be obtained. The panel was therefore reduced to the narrowest form that still
allows the study to be reproduced: identifiers and irreversible one-bit
flags, with every continuous value regenerated from public sources at run
time. The United States panel can additionally be rebuilt end to end from SEC
EDGAR alone (`scripts/fetch_us_edgar.py`), with no vendor input of any kind.
"""


def export(market: str) -> pd.DataFrame:
    panel = load_team_scores(market, ROOT / "data")
    missing = [c for c in KEEP if c not in panel.columns]
    if missing:
        raise KeyError(f"{market}: expected columns absent: {missing}")
    out = panel[KEEP].copy()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{market}_fscore_panel.csv"
    out.to_csv(path, index=False)
    dropped = [c for c in DROP if c in panel.columns]
    print(f"{market}: {len(out)} rows x {len(out.columns)} cols -> {path.name}")
    print(f"  dropped raw vendor columns: {dropped or 'none present'}")
    print(f"  years {int(out.score_year.min())}–{int(out.score_year.max())}, "
          f"{out.ticker.nunique()} tickers")
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    for m in (["us", "japan"] if which == "both" else [which]):
        export(m)
    (OUT / "PROVENANCE.md").write_text(PROVENANCE, encoding="utf-8")
    print(f"\nwrote {OUT / 'PROVENANCE.md'}")
