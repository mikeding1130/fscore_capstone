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

Raw vendor values are dropped rather than shipped: `shares_outstanding` is a
per-security figure from the workbook and is removed here. Nothing downstream
reads it — the EQ_OFFER flag was computed inside the workbook from its own
year-on-year share columns, so the flag survives the removal intact.

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

# Columns that may leave the repository, and why each one is safe.
KEEP = ["score_year", "fiscal_year", "ticker", "fscore"] + \
       [c.lower() for c in SIGNAL_COLS] + ["bm", "market_cap", "sector"]
# Raw vendor values: never exported.
DROP = ["shares_outstanding"]

PROVENANCE = """# Panel provenance

What each column is, where it came from, and why it can be published.

| Column | Source | Nature |
|---|---|---|
| `ticker`, `score_year`, `fiscal_year` | identifiers | not data |
| `f_roa` … `f_dturn` (nine flags) | derived from licensed-terminal fundamentals | **irreversible**: each is one bit, the outcome of a comparison (e.g. ROA > 0). The underlying value cannot be recovered. |
| `fscore` | sum of the nine flags | further aggregation, 0–9 |
| `bm`, `market_cap` | this project's own Yahoo Finance / SEC EDGAR cache | not vendor data |
| `sector` | Yahoo Finance classification | not vendor data |

Deliberately **not** exported: `shares_outstanding` — a per-security raw
value from the source workbook. It is unused downstream; the EQ_OFFER flag
was computed inside the workbook from its own year-on-year share columns, so
removing it changes no result.

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

`results/panel/*.csv` plus the price caches rebuilt by
`scripts/fetch_us_japan.py` / `scripts/fetch_us_edgar.py` are enough to
re-run every notebook: the study consumes `fscore` for ranking and
`bm` / `market_cap` / `sector` for the controls and constraints.

## Status of permission

The terminal licence treats genuinely derived data more permissively than raw
redistribution, but prior approval is commonly still required. Confirmation
was requested from the vendor; until it is on file, these CSVs are generated
locally and kept out of the repository (see .gitignore).
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
