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

The flags are computed by this repository's own signal code
(`fscore.signal.piotroski`), so what ships is our output rather than a
vendor's. No continuous per-security value is shipped at all: `bm` and
`market_cap` are ours rather than the vendor's but are still per-security
numbers, so they are recomputed at load time from the rebuildable caches
instead of travelling with the panel.

**The panel is exported from the same statements each market's study reads** —
SEC EDGAR for the US, the Bloomberg tree for Japan — through the same
`build_score_panel` the grid uses. That equality is the point of the
deliverable: a panel exported from some other source would document a study
nobody ran. It is therefore regenerated whenever a market changes source.

Run:  python scripts/export_panel.py [us|japan|both]
Writes results/panel/{market}_fscore_panel.csv and results/panel/PROVENANCE.md
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fscore.data.score_panel import build_score_panel  # noqa: E402
from fscore.data.team_scores import SIGNAL_COLS  # noqa: E402

# Formation years the studies run; a score year y-1 feeds formation y.
YEARS = list(range(2012, 2025))


def load_market_scores(market: str) -> pd.DataFrame:
    """The score panel for `market`, from the source its study actually uses."""
    data = ROOT / "data"
    score_years = [y - 1 for y in YEARS]
    if market == "us":
        from fscore.data.edgar import load_membership
        fund = pd.read_csv(data / "us_fundamentals.csv", parse_dates=["report_date"])
        sectors = pd.read_csv(data / "us_sectors.csv").set_index("ticker")["sector"]
        membership = load_membership(data)
    elif market == "japan":
        from fscore.data.bbg_processed import constituents
        fund = pd.read_csv(data / "japan_bbg_fundamentals.csv",
                           parse_dates=["report_date"])
        sectors = pd.read_csv(data / "japan_sectors.csv").set_index("ticker")["sector"]
        membership = constituents(market, data, YEARS)
    else:
        raise ValueError(f"unknown market {market!r}")
    return build_score_panel(fund, score_years, sectors=sectors,
                             membership=membership)

OUT = ROOT / "results" / "panel"

# Only irreversible content leaves the repository: identifiers, the nine
# one-bit flags and their sum. Every continuous per-security value is
# excluded and recomputed at load time from the freely rebuildable caches, so
# nothing here is a data point that could stand in for the source.
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
| `f_roa` … `f_dturn` (nine flags) | computed here by `fscore.signal.piotroski` | **irreversible**: each is one bit, the outcome of a comparison (e.g. ROA > 0). The underlying value cannot be recovered. |
| `fscore` | sum of the nine flags | further aggregation, 0–9 |
| `sector` | Yahoo Finance classification | not vendor data |

**Statement sources.** The US flags come from SEC EDGAR XBRL — public filings,
no vendor involved, rebuildable end to end by `scripts/fetch_us_edgar.py`. The
Japanese flags come from the Bloomberg statements under `data/processed/Japan/`
(git-ignored, never redistributed), rebuilt into a canonical frame by
`scripts/build_japan_bbg.py`; the facts underneath are the statutory 有価証券
報告書, which any reader can recompute from independently.

**No continuous per-security value is exported.** `bm` and `market_cap` are
ours rather than the vendor's, but they are still per-security numbers, so
they are recomputed at load time from the caches that
`scripts/fetch_us_edgar.py`, `scripts/fetch_us_japan.py` and
`scripts/build_japan_bbg.py` rebuild; shipping them would add nothing but
exposure. Raw statement values never leave the machine at all.

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

`results/panel/*.csv` plus the caches rebuilt by `scripts/fetch_us_edgar.py`,
`scripts/fetch_us_japan.py` and `scripts/build_japan_bbg.py` are enough to
re-run every notebook. The
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
    panel = load_market_scores(market)
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
