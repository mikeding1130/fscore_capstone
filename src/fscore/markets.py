"""Per-market trading constraints that change which strategies are testable.

The study covers three markets — the US and Japan (developed) and Vietnam
(emerging); Malaysia was dropped from scope when a usable data source could
not be secured. They do not offer the same instrument set, so a strategy that
is implementable in one is not automatically implementable in another. The long-short F-Score portfolio (long high scores, short low
scores) is the clearest case: it can only be run where an investor can
actually borrow and sell the low-score names.

  * US, Japan — established stock-borrow markets; long-short is testable.
  * Vietnam   — short selling of ordinary shares is not available to
                investors on HOSE/HNX; only long positions (and derivatives
                outside this study's scope) are possible. The pipeline
                therefore runs Vietnam LONG-ONLY: the long-short strategy is
                skipped rather than reported as a hypothetical.
Markets outside the study, and any unknown market, default to long-only — the conservative direction, since
claiming a short leg that cannot be traded would overstate the results.
"""
from __future__ import annotations

SHORTING_ALLOWED: dict[str, bool] = {
    "us": True,
    "united states": True,
    "japan": True,
    "jp": True,
    "vietnam": False,
    "vn": False,
}

# Annual stock-borrow fee charged on the short leg's notional. General
# collateral large caps sit well below this; it is deliberately conservative
# so a long-short result never looks good purely because borrowing was free.
SHORT_BORROW_ANNUAL = 0.01


def allows_shorting(market: str) -> bool:
    """Whether a long-short strategy is implementable in `market`."""
    return SHORTING_ALLOWED.get(str(market).lower(), False)
