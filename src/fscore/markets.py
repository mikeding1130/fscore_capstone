"""Per-market trading constraints that change which strategies are testable.

The study covers three markets — the US and Japan (developed) and Vietnam
(emerging); Malaysia was dropped from scope when a usable data source could
not be secured. They do not offer the same instrument set, so a strategy that
is implementable in one is not automatically implementable in another. The long-short F-Score portfolio (long high scores, short low
scores) is the clearest case: it can only be *traded* where an investor can
actually borrow and sell the low-score names.

  * US, Japan — established stock-borrow markets; long-short is tradable.
  * Vietnam   — short selling of ordinary shares is **not** available to
                investors on HOSE/HNX. The study nevertheless runs Vietnam
                LONG-SHORT by explicit decision, so the high-minus-low
                F-Score spread is measured in all three markets on the same
                design instead of being missing in exactly the market where
                the selection signal looks strongest. `fscore_LS` is
                therefore a **research construct in Vietnam, not an
                implementable book**: it isolates how much of the result
                comes from the ranking's low end rather than from the market
                beta the long-only lines carry. Read it as a decomposition of
                the signal, never as a portfolio the reader could run. The
                same stock-borrow fee is charged as in the developed pair,
                which if anything understates what borrowing would cost in a
                market that does not offer it.
Markets outside the study, and any unknown market, default to long-only — the conservative direction, since
claiming a short leg that has not been thought about would overstate the results.
"""
from __future__ import annotations

SHORTING_ALLOWED: dict[str, bool] = {
    "us": True,
    "united states": True,
    "japan": True,
    "jp": True,
    # Not tradable on HOSE/HNX — enabled deliberately so the spread is
    # reported as a hypothetical decomposition. See the module docstring and
    # `HYPOTHETICAL_SHORT` below, which is what the notebooks label it with.
    "vietnam": True,
    "vn": True,
}

# Markets where `fscore_LS` runs but could not actually be traded. Every
# report that prints a long-short row for one of these has to say so; keeping
# the list here means one place decides, rather than each notebook's prose.
HYPOTHETICAL_SHORT: frozenset[str] = frozenset({"vietnam", "vn"})

# Annual stock-borrow fee charged on the short leg's notional. General
# collateral large caps sit well below this; it is deliberately conservative
# so a long-short result never looks good purely because borrowing was free.
SHORT_BORROW_ANNUAL = 0.01


def allows_shorting(market: str) -> bool:
    """Whether the long-short strategy is RUN for `market`.

    Not the same question as whether it is tradable there: Vietnam returns
    True and is untradable. `is_hypothetical_short` is the tradability test.
    """
    return SHORTING_ALLOWED.get(str(market).lower(), False)


def is_hypothetical_short(market: str) -> bool:
    """Whether `market`'s long-short book is a hypothetical rather than a
    tradable portfolio — true where shorting is run but not available."""
    return str(market).lower() in HYPOTHETICAL_SHORT
