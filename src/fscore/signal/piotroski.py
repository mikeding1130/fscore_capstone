"""Signal layer: the nine Piotroski (2000) binary signals and composite score.

Input: canonical `fundamentals` frame containing the scoring year (t) and the
prior year (t-1) for each ticker. Output: one row per ticker with the nine
0/1 signals and `fscore` (0..9).

EQ_OFFER (`no_issuance`) is measured from the **cash-flow statement's
equity-issuance line** (`equity_issued` — proceeds from issuing common /
preferred stock, the analogue of Piotroski's Compustat SSTK), not from the
year-on-year share count. Share counts move for reasons that are not equity
offerings — buybacks net against issuance, splits and stock dividends change
the count without raising capital, and employee-plan vesting drifts it up —
so the share-count test both misses real offerings and flags non-offerings.
It remains as a fallback only where the cash-flow line is absent, and the
split between the two is reported in `.attrs`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SIGNALS = [
    "roa_pos", "cfo_pos", "delta_roa_pos", "accruals_ok",      # profitability
    "delta_leverage_down", "delta_liquidity_up", "no_issuance",  # leverage/liquidity
    "delta_margin_up", "delta_turnover_up",                     # efficiency
]


def piotroski_signals(fundamentals: pd.DataFrame, year: int) -> pd.DataFrame:
    """Compute the nine signals for scoring year `year`.

    Needs rows for years t and t-1, and — for the year-on-year deltas scaled
    by beginning-of-year assets — the total assets reported for t-2. A firm
    without the t-2 row still scores, falling back to period-end assets for
    the prior-year ratio; the count is reported in `.attrs`.
    """
    f = fundamentals
    t = f[f.fiscal_year == year].set_index("ticker")
    tm1 = f[f.fiscal_year == year - 1].set_index("ticker")
    tm2_assets = (f[f.fiscal_year == year - 2]
                  .set_index("ticker")["total_assets"])
    common = t.index.intersection(tm1.index)
    t, tm1 = t.loc[common], tm1.loc[common]

    # a missing input must exclude the firm, not silently score 0
    # (NaN comparisons evaluate False): require both years complete, so every
    # score reported is a full nine-signal score. The count of firms dropped
    # this way is attached as `.attrs["dropped_incomplete"]` for the write-up.
    needed = ["total_assets", "net_income", "cfo", "long_term_debt",
              "current_assets", "current_liabilities", "shares_outstanding",
              "revenue", "cogs"]
    complete = t[needed].notna().all(axis=1) & tm1[needed].notna().all(axis=1)
    n_dropped = int((~complete).sum())
    t, tm1, common = t[complete], tm1[complete], common[complete]

    # EQ_OFFER source: the cash-flow statement's equity-issuance line is the
    # primary measure (Piotroski's Compustat SSTK analogue). The share-count
    # comparison is only a fallback where that line is unavailable, and how
    # often it was used is reported rather than hidden.
    has_cf_issue = (t["equity_issued"].notna() if "equity_issued" in t.columns
                    else pd.Series(False, index=t.index))

    # Denominators follow Piotroski (2000): ratios are scaled by
    # BEGINNING-of-year total assets, and the year-on-year deltas compare two
    # ratios built the same way. Scaling the current year by average assets
    # while the prior year used period-end assets — as an earlier version of
    # this file did — puts a change of denominator convention inside the
    # difference, so ΔROA and Δturnover then measure partly that rather than
    # the operating change they are supposed to capture.
    beg = tm1.total_assets                                  # assets at t-1
    beg_prior = tm2_assets.reindex(common)                   # assets at t-2
    n_no_tm2 = int(beg_prior.isna().sum())
    beg_prior = beg_prior.fillna(tm1.total_assets)           # documented fallback
    avg = (t.total_assets + tm1.total_assets) / 2
    avg_prior = (tm1.total_assets + beg_prior) / 2

    roa_t = t.net_income / beg
    roa_tm1 = tm1.net_income / beg_prior

    out = pd.DataFrame(index=common)
    # -- profitability
    out["roa_pos"] = (roa_t > 0).astype(int)
    out["cfo_pos"] = (t.cfo > 0).astype(int)
    out["delta_roa_pos"] = (roa_t > roa_tm1).astype(int)
    out["accruals_ok"] = (t.cfo / beg > roa_t).astype(int)   # CFO > NI, same scaling
    # -- leverage / liquidity / dilution
    # leverage uses average assets in the original, on both sides of the delta
    out["delta_leverage_down"] = (
        t.long_term_debt / avg < tm1.long_term_debt / avg_prior
    ).astype(int)
    curr_t = t.current_assets / t.current_liabilities
    curr_tm1 = tm1.current_assets / tm1.current_liabilities
    out["delta_liquidity_up"] = (curr_t > curr_tm1).astype(int)
    # no common equity issued during the year: cash raised from issuing
    # equity is zero (a tiny positive figure is employee-plan noise, so the
    # test is <= 0 on the reported proceeds, matching Piotroski's "did not
    # issue common equity" condition)
    by_shares = (t.shares_outstanding <= tm1.shares_outstanding).astype(int)
    if has_cf_issue.any():
        by_cash = (t["equity_issued"].fillna(0) <= 0).astype(int)
        out["no_issuance"] = by_cash.where(has_cf_issue, by_shares)
    else:
        out["no_issuance"] = by_shares
    # -- operating efficiency
    gm_t = (t.revenue - t.cogs) / t.revenue
    gm_tm1 = (tm1.revenue - tm1.cogs) / tm1.revenue
    out["delta_margin_up"] = (gm_t > gm_tm1).astype(int)
    turn_t = t.revenue / beg
    turn_tm1 = tm1.revenue / beg_prior
    out["delta_turnover_up"] = (turn_t > turn_tm1).astype(int)

    out["fscore"] = out[SIGNALS].sum(axis=1)
    out = out.reset_index().rename(columns={"index": "ticker"})
    out.attrs["dropped_incomplete"] = n_dropped
    out.attrs["scored"] = len(out)
    out.attrs["eq_offer_from_cashflow"] = int(has_cf_issue.sum())
    out.attrs["eq_offer_from_shares"] = int((~has_cf_issue).sum())
    out.attrs["no_tm2_assets"] = n_no_tm2
    return out
