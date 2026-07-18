"""Signal layer: the nine Piotroski (2000) binary signals and composite score.

Input: canonical `fundamentals` frame containing the scoring year (t) and the
prior year (t-1) for each ticker. Output: one row per ticker with the nine
0/1 signals and `fscore` (0..9).
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
    """Compute the nine signals for scoring year `year` (needs year-1 rows)."""
    f = fundamentals
    t = f[f.fiscal_year == year].set_index("ticker")
    tm1 = f[f.fiscal_year == year - 1].set_index("ticker")
    common = t.index.intersection(tm1.index)
    t, tm1 = t.loc[common], tm1.loc[common]

    # a missing input must exclude the firm, not silently score 0
    # (NaN comparisons evaluate False): require both years complete
    needed = ["total_assets", "net_income", "cfo", "long_term_debt",
              "current_assets", "current_liabilities", "shares_outstanding",
              "revenue", "cogs"]
    complete = t[needed].notna().all(axis=1) & tm1[needed].notna().all(axis=1)
    t, tm1, common = t[complete], tm1[complete], common[complete]

    # denominators: average assets, guarding zeros
    avg_assets = (t.total_assets + tm1.total_assets) / 2
    roa_t = t.net_income / avg_assets
    roa_tm1 = tm1.net_income / tm1.total_assets  # simple prior-year ROA

    out = pd.DataFrame(index=common)
    # -- profitability
    out["roa_pos"] = (roa_t > 0).astype(int)
    out["cfo_pos"] = (t.cfo > 0).astype(int)
    out["delta_roa_pos"] = (roa_t > roa_tm1).astype(int)
    out["accruals_ok"] = (t.cfo / avg_assets > roa_t).astype(int)  # CFO > NI (scaled)
    # -- leverage / liquidity / dilution
    out["delta_leverage_down"] = (
        t.long_term_debt / avg_assets < tm1.long_term_debt / tm1.total_assets
    ).astype(int)
    curr_t = t.current_assets / t.current_liabilities
    curr_tm1 = tm1.current_assets / tm1.current_liabilities
    out["delta_liquidity_up"] = (curr_t > curr_tm1).astype(int)
    out["no_issuance"] = (t.shares_outstanding <= tm1.shares_outstanding).astype(int)
    # -- operating efficiency
    gm_t = (t.revenue - t.cogs) / t.revenue
    gm_tm1 = (tm1.revenue - tm1.cogs) / tm1.revenue
    out["delta_margin_up"] = (gm_t > gm_tm1).astype(int)
    turn_t = t.revenue / avg_assets
    turn_tm1 = tm1.revenue / tm1.total_assets
    out["delta_turnover_up"] = (turn_t > turn_tm1).astype(int)

    out["fscore"] = out[SIGNALS].sum(axis=1)
    return out.reset_index().rename(columns={"index": "ticker"})
