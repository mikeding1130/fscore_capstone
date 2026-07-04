"""Data layer: loading, point-in-time alignment, and universe construction.

Real data sources plug in here. Every downstream module consumes only the
two canonical DataFrames defined below, so swapping a source never touches
signal/selection/construction/evaluation code.

Canonical schemas
-----------------
fundamentals : one row per (ticker, fiscal_year)
    columns = [ticker, fiscal_year, report_date, total_assets, net_income,
               cfo, long_term_debt, current_assets, current_liabilities,
               shares_outstanding, revenue, cogs, book_value, market_cap]

prices : one row per (ticker, date)
    columns = [ticker, date, adj_close, volume]
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_FUND_COLS = [
    "ticker", "fiscal_year", "report_date", "total_assets", "net_income",
    "cfo", "long_term_debt", "current_assets", "current_liabilities",
    "shares_outstanding", "revenue", "cogs", "book_value", "market_cap",
]


def load_fundamentals(market: str, path: str | None = None) -> pd.DataFrame:
    """Load annual fundamentals for a market into the canonical schema.

    TODO(data-team): implement per-market adapters.
      - US:      e.g. Compustat/WRDS export, or SEC-derived set
      - Japan:   e.g. J-Quants / Refinitiv export
      - Malaysia: source under evaluation (window 2019-12..2025-12)
      - Vietnam: adapter over the team-maintained normalized database
    """
    raise NotImplementedError(f"fundamentals adapter for {market!r} not wired yet")


def load_prices(market: str, path: str | None = None) -> pd.DataFrame:
    """Load adjusted daily prices into the canonical schema. TODO: as above."""
    raise NotImplementedError(f"price adapter for {market!r} not wired yet")


def apply_reporting_lag(fundamentals: pd.DataFrame, lag_months: int = 5) -> pd.DataFrame:
    """Point-in-time discipline: fundamentals become usable only after
    `report_date + lag_months`. Adds an `available_date` column; portfolio
    formation must filter on it (never on fiscal_year alone)."""
    out = fundamentals.copy()
    out["available_date"] = pd.to_datetime(out["report_date"]) + pd.DateOffset(months=lag_months)
    return out


def high_bm_subset(universe: pd.DataFrame, quantile: float = 0.2) -> pd.DataFrame:
    """Cornerstone: keep the high book-to-market slice (top `quantile` of B/M).

    Piotroski (2000) applies the F-Score *within* the value universe; scoring
    the whole market would test a different question.
    """
    u = universe.copy()
    u["bm"] = u["book_value"] / u["market_cap"]
    cutoff = u["bm"].quantile(1 - quantile)
    return u[u["bm"] >= cutoff].reset_index(drop=True)


# ----------------------------------------------------------------------
# Synthetic demo data — clearly labeled; exists so the pipeline is runnable
# end-to-end before real adapters land. Not for any empirical claim.
# ----------------------------------------------------------------------

def make_demo_market(n_stocks: int = 100, year: int = 2023, seed: int = 7,
                     prefix: str = "DEMO") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a synthetic (fundamentals, prices) pair in canonical schema."""
    rng = np.random.default_rng(seed)
    tickers = [f"{prefix}{i:03d}" for i in range(n_stocks)]

    def one_year(fy: int) -> pd.DataFrame:
        assets = rng.lognormal(6.0, 1.0, n_stocks) * 1e6
        ni = assets * rng.normal(0.03, 0.08, n_stocks)
        return pd.DataFrame({
            "ticker": tickers, "fiscal_year": fy,
            "report_date": pd.Timestamp(f"{fy + 1}-03-31"),
            "total_assets": assets, "net_income": ni,
            "cfo": ni + assets * rng.normal(0.01, 0.05, n_stocks),
            "long_term_debt": assets * rng.uniform(0.0, 0.5, n_stocks),
            "current_assets": assets * rng.uniform(0.2, 0.6, n_stocks),
            "current_liabilities": assets * rng.uniform(0.1, 0.5, n_stocks),
            "shares_outstanding": rng.lognormal(4, 0.8, n_stocks) * 1e6,
            "revenue": assets * rng.uniform(0.4, 1.4, n_stocks),
            "cogs": assets * rng.uniform(0.2, 1.0, n_stocks),
            "book_value": assets * rng.uniform(0.2, 0.8, n_stocks),
            "market_cap": assets * rng.lognormal(-0.3, 0.7, n_stocks),
        })

    fundamentals = pd.concat([one_year(year - 1), one_year(year)], ignore_index=True)

    dates = pd.bdate_range(f"{year}-01-01", f"{year + 1}-12-31")
    rets = rng.normal(0.0003, 0.02, (len(dates), n_stocks))
    px = 100 * np.exp(np.cumsum(rets, axis=0))
    prices = (pd.DataFrame(px, index=dates, columns=tickers)
              .rename_axis("date").reset_index()
              .melt(id_vars="date", var_name="ticker", value_name="adj_close"))
    prices["volume"] = rng.integers(1e4, 1e7, len(prices))
    return fundamentals, prices
