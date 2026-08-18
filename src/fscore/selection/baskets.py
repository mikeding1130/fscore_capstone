"""Selection layer: fixed-size baskets under competing selection rules.

Basket size is held constant across rules so that performance differences are
attributable to *selection*, not diversification breadth. All rules draw from
the same high-B/M universe; the random rule returns MANY baskets (a Monte
Carlo distribution), never a single draw.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BASKET_SIZE = 30  # frozen-scope default;


def rank_by_fscore(scored: pd.DataFrame, seed: int | None = None) -> pd.DataFrame:
    """Order by F-Score, highest first, **ties broken at random**.

    The F-Score is an integer 0-9, so a 25-name basket drawn from ~70 names
    is roughly one third decided by the tie-break: ~8 slots are filled from
    the ~18 names sitting on the cut-off score. Breaking those ties by B/M
    (the previous rule) let the value factor pick a third of the "F-Score"
    basket, which is precisely what the value control is supposed to test
    against. A seeded random tie-break keeps the basket a function of the
    score alone, and stays reproducible.
    """
    rng = np.random.default_rng(seed)
    shuffled = scored.iloc[rng.permutation(len(scored))]
    return shuffled.sort_values("fscore", ascending=False, kind="stable")


def tie_break_slots(scored: pd.DataFrame, k: int) -> int:
    """How many of the k slots are decided by the tie-break (diagnostic)."""
    if len(scored) < k:
        return 0
    ordered = scored.sort_values("fscore", ascending=False)
    cutoff = ordered["fscore"].iloc[k - 1]
    return int(k - (ordered["fscore"] > cutoff).sum())


def fscore_basket(scored: pd.DataFrame, k: int = BASKET_SIZE,
                  seed: int | None = None) -> list[str]:
    """Top-k by F-Score, ties broken at random — see `rank_by_fscore`."""
    return rank_by_fscore(scored, seed).head(k)["ticker"].tolist()


def random_baskets(universe: list[str], k: int = BASKET_SIZE,
                   n_draws: int = 1000, seed: int = 42) -> list[list[str]]:
    """Monte Carlo control: `n_draws` random baskets from the SAME universe.

    Because the universe is already the high-B/M subset, these baskets carry
    the same value-factor exposure as the F-Score basket — the comparison
    therefore isolates the score's contribution beyond the value tilt.
    """
    rng = np.random.default_rng(seed)
    u = list(universe)
    return [list(rng.choice(u, size=k, replace=False)) for _ in range(n_draws)]


def value_basket(universe_df: pd.DataFrame, k: int = BASKET_SIZE) -> list[str]:
    """Pure value screen: top-k by book-to-market. The 'is F-Score just
    value?' control."""
    return universe_df.sort_values("bm", ascending=False).head(k)["ticker"].tolist()


def mktcap_basket(universe_df: pd.DataFrame, k: int = BASKET_SIZE,
                  largest: bool = True) -> list[str]:
    """Size control: top-k by market cap (guards against a disguised
    small-cap premium)."""
    return (universe_df.sort_values("market_cap", ascending=not largest)
                       .head(k)["ticker"].tolist())


def liquidity_matched_basket(universe_df: pd.DataFrame, target: list[str],
                             liquidity_col: str = "adv",
                             k: int = BASKET_SIZE, seed: int = 0) -> list[str]:
    """Liquidity control: sample non-target names whose liquidity profile
    matches the target basket's (quantile-bucket matching)."""
    rng = np.random.default_rng(seed)
    df = universe_df.copy()
    df["bucket"] = pd.qcut(df[liquidity_col].rank(method="first"), 5, labels=False)
    tgt_counts = df[df.ticker.isin(target)]["bucket"].value_counts()
    pool = df[~df.ticker.isin(target)]
    picks: list[str] = []
    for bucket, cnt in tgt_counts.items():
        cand = pool[pool.bucket == bucket]["ticker"].tolist()
        take = min(cnt, len(cand))
        picks += list(rng.choice(cand, size=take, replace=False))
    # top up if some buckets ran short
    if len(picks) < k:
        rest = pool[~pool.ticker.isin(picks)]["ticker"].tolist()
        take = min(k - len(picks), len(rest))
        if take:
            picks += list(rng.choice(rest, size=take, replace=False))
    if len(picks) < k:
        # universe smaller than 2k: keep the basket size fixed by accepting
        # overlap with the target (target names match its liquidity trivially)
        picks += [t for t in target if t not in picks][:k - len(picks)]
    return picks[:k]
