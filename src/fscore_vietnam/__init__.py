"""The Vietnamese preprocessing pipeline: crawl -> reconciled, scored panel.

This is the source of record for every Vietnamese number in the study. It
crawls FireAnt, CafeF and TCBS into `fscore.db`, reconciles the three,
applies accounting checks, computes the nine Piotroski signals, screens for
tradability and writes the panels `fscore_vietnam.schema_adapter` then consumes.

It lived in a sibling `thesis` repository until this package was created; the
notebooks are unchanged apart from their paths, which now come from
`fscore_vietnam.paths` instead of being spelled out relative to a checkout.

`fscore.signal.piotroski` does NOT score Vietnam — `f_score_calculation`
here does, and the two agree flag for flag on every jointly scored firm-year.
See `README.md` in this directory for the run order.
"""
from . import paths                                   # noqa: F401

__all__ = ["paths"]
