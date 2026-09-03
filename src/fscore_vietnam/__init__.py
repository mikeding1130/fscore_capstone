"""The Vietnamese preprocessing pipeline: crawl -> reconciled, scored panel.

This is the source of record for every Vietnamese number in the study. It
crawls FireAnt, CafeF and TCBS into `fscore.db`, reconciles the three,
applies accounting checks, computes the nine Piotroski signals, screens for
tradability and writes the panels `fscore_vietnam.schema_adapter` then consumes.

Every location the notebooks read and write comes from `fscore_vietnam.paths`
rather than being spelled out relative to whatever the working directory
happens to be, so the tree lives under this repository and moves as one.

`fscore.signal.piotroski` does NOT score Vietnam — `f_score_calculation`
here does, and the two agree flag for flag on every jointly scored firm-year.
See `README.md` in this directory for the run order.
"""
from . import paths                                   # noqa: F401

__all__ = ["paths"]
