"""Figure defaults and saving, so every chart in the study looks the same.

One place decides resolution and styling; notebooks call `setup_plots()` once
and `save_fig(name)` for each chart, instead of repeating `dpi=` at every
call site.

Resolution: **every chart written to disk is 300 dpi** — those PNGs are what
goes into the reports. The inline copy embedded in the notebook is rendered
at `DISPLAY_DPI` (110) on purpose: an .ipynb stores its figures as embedded
base64 PNGs, so rendering the inline copies at 300 dpi too would multiply
every notebook's size by roughly nine (the 20 notebooks' figures would go
from ~13 MB to >100 MB in git) for pixels no one reads on screen. Pass
`dpi=` to `save_fig` only if a single chart genuinely needs something else.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

DPI = 300          # every saved figure
DISPLAY_DPI = 110  # inline copy only, keeps notebook files small
FIG_DIR = "results/figures"


def setup_plots(display_dpi: int = DISPLAY_DPI, dpi: int = DPI) -> None:
    """Apply the study-wide figure defaults. Call once per notebook."""
    mpl.rcParams.update({
        "figure.dpi": display_dpi,
        "savefig.dpi": dpi,          # any bare plt.savefig() is 300 dpi too
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_fig(name: str, fig=None, directory: str | Path = FIG_DIR,
             dpi: int = DPI) -> Path:
    """Save the current (or given) figure as a 300 dpi PNG under `directory`."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    path = d / (name if name.endswith(".png") else f"{name}.png")
    (fig or plt.gcf()).savefig(path, dpi=dpi, bbox_inches="tight")
    return path
