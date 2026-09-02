"""Render the interaction test as a k x N table per market, one image each.

Matches the layout used in the report: rows are basket size k, columns are the
number of random draws N, and each cell carries D, the random-basket mean of
the same difference, and the p-value.

The N columns are identical by construction, and that is the point of showing
them: the same seed draws the same first 1,000 baskets whether the run asks
for 1,000 or 5,000, so D cannot move with N. Only the p-value can, and only in
its last digits. A reader who sees three identical columns has learned that N
buys resolution, not robustness — and a column that *did* differ would mean
the seeding had broken.

Numbers come from the per-cell synergy files the grid already wrote
(`results/grid/{market}_k{k}_mc{N}_synergy.csv`), so this renders the study's
own output rather than recomputing it.

Run:  python scripts/d_table_figure.py
Writes results/figures/d_interaction_{market}.png and a combined CSV.
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fscore.plotting import save_fig, setup_plots  # noqa: E402

GRID = ROOT / "results" / "grid"
FIGS = ROOT / "results" / "figures"
KS = [20, 25, 30]
NS = [1000, 2000, 5000]
TITLES = {"us": "United States", "japan": "Japan", "vietnam": "Vietnam"}

HEADER_FILL = "#e8edf3"
EDGE = "#333333"


def cell_values(market: str, k: int, n: int) -> dict | None:
    p = GRID / f"{market}_k{k}_mc{n}_synergy.csv"
    if not p.exists():
        return None
    s = pd.read_csv(p, index_col=0)["value"]
    return {"D": float(s["D_fscore"]),
            "random_mean": float(s["D_random_mean"]),
            "p": float(s["p_value"])}


def build(market: str) -> tuple[pd.DataFrame, list[list[str]]]:
    rows, text = [], []
    for k in KS:
        line = []
        for n in NS:
            v = cell_values(market, k, n)
            if v is None:
                line.append("—")
                continue
            line.append(f"D = {v['D']:+.2f}  |  Random mean {v['random_mean']:+.2f}\n"
                        f"(p = {v['p']:.2f})")
            rows.append({"market": market, "k": k, "N": n,
                         "D": round(v["D"], 4),
                         "D_random_mean": round(v["random_mean"], 4),
                         "p_value": round(v["p"], 4),
                         "significant_5pct": v["p"] < 0.05})
        text.append(line)
    return pd.DataFrame(rows), text


def render(market: str, text: list[list[str]]) -> pathlib.Path | None:
    if not any(c != "—" for line in text for c in line):
        return None
    header = [f"{TITLES[market]}:\nOptimization gain D"] + [f"N = {n:,}" for n in NS]
    body = [[f"K = {k}"] + line for k, line in zip(KS, text)]

    fig, ax = plt.subplots(figsize=(11.0, 2.15))
    ax.axis("off")
    tbl = ax.table(cellText=body, colLabels=header, cellLoc="left",
                   colWidths=[0.19, 0.27, 0.27, 0.27], loc="upper left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(EDGE)
        cell.set_linewidth(0.8)
        cell.PAD = 0.045
        if r == 0:
            cell.set_facecolor(HEADER_FILL)
            cell.set_height(0.26)
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_height(0.245)
            if c == 0:
                cell.get_text().set_fontweight("bold")
    ax.set_title("Interaction test: basket optimisation gain "
                 "D = Sharpe(GMV) − Sharpe(EW)",
                 loc="left", fontsize=11, fontweight="bold", pad=14)
    plt.tight_layout()
    out = save_fig(f"d_interaction_{market}", directory=FIGS)
    plt.close(fig)
    return out


def main() -> None:
    setup_plots()
    FIGS.mkdir(parents=True, exist_ok=True)
    frames = []
    for market in TITLES:
        df, text = build(market)
        if df.empty:
            print(f"  {market}: no synergy files - skipped")
            continue
        frames.append(df)
        render(market, text)
        print(f"  {market}: rendered -> results/figures/d_interaction_{market}.png")

    if frames:
        allrows = pd.concat(frames, ignore_index=True)
        allrows.to_csv(ROOT / "results" / "d_interaction_grid.csv", index=False)
        print(f"\n{allrows.to_string(index=False)}")
        same = allrows.groupby(["market", "k"]).D.nunique().eq(1).all()
        print(f"\nD identical across N within every (market, k): {same}"
              "  <- expected; the seed fixes the draws")
        print(f"saved -> {ROOT / 'results' / 'd_interaction_grid.csv'}")


if __name__ == "__main__":
    main()
