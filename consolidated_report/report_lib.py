"""Shared helpers for `consolidated_report.ipynb`.

The notebook composes one publication-ready table per PLACEHOLDER in the
capstone draft. This module holds everything that would otherwise be repeated
in every cell: column/row label maps, number formatting, the significance
convention, and the three-way emitter (notebook HTML + LaTeX + CSV).

Conventions frozen by the pipeline and mirrored here — see
`fscore.evaluation.backtest`:

  * ALPHA = 0.05, one-sided empirical p-value, verdict is ``p < ALPHA``
    (strict, so p = 0.050 is NOT significant).
  * Sharpe ratio uses rf = 0 (`metrics(..., rf_annual=0.0)`).
  * Maximum drawdown is stored as a NEGATIVE return.
  * Turnover is one-way: sum |w_new - w_prev| / 2.
"""
from __future__ import annotations

import html
import pathlib
import re
import shutil

import pandas as pd
from IPython.display import HTML, Markdown, display

# ----------------------------------------------------------------------
# paths
# ----------------------------------------------------------------------
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
GRID = RESULTS / "grid"
FIGSRC = RESULTS / "figures"
RES = HERE / "resources"
TABLES = RES / "tables"          # paper tables, pulled into the LaTeX build
FIGURES = RES / "figures"        # paper figures
NOTES = RES / "notes"            # review tables, deliberately NOT in the paper
for _d in (TABLES, FIGURES, NOTES):
    _d.mkdir(parents=True, exist_ok=True)

ALPHA = 0.05
# a cell longer than this many characters makes its column a wrapping p{} column
LONG_CELL = 40
MARKETS = ["us", "japan", "vietnam"]

# ----------------------------------------------------------------------
# label maps — variable name -> human-readable label used in the paper
# ----------------------------------------------------------------------
MARKET_LABEL = {"us": "United States", "japan": "Japan", "vietnam": "Vietnam"}

STRATEGY_LABEL = {
    "fscore_EW": "F-Score, equal-weight",
    "fscore_GMV": "F-Score, GMV (RMT-denoised)",
    "fscore_GMVsec": "F-Score, sector-capped GMV",
    "fscore_LS": "F-Score, long-short spread",
    "fscore_high_EW": "F-Score $\\geq$ 8, equal-weight",
    "value_EW": "Highest B/M, equal-weight",
    "mktcap_EW": "Largest market cap, equal-weight",
    "liquidity_EW": "Liquidity-matched, equal-weight",
    "universe_EW": "Whole universe, equal-weight",
    "universe_GMV": "Whole universe, GMV",
    "random_EW": "Matched random, equal-weight",
    "random_GMV": "Matched random, GMV",
    "random_GMVsec": "Matched random, sector-capped GMV",
}

METRIC_LABEL = {
    "ann_return": "Annualised return",
    "ann_vol": "Annualised volatility",
    "sharpe": "Sharpe ratio",
    "max_drawdown": "Maximum drawdown",
    "net_ann_return": "Net annualised return",
    "net_sharpe": "Net Sharpe ratio",
    "turnover": "Turnover (one-way)",
    "cost_drag": "Cost drag",
    "nominal_k": "Nominal basket size",
    "effective_n": "Effective N",
    "percentile": "MC percentile",
    "p_value": "p-value",
}

DIAG_LABEL = {
    "year": "Formation year",
    "universe": "Eligible universe",
    "value_set": "High-B/M set",
    "scored": "Scoreable",
    "k": "Selected K",
    "dropped_incomplete_signals": "Dropped: incomplete signals",
    "tie_break_slots": "Tie-break slots",
    "cov_est_days": "Covariance sessions",
    "delisted_in_holding_year": "Delisted in holding year",
    "fscore_mean": "Mean F-Score",
    "fscore_basket_min": "Min F-Score in basket",
}

# Figures the draft asks for: (market, source file, destination, what it shows).
# `consolidated_report` copies them; `consolidated_report_notes` audits which
# ones the repository actually produces.
FIG_PLAN = [
    ("us", "us_nav_vs_benchmarks.png", "fig_7_1a_us_cumulative.png", "Cumulative performance"),
    ("us", "us_mc_placement.png", "fig_7_1b_us_mc_null.png", "Monte Carlo null distribution"),
    ("us", "us_fscore_distribution.png", "fig_7_1c_us_fscore_distribution.png", "F-Score distribution"),
    ("us", "us_fullperiod_nav.png", "fig_7_1d_us_cumulative_full.png", "Cumulative performance, full window"),
    ("japan", "japan_nav_vs_benchmarks.png", "fig_7_2a_japan_cumulative.png", "Cumulative performance"),
    ("japan", "japan_mc_placement.png", "fig_7_2b_japan_mc_null.png", "Monte Carlo null distribution"),
    ("japan", "japan_fscore_distribution.png", "fig_7_2c_japan_fscore_distribution.png", "F-Score distribution"),
    ("japan", "japan_fullperiod_nav.png", "fig_7_2d_japan_cumulative_full.png", "Cumulative performance, full window"),
    ("vietnam", "vietnam_nav_vs_benchmarks.png", "fig_7_3a_vietnam_cumulative.png", "Cumulative performance"),
    ("vietnam", "vietnam_mc_placement.png", "fig_7_3b_vietnam_mc_null.png", "Monte Carlo null distribution"),
    ("vietnam", "vietnam_fscore_distribution.png", "fig_7_3c_vietnam_fscore_distribution.png", "F-Score distribution"),
    ("vietnam", "vietnam_fullperiod_nav.png", "fig_7_3d_vietnam_cumulative_full.png", "Cumulative performance, full window"),
    ("us", "us_grid_sensitivity.png", "fig_9_1a_us_grid_sensitivity.png", "Basket-size sensitivity"),
    ("japan", "japan_grid_sensitivity.png", "fig_9_1b_japan_grid_sensitivity.png", "Basket-size sensitivity"),
    ("vietnam", "vietnam_grid_sensitivity.png", "fig_9_1c_vietnam_grid_sensitivity.png", "Basket-size sensitivity"),
]
FIG_PLACEHOLDER = {"us": "P17", "japan": "P20", "vietnam": "P23"}

# Figures §7 asks for that no script in the repository produces.
FIG_NOT_PRODUCED = ["Drawdown", "Annual GMV weights"]

# sentinels: kept out of the data so formatting survives a round-trip
B0, B1 = "\x01", "\x02"
DASH = "—"


# ----------------------------------------------------------------------
# number formatting
# ----------------------------------------------------------------------
def pct(x, dp: int = 2) -> str:
    """Percentage with an explicit sign convention preserved (MDD stays negative)."""
    return DASH if pd.isna(x) else f"{100 * float(x):.{dp}f}%"


def num(x, dp: int = 3) -> str:
    return DASH if pd.isna(x) else f"{float(x):.{dp}f}"


def integer(x) -> str:
    return DASH if pd.isna(x) else f"{int(round(float(x))):d}"


def pm(mean, sd, formatter=num) -> str:
    """`mean ± sd`, used for the matched-random rows."""
    if pd.isna(mean):
        return DASH
    return f"{formatter(mean)} $\\pm$ {formatter(sd)}" if not pd.isna(sd) else formatter(mean)


def sig(text: str, significant: bool) -> str:
    """Bold + trailing asterisk when significant at ALPHA."""
    return f"{B0}{text}{B1}*" if significant else text


# Significance stars. The study froze ONE level (ALPHA = 5%) in
# `fscore.evaluation.backtest`, and its docstring says no other threshold is
# reported or implied, so that is the default here.
#
# To match the three-level convention common in the published F-Score papers
# (Ng and Shen 2016, Hyde 2018), replace the list with:
#
#     STAR_LEVELS = [(0.01, "***"), (0.05, "**"), (0.10, "*")]
#
# and say so in the note under every table. Be aware this contradicts §6 of
# the draft, which fixes a single 5% level in advance.
STAR_LEVELS: list[tuple[float, str]] = [(0.05, "*")]


def stars(p) -> str:
    """The star suffix earned by an empirical p-value."""
    if pd.isna(p):
        return ""
    for level, mark in STAR_LEVELS:
        if float(p) < level:
            return mark
    return ""


def marked(text: str, p) -> str:
    """A statistic bolded and starred according to its own p-value.

    Used so every tested number carries its own verdict, rather than the
    whole row inheriting the Sharpe ratio's.
    """
    star = stars(p)
    return f"{B0}{text}{B1}{star}" if star else text


def pval(p, dp: int = 3) -> str:
    """A p-value, bolded and starred on its own value.

    Routed through `marked` so that changing STAR_LEVELS changes every starred
    number in the report, p-values included.
    """
    if pd.isna(p):
        return DASH
    return marked(f"{float(p):.{dp}f}", p)


# ----------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------
MISSING: list[str] = []


def read(path: pathlib.Path, **kw) -> pd.DataFrame | None:
    """Read a results CSV, recording (rather than raising on) an absent file."""
    if not path.exists():
        MISSING.append(str(path.relative_to(ROOT)))
        return None
    return pd.read_csv(path, **kw)


def read_indexed(path: pathlib.Path) -> pd.DataFrame | None:
    """Results tables whose first column is an unnamed row label."""
    df = read(path)
    if df is None:
        return None
    return df.rename(columns={df.columns[0]: "row"}).set_index("row")


def read_placement(path: pathlib.Path) -> pd.DataFrame | None:
    """`*_mc_placement.csv`: (strategy, metric) -> MC placement."""
    df = read(path)
    if df is None:
        return None
    df = df.rename(columns={df.columns[0]: "strategy", df.columns[1]: "metric"})
    return df.set_index(["strategy", "metric"])


# ----------------------------------------------------------------------
# emit: one composed table -> notebook + LaTeX + CSV
# ----------------------------------------------------------------------
MANIFEST: list[dict] = []


def _to_html(s: str) -> str:
    s = html.escape(str(s))
    s = s.replace(B0, "<b>").replace(B1, "</b>")
    s = s.replace("$\\pm$", "&plusmn;").replace("$\\geq$", "&ge;")
    s = re.sub(r"\$\^\{(.*?)\}\$", r"<sup>\1</sup>", s)
    return s


_TEX_ESCAPE = {"%": r"\%", "&": r"\&", "_": r"\_", "#": r"\#"}


def _to_tex(s: str) -> str:
    s = str(s)
    keep = {}
    for i, m in enumerate(re.findall(r"\$[^$]*\$", s)):      # protect math
        key = f"@@M{i}@@"
        keep[key] = m
        s = s.replace(m, key, 1)
    for ch, esc in _TEX_ESCAPE.items():
        s = s.replace(ch, esc)
    # file names and identifiers have no hyphenation points, so a long one
    # overflows its column; allow a line break after each underscore and slash
    s = s.replace(r"\_", r"\_\allowbreak{}").replace("/", r"/\allowbreak{}")
    for key, m in keep.items():
        s = s.replace(key, m)
    s = s.replace(B0, r"\textbf{").replace(B1, "}")
    # significance stars: one, two or three, depending on STAR_LEVELS
    m = re.search(r"(\*+)$", s)
    if m and r"\textbf{" in s:
        s = s[:m.start()] + rf"$^{{{m.group(1)}}}$"
    # an em dash as a literal byte needs inputenc/fontspec; the control
    # sequence renders under any engine
    s = s.replace(DASH, "---")
    return s


def _to_plain(s: str) -> str:
    return (str(s).replace(B0, "").replace(B1, "")
            .replace("$\\pm$", "+/-").replace("$\\geq$", ">="))


def _column_spec(df: pd.DataFrame, idx_name: str) -> tuple[str, bool]:
    """Choose a tabular preamble, and say whether the table needs scaling.

    Three cases:

      * every cell is short — right-aligned columns, scaled to the text width
        when there are six or more of them;
      * short cells but a long row label — the label gets a fixed `p{}` column
        and the table is scaled, which cannot overflow;
      * cells carrying sentences — every column becomes a `p{}` whose width is
        proportional to the square root of its content length (the square root
        keeps one very wide column from starving the others). Widths are
        expressed as ``\dimexpr ... - 2\tabcolsep`` so the inter-column padding
        is subtracted exactly and the row fits whatever ``\textwidth`` is.

    Requires the `array` package for `\arraybackslash`, and `graphicx` for the
    scaled case.
    """
    import math

    def measure(values) -> int:
        return max([len(_to_plain(str(v))) for v in values] + [1])

    idx_w = max(measure(df.index), len(str(idx_name)))
    col_w = [max(measure(df.iloc[:, i]), len(str(c)))
             for i, c in enumerate(df.columns)]
    long_cols = [i for i, w in enumerate(col_w) if w > LONG_CELL]

    def para(share: float) -> str:
        return (r">{\raggedright\arraybackslash}"
                rf"p{{\dimexpr {share:.4f}\textwidth-2\tabcolsep\relax}}")

    if not long_cols:
        if idx_w > LONG_CELL:                       # long label, short columns
            return para(0.22) + "r" * len(df.columns), True
        return "l" + "r" * len(df.columns), len(df.columns) >= 6

    weights = [math.sqrt(w) for w in [idx_w] + col_w]
    total = sum(weights)
    return "".join(para(w / total) for w in weights), False


def emit(df: pd.DataFrame, *, name: str, number: str, title: str,
         placeholder: str, section: str, notes: str = "",
         index_header: str = "", outdir: pathlib.Path | None = None) -> pd.DataFrame:
    """Render one composed table in the notebook and save it to `resources/`.

    `df` holds strings already formatted by the helpers above; significance is
    carried by the B0/B1 sentinels, so all three renderings stay consistent.

    Writes `<outdir>/<name>.tex` (booktabs, ready to \\input) and
    `<outdir>/<name>.csv` (plain text, for spot-checking in a sheet). `outdir`
    defaults to `resources/tables/`; the review notebook passes `NOTES` so its
    tables never reach the LaTeX build.
    """
    outdir = TABLES if outdir is None else outdir
    caption = f"{title}"
    label = f"tab:{name}"
    idx_name = index_header or (df.index.name or "")

    # ---- notebook ----
    head = "".join(f"<th style='text-align:right;padding:4px 10px'>{_to_html(c)}</th>"
                   for c in df.columns)
    body = ""
    for ix, row in df.iterrows():
        cells = "".join(f"<td style='text-align:right;padding:4px 10px'>{_to_html(v)}</td>"
                        for v in row)
        body += (f"<tr><td style='text-align:left;padding:4px 10px'>"
                 f"<b>{_to_html(ix)}</b></td>{cells}</tr>")
    display(Markdown(f"**{number}. {title}**  \n"
                     f"<sub>{section} &middot; fills {placeholder} &middot; "
                     f"`{outdir.relative_to(HERE)}/{name}.tex`</sub>"))
    display(HTML(
        "<table style='border-collapse:collapse;font-size:13px;margin:6px 0'>"
        f"<thead><tr><th style='text-align:left;padding:4px 10px'>{_to_html(idx_name)}</th>"
        f"{head}</tr></thead><tbody>{body}</tbody></table>"))
    if notes:
        display(Markdown(f"<sub>*Notes.* {notes}</sub>"))

    # ---- LaTeX ----
    align, wide = _column_spec(df, idx_name)
    lines = [r"% auto-generated by consolidated_report.ipynb - do not edit by hand",
             r"\begin{table}[htbp]", r"\centering",
             rf"\caption{{{_to_tex(caption)}}}", rf"\label{{{label}}}",
             r"\small"]
    if wide:
        # needs graphicx; keeps a nine-column table inside \textwidth
        lines.append(r"\resizebox{\textwidth}{!}{%")
    lines += [rf"\begin{{tabular}}{{{align}}}", r"\toprule",
             " & ".join([_to_tex(idx_name)] + [_to_tex(c) for c in df.columns]) + r" \\",
             r"\midrule"]
    for ix, row in df.iterrows():
        lines.append(" & ".join([_to_tex(ix)] + [_to_tex(v) for v in row]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if wide:
        lines.append("}")
    if notes:
        lines += [r"\begin{minipage}{\linewidth}\vspace{4pt}\footnotesize",
                  _to_tex(notes), r"\end{minipage}"]
    lines += [r"\end{table}", ""]
    tex = outdir / f"{name}.tex"
    tex.write_text("\n".join(lines), encoding="utf-8")

    # ---- CSV ----
    plain = pd.DataFrame(
        [[_to_plain(v) for v in row] for _, row in df.iterrows()],
        index=[_to_plain(i) for i in df.index],
        columns=[_to_plain(c) for c in df.columns])
    plain.index.name = idx_name or None
    csv = outdir / f"{name}.csv"
    plain.to_csv(csv, encoding="utf-8")

    MANIFEST.append({"Table": number, "Title": title, "Section": section,
                     "Placeholder": placeholder,
                     "File": f"{outdir.relative_to(RES)}/{name}.tex"})
    return df


def copy_figure(src_name: str, dest_name: str, *, number: str, title: str,
                placeholder: str, section: str) -> bool:
    """Copy one results figure into `resources/figures/` under a paper-ready name."""
    src = FIGSRC / src_name
    if not src.exists():
        MISSING.append(str(src.relative_to(ROOT)))
        return False
    shutil.copyfile(src, FIGURES / dest_name)
    MANIFEST.append({"Table": number, "Title": title, "Section": section,
                     "Placeholder": placeholder, "File": f"figures/{dest_name}"})
    return True
