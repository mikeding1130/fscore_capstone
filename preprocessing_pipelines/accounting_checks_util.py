"""
Accounting checks for the widened financial-statement panel.

Columns assumed: owner_equity, minority_interest, paid_in_capital, treasury_stock,
total_assets, long_term_debt, current_assets, current_liabilities,
net_income_parent, gross_profit, net_sales, cfo, stock_issuance_proceeds.
Index: (symbol, period).

SEVERITY
  hard  - a violation means the column is wrong (mapping, sign, unit, or template).
          Drop the row or fix the mapping. Never explain it away.
  soft  - legal but rare. A handful is normal; a cluster means something systematic.
  info  - not a fault. Reports a real state or a storage convention. Its False rows
          are an input to downstream logic, not rows to discard.

FRAMEWORK
  universal - follows from double-entry itself; holds under VAS, IFRS and US GAAP.
  vas       - depends on a Vietnamese reporting convention (charter capital at fixed
              par, the TT200/TT202 vs QD15 layout, NCI treatment). Re-derive before
              applying to an IFRS-reporting market.

Every check returns True (pass) / False (fail) / NA (untestable). NA is never a
failure: missing data is a coverage problem, and a ratio that cannot be formed
(zero denominator) is not evidence of a broken column.

    res = run_checks(panel)
    print(summary(res))
    print(offenders(res, "cl_plus_equity_le_assets", panel).head(20))
    print(failures_by_year(res, "nci_le_equity"))
    panel_clean = panel[clean_mask(res, "hard")]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

REL_TOL = 1e-6          # exact identities
BOUND_TOL = 1e-9        # inequalities, relative to the larger side

UNIVERSAL = "universal"
VAS = "vas"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _na_mask(*series: pd.Series) -> pd.Series:
    """True where any input is missing -> the check is not applicable."""
    m = series[0].isna()
    for s in series[1:]:
        m = m | s.isna()
    return m


def _le(a: pd.Series, b: pd.Series, tol: float = BOUND_TOL) -> pd.Series:
    """a <= b, with relative slack so rounding noise does not fail the check."""
    scale = pd.concat([a.abs(), b.abs()], axis=1).max(axis=1)
    return a <= b + tol * scale


def _eq(a: pd.Series, b: pd.Series, tol: float = REL_TOL) -> pd.Series:
    scale = pd.concat([a.abs(), b.abs()], axis=1).max(axis=1).replace(0, 1.0)
    return (a - b).abs() / scale <= tol


def _eq_by(a: pd.Series, b: pd.Series, scale: pd.Series,
           tol: float = REL_TOL) -> pd.Series:
    """
    a == b, judged against an external scale (normally total assets) rather than
    against |a| and |b| themselves. Needed for identities whose two sides can both
    sit near zero — a cash change of ~0 makes a self-scaled relative tolerance
    absurdly strict, and every quiet year would fail.
    """
    denom = scale.abs().replace(0, np.nan)
    return (a - b).abs() / denom <= tol


def _div(num: pd.Series, den: pd.Series) -> pd.Series:
    """num/den with a zero denominator yielding NaN rather than inf."""
    return num / den.replace(0, np.nan)


def _between(x: pd.Series, lo: float, hi: float) -> pd.Series:
    """
    lo <= x <= hi, with NaN propagating to NA instead of collapsing to False.
    A plain comparison against NaN returns False, which would mark every
    zero-revenue firm-year as a data error rather than as untestable.
    """
    out = ((x >= lo) & (x <= hi)).astype("boolean")
    out[x.isna()] = pd.NA
    return out


def add_lags(panel: pd.DataFrame, cols: list[str] | None = None,
             year_col: str | None = None) -> pd.DataFrame:
    """
    Add <col>_lag for the previous period of the same symbol, plus `lag_contiguous`
    marking rows whose previous period is exactly one year back.

    Cross-period checks are valid only where lag_contiguous is True — a symbol that
    skips a year would otherwise be compared against t-2 with nothing to show for it.
    """
    out = panel.copy()
    cols = cols or [c for c in out.columns if out[c].dtype.kind in "fi"]
    g = out.groupby(level="symbol", sort=False)
    for c in cols:
        out[f"{c}_lag"] = g[c].shift(1)

    if year_col and year_col in out.columns:
        yr = out[year_col].astype("float")
    else:
        yr = pd.Series(out.index.get_level_values("period"), index=out.index)
        yr = yr.astype(str).str[:4].astype("float")
    out["_year"] = yr
    out["lag_contiguous"] = (
        yr - out.groupby(level="symbol", sort=False)["_year"].shift(1)
    ).eq(1)
    return out.drop(columns="_year")


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Check:
    name: str
    severity: str                  # hard | soft | info
    framework: str                 # universal | vas
    inputs: tuple                  # columns required; missing -> NA
    fn: Callable[[pd.DataFrame], pd.Series]
    note: str
    cross_period: bool = False


CHECKS: list[Check] = [

    # ---------------------------------------------------------- balance sheet
    Check(
        "balance_identity", "hard", UNIVERSAL,
        ("total_liabilities", "equity_section", "total_assets"),
        lambda d: _eq(d["total_liabilities"] + d["equity_section"], d["total_assets"]),
        "THE identity: liabilities + equity = assets. The only exact equality in this "
        "rule set - everything else is an inequality or a plausibility bound, because "
        "the other columns cannot pin down an exact relation on their own. Requires "
        "total_liabilities (code 300) and equity_section (code 400), which are NOT in "
        "the 13-column panel; until they are pulled this check reports NA for every row. "
        "Two extra fields in the query buy the strongest test available.",
    ),
    Check(
        "gross_profit_decomposes", "hard", UNIVERSAL,
        ("gross_profit", "net_sales", "cogs", "total_assets"),
        lambda d: _eq_by(d["gross_profit"], d["net_sales"] - d["cogs"], d["total_assets"]),
        "Gross profit = Net sales - COGS",
    ),
    Check(
        "cf_sums_to_net", "hard", UNIVERSAL,
        ("cfo", "cfi", "cff", "net_cash_flow", "total_assets"),
        lambda d: _eq_by(d["cfo"] + d["cfi"] + d["cff"], d["net_cash_flow"],
                         d["total_assets"]),
        "Under VAS, net cash flow (code 50) is the sum of the three activity sections "
        "and EXCLUDES the FX translation effect (code 61), which is added afterwards to "
        "reach closing cash. Internal to one table, no lag needed.",
    ),
    Check("cf_internal_roll", "hard", UNIVERSAL,
      ("cash_begin", "cfo", "cfi", "cff", "fx_effect", "cash_end", "total_assets"),
      lambda d: _eq_by(d["cash_begin"] + d["cfo"] + d["cfi"] + d["cff"]
                       + d["fx_effect"].fillna(0),
                       d["cash_end"], d["total_assets"]),
      ""),

    # Check(
    #     "cf_ties_to_cash", "soft", UNIVERSAL,
    #     ("cfo", "cfi", "cff", "fx_effect",
    #      "cash_and_equivalents", "cash_and_equivalents_lag", "total_assets"),
    #     lambda d: _eq_by(d["cfo"] + d["cfi"] + d["cff"] + d["fx_effect"].fillna(0),
    #                      d["cash_and_equivalents"] - d["cash_and_equivalents_lag"],
    #                      d["total_assets"]),
    #     "The only tie between the cash-flow statement and the balance sheet: the three "
    #     "activity sections plus FX translation must equal the movement in balance-sheet "
    #     "cash. Nothing else in the panel constrains the two statements against each "
    #     "other. NA in each symbol's first year, and wherever the prior period is not "
    #     "exactly one year back.",
    #     cross_period=True,
    # ),
    Check(
        "total_liabilities_identity", "hard", UNIVERSAL,
        ("total_liabilities", "long_term_liabilities", "current_liabilities", "total_assets"),
        lambda d: _eq_by(d["total_liabilities"], d["long_term_liabilities"] + d["current_liabilities"], d["total_assets"]),
        "Total liabilities = Long term liabilities + Current liabilities",
    ),
    Check(
        "net_income_parent_identity", "hard", UNIVERSAL,
        ("net_income_parent", "net_income_total", "net_income_minority", "total_assets"),
        lambda d: _eq_by(d["net_income_parent"], d["net_income_total"] - d["net_income_minority"], d["total_assets"]),
        "Parent company net income = Total net income - Net income of minority",
    ),
    Check(
        "issuance_nonneg", "hard", UNIVERSAL,
        ("stock_issuance_proceeds",),
        lambda d: d["stock_issuance_proceeds"] >= 0,
        "Proceeds RECEIVED from issuing shares. Negative means buybacks were folded into "
        "the same line, or the sign convention is flipped.",
    )
]


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------

def checks_for(framework: str | None = None, severity: str | None = None) -> list[Check]:
    """Subset the rule set, e.g. checks_for(framework=UNIVERSAL) for an IFRS market."""
    out = CHECKS
    if framework:
        out = [c for c in out if c.framework == framework]
    if severity:
        out = [c for c in out if c.severity == severity]
    return out


def run_checks(panel: pd.DataFrame, checks: list[Check] | None = None) -> pd.DataFrame:
    """
    Boolean frame, one column per check, index = panel index.
    True = pass, False = fail, NA = untestable.
    Cross-period checks are NA where the previous period is not exactly one year back.
    """
    checks = checks or CHECKS
    needs_lag = any(c.cross_period for c in checks)
    d = add_lags(panel) if needs_lag and "lag_contiguous" not in panel.columns else panel

    res = {}
    for c in checks:
        missing = [col for col in c.inputs if col not in d.columns]
        if missing:
            res[c.name] = pd.Series(pd.NA, index=d.index, dtype="boolean")
            continue
        cols = [d[col] for col in c.inputs]
        out = c.fn(d).astype("boolean")
        out[_na_mask(*cols)] = pd.NA
        if c.cross_period and "lag_contiguous" in d.columns:
            out[~d["lag_contiguous"].fillna(False)] = pd.NA
        res[c.name] = out
    return pd.DataFrame(res, index=d.index)


def summary(res: pd.DataFrame, checks: list[Check] | None = None) -> pd.DataFrame:
    """Per-check pass/fail/NA counts, worst first. This is the table to read."""
    spec = {c.name: c for c in (checks or CHECKS)}
    rows = []
    for name in res.columns:
        col = res[name]
        tested = int(col.notna().sum())
        failed = int((col == False).sum())  # noqa: E712
        c = spec.get(name)
        rows.append({
            "check": name,
            "severity": c.severity if c else "",
            "framework": c.framework if c else "",
            "tested": tested,
            "failed": failed,
            "fail_pct": round(100 * failed / tested, 3) if tested else np.nan,
            "not_applicable": int(col.isna().sum()),
        })
    out = pd.DataFrame(rows).set_index("check")
    order = {"hard": 0, "soft": 1, "info": 2}
    return (out.assign(_o=out["severity"].map(order))
               .sort_values(["_o", "fail_pct"], ascending=[True, False])
               .drop(columns="_o"))


def offenders(res: pd.DataFrame, check: str, panel: pd.DataFrame,
              cols: list[str] | None = None) -> pd.DataFrame:
    """Rows that failed one check, showing the columns that check reads."""
    spec = {c.name: c for c in CHECKS}.get(check)
    cols = cols or [c for c in (spec.inputs if spec else ()) if c in panel.columns]
    return panel.loc[res[check] == False, cols]  # noqa: E712

def passed(res: pd.DataFrame, check: str, panel: pd.DataFrame,
              cols: list[str] | None = None) -> pd.DataFrame:
    """Rows that failed one check, showing the columns that check reads."""
    spec = {c.name: c for c in CHECKS}.get(check)
    cols = cols or [c for c in (spec.inputs if spec else ()) if c in panel.columns]
    return panel.loc[res[check] == True, cols]  # noqa: E712

def not_applicable(res: pd.DataFrame, check: str, panel: pd.DataFrame,
              cols: list[str] | None = None) -> pd.DataFrame:
    """Rows that failed one check, showing the columns that check reads."""
    spec = {c.name: c for c in CHECKS}.get(check)
    cols = cols or [c for c in (spec.inputs if spec else ()) if c in panel.columns]
    return panel.loc[res[check].isna(), cols]  # noqa: E712

def failures_by_year(res: pd.DataFrame, check: str) -> pd.Series:
    """Fail rate per year - the fastest way to tell a template break from a data error."""
    yr = pd.Series(res.index.get_level_values("period"), index=res.index).astype(str).str[:4]
    col = res[check]
    return (col == False).groupby(yr).sum() / col.notna().groupby(yr).sum()  # noqa: E712


def clean_mask(res: pd.DataFrame, severity: str = "hard",
               checks: list[Check] | None = None) -> pd.Series:
    """
    True for rows passing every check at or above the given severity.
    NA counts as passing - untestable is not the same as wrong.
    info-level checks never gate the universe.
    """
    checks = checks or CHECKS
    keep = {"hard": ("hard",), "soft": ("hard", "soft")}[severity]
    names = [c.name for c in checks if c.severity in keep and c.name in res.columns]
    return ~(res[names] == False).any(axis=1)  # noqa: E712


def rules_table() -> pd.DataFrame:
    """The rule set as a frame - for review, export, or the appendix."""
    return pd.DataFrame([{
        "check": c.name,
        "severity": c.severity,
        "framework": c.framework,
        "cross_period": c.cross_period,
        "inputs": ", ".join(c.inputs),
        "note": " ".join(c.note.split()),
    } for c in CHECKS]).set_index("check")


def nci_placement(panel: pd.DataFrame, tol: float = REL_TOL) -> pd.Series:
    """
    Which side of the equity line minority interest sits on, per row.

    Needs total_liabilities and equity_section. The residual
        r = total_assets - total_liabilities - equity_section
    is 0 when NCI is nested inside equity (TT202, code 429), and equals NCI when it
    sits outside both (QD15, code 439) - and there book_equity must NOT subtract it.

    Returns 'inside' / 'outside' / 'unexplained' / 'untestable'. This is the direct
    answer that nci_le_equity can only approximate:

        pd.crosstab(panel.index.get_level_values("period").str[:4],
                    nci_placement(panel))
    """
    need = ("total_assets", "total_liabilities", "equity_section", "minority_interest")
    if any(c not in panel.columns for c in need):
        return pd.Series("untestable", index=panel.index)

    resid = panel["total_assets"] - panel["total_liabilities"] - panel["equity_section"]
    scale = panel["total_assets"].abs().replace(0, np.nan)
    inside = (resid.abs() / scale) <= tol
    outside = ((resid - panel["minority_interest"]).abs() / scale) <= tol
    return pd.Series(
        np.select([inside, outside], ["inside", "outside"], default="unexplained"),
        index=panel.index,
    ).where(resid.notna(), "untestable")