"""
Source-agnostic field mapping for Vietnamese financial statements (VAS, non-financial).

Two layers:
  1. Canonical layer  - logical statement names + canonical field keys. Never changes.
                        FSCORE_INPUTS / BM_INPUTS are written here, once.
  2. Source layer     - one Source per data provider: its table names, its labels,
                        its unit. Adding CafeF/TCBS = adding a Source, nothing else.

Downstream code takes a Source as an argument and never sees a Vietnamese label.
"""

from dataclasses import dataclass, field as _field

import pandas as pd

# ===========================================================================
# 1. CANONICAL LAYER
# ===========================================================================

BALANCE_SHEET = "balance_sheet"
INCOME_STATEMENT = "income_statement"
CASH_FLOW = "cash_flow"
CASH_FLOW_DIRECT = "cash_flow_direct"

STATEMENTS = (BALANCE_SHEET, INCOME_STATEMENT, CASH_FLOW, CASH_FLOW_DIRECT)

PAR_VALUE_VND = 10_000  # HOSE/HNX standard par

# book_equity   = owner_equity - minority_interest
# shares_issued = paid_in_capital * source.unit_vnd / PAR_VALUE_VND
# BM            = book_equity / (close_raw * shares), both at period end
#
# `shares_issued` counts treasury stock, `book_equity` does not, so the two are not on the
# same basis. book_to_market_calculation.ipynb resolves that with a share count outside
# this module (fireant_financial_data_general.ShareAtPeriodEnd, guarded) and no longer
# calls book_to_market() below. The helpers here are kept for the gross-count case.
BM_INPUTS = (
    (BALANCE_SHEET, "owner_equity"),
    (BALANCE_SHEET, "minority_interest"),
    (BALANCE_SHEET, "paid_in_capital"),
    (BALANCE_SHEET, "treasury_stock"),
)

# Not used by any signal — pulled so the accounting checks can run the exact
# balance identity (total_liabilities + equity_section = total_assets) and resolve
# whether minority interest sits inside or outside equity. A tuple, not a set:
# REQUIRED drives column order in widen(), and set iteration order is not stable
# across runs, which would silently reshuffle the panel between sessions.
OTHERS_INPUTS = (
    # balance identity: total_liabilities + equity_section = total_assets,
    # and the residual against minority_interest resolves TT202 vs QD15 nesting
    (BALANCE_SHEET, "total_liabilities"),
    (BALANCE_SHEET, "equity_section"),
    (BALANCE_SHEET, "long_term_liabilities"),



    # pretax_profit = operating_profit + other_profit, within one filing
    (INCOME_STATEMENT, "operating_profit"),
    (INCOME_STATEMENT, "other_profit"),
    (INCOME_STATEMENT, "pretax_profit"),
    (INCOME_STATEMENT, "cogs"),
    (INCOME_STATEMENT, "net_income_total"),
    (INCOME_STATEMENT, "net_income_minority"),
    # cfo + cfi + cff + fx_effect = change in balance-sheet cash.
    # This is the only tie between the cash-flow statement and the balance sheet;
    # nothing else in the panel constrains the two against each other.
    (BALANCE_SHEET, "cash_and_equivalents"),
    (CASH_FLOW, "cfi"),
    (CASH_FLOW, "cff"),
    (CASH_FLOW, "fx_effect"),
    (CASH_FLOW, "net_cash_flow"),
    (CASH_FLOW, "cash_begin"),
    (CASH_FLOW, "cash_end"),
    # same identity on the direct table, plus cfo_direct for the two-vintage
    # comparison: net totals can agree to the dong while CFO/CFI/CFF differ,
    # so both checks are needed to tell the two failures apart
    (CASH_FLOW_DIRECT, "cfo_direct"),
    (CASH_FLOW_DIRECT, "cfi_direct"),
    (CASH_FLOW_DIRECT, "cff_direct"),
    (CASH_FLOW_DIRECT, "fx_effect_direct"),
    (CASH_FLOW_DIRECT, "net_cash_flow_direct"),
    (CASH_FLOW_DIRECT, "cash_begin_direct"),
    (CASH_FLOW_DIRECT, "cash_end_direct"),
)

FSCORE_INPUTS = {
    "roa": ((INCOME_STATEMENT, "net_income_parent"),
            (BALANCE_SHEET, "total_assets")),
    "cfoa": ((CASH_FLOW, "cfo"),
             (BALANCE_SHEET, "total_assets")),
    "d_roa": ((INCOME_STATEMENT, "net_income_parent"),
              (BALANCE_SHEET, "total_assets")),
    "accrual": ((CASH_FLOW, "cfo"),
                (INCOME_STATEMENT, "net_income_parent"),
                (BALANCE_SHEET, "total_assets")),
    "d_lever": ((BALANCE_SHEET, "long_term_debt"),
                (BALANCE_SHEET, "total_assets")),
    "d_liquid": ((BALANCE_SHEET, "current_assets"),
                 (BALANCE_SHEET, "current_liabilities")),
    "eq_offer": ((CASH_FLOW, "stock_issuance_proceeds"),
                 (BALANCE_SHEET, "paid_in_capital")),
    "d_margin": ((INCOME_STATEMENT, "gross_profit"),
                 (INCOME_STATEMENT, "net_sales")),
    "d_turn": ((INCOME_STATEMENT, "net_sales"),
               (BALANCE_SHEET, "total_assets")),
}

# Every (statement, key) any strategy code is allowed to ask for.
REQUIRED = tuple(dict.fromkeys(
    list(BM_INPUTS)
    + [p for pairs in FSCORE_INPUTS.values() for p in pairs]
    + list(OTHERS_INPUTS)
))


# ===========================================================================
# 2. SOURCE LAYER
# ===========================================================================

@dataclass(frozen=True)
class Source:
    name: str
    unit_vnd: float                      # multiply a stored value by this to get VND
    tables: dict                         # statement -> SQL table name
    fields: dict                         # statement -> {canonical key: source label}
    notes: dict = _field(default_factory=dict)

    def table(self, statement: str) -> str:
        return self.tables[statement]

    def label(self, statement: str, key: str) -> str:
        return self.fields[statement][key]

    def has(self, statement: str, key: str) -> bool:
        return key in self.fields.get(statement, {})

    def labels(self, statement: str, keys=None) -> dict:
        """{canonical key: source label} for the requested keys (default: all)."""
        m = self.fields[statement]
        return dict(m) if keys is None else {k: m[k] for k in keys if k in m}

    def keys_for(self, statement: str) -> list:
        """Canonical keys this source can supply for a statement, in REQUIRED order."""
        return [k for s, k in REQUIRED if s == statement and self.has(s, k)]

    def missing(self) -> list:
        """(statement, key) pairs in REQUIRED that this source does not provide."""
        return [(s, k) for s, k in REQUIRED if not self.has(s, k)]

    def signals_supported(self) -> list:
        return [sig for sig, pairs in FSCORE_INPUTS.items()
                if all(self.has(s, k) for s, k in pairs)]


# ---------------------------------------------------------------- Fireant

FIREANT = Source(
    name="fireant",
    unit_vnd=1e6,  # values stored in millions of VND
    tables={
        BALANCE_SHEET: "fireant_balance_sheet",
        INCOME_STATEMENT: "fireant_income_statement",
        CASH_FLOW: "fireant_cash_flow",
        CASH_FLOW_DIRECT: "fireant_cash_flow_direct",
    },
    fields={
        BALANCE_SHEET: {
            "total_assets": "TỔNG CỘNG TÀI SẢN",
            "total_resources": "TỔNG CỘNG NGUỒN VỐN",
            "current_assets": "A. Tài sản lưu động và đầu tư ngắn hạn",
            "long_term_assets": "B. Tài sản cố định và đầu tư dài hạn",
            "cash_and_equivalents": "I. Tiền và các khoản tương đương tiền",
            "inventory": "IV. Tổng hàng tồn kho",
            "total_liabilities": "A. Nợ phải trả",
            "current_liabilities": "I. Nợ ngắn hạn",
            "long_term_liabilities": "II. Nợ dài hạn",
            "short_term_debt": "1. Vay và nợ thuê tài chính ngắn hạn",
            "long_term_debt_current_portion": "2. Vay và nợ dài hạn đến hạn phải trả",
            "long_term_debt": "6. Vay và nợ thuê tài chính dài hạn",
            "equity_section": "B. Nguồn vốn chủ sở hữu",
            "owner_equity": "I. Vốn chủ sở hữu",
            "funds_other": "II. Nguồn kinh phí và quỹ khác",
            "paid_in_capital": "1. Vốn đầu tư của chủ sở hữu",
            "share_premium": "2. Thặng dư vốn cổ phần",
            "treasury_stock": "5. Cổ phiếu quỹ",
            "retained_earnings": "11. Lợi nhuận sau thuế chưa phân phối",
            "minority_interest": "14. Lợi ích của cổ đông không kiểm soát",
        },
        INCOME_STATEMENT: {
            "gross_revenue": "1. Tổng doanh thu hoạt động kinh doanh",
            "revenue_deductions": "2. Các khoản giảm trừ doanh thu",
            "net_sales": "3. Doanh thu thuần (1)-(2)",
            "cogs": "4. Giá vốn hàng bán",
            "gross_profit": "5. Lợi nhuận gộp (3)-(4)",
            "financial_income": "6. Doanh thu hoạt động tài chính",
            "financial_expense": "7. Chi phí tài chính",
            "interest_expense": "-Trong đó: Chi phí lãi vay",
            "selling_expense": "9. Chi phí bán hàng",
            "admin_expense": "10. Chi phí quản lý doanh nghiệp",
            "operating_profit": "11. Lợi nhuận thuần từ hoạt động kinh doanh (5)+(6)-(7)+(8)-(9)-(10)",
            "other_income": "12. Thu nhập khác",
            "other_expense": "13. Chi phí khác",
            "other_profit": "14. Lợi nhuận khác (12)-(13)",
            "pretax_profit": "15. Tổng lợi nhuận kế toán trước thuế (11)+(14)",
            "tax_expense": "18. Chi phí thuế TNDN (16)+(17)",
            "net_income_total": "19. Lợi nhuận sau thuế thu nhập doanh nghiệp (15)-(18)",
            "net_income_minority": "20. Lợi nhuận sau thuế của cổ đông không kiểm soát",
            "net_income_parent": "21. Lợi nhuận sau thuế của cổ đông của công ty mẹ (19)-(20)",
        },
        CASH_FLOW: {
            "cfo": "Lưu chuyển tiền thuần từ hoạt động kinh doanh",
            "cfi": "Lưu chuyển tiền thuần từ hoạt động đầu tư",
            "cff": "Lưu chuyển tiền thuần từ hoạt động tài chính",
            "net_cash_flow": "Lưu chuyển tiền thuần trong kỳ",
            "fx_effect": "Ảnh hưởng của thay đổi tỷ giá hối đoái quy đổi ngoại tệ",
            "cash_begin": "Tiền và tương đương tiền đầu kỳ",
            "cash_end": "Tiền và tương đương tiền cuối kỳ",
            "pretax_profit": "1. Lợi nhuận trước thuế",
            "depreciation": "- Khấu hao TSCĐ",
            "capex": "1. Tiền chi để mua sắm, xây dựng TSCĐ và các tài sản dài hạn khác",
            "stock_issuance_proceeds": "1. Tiền thu từ phát hành cổ phiếu, nhận vốn góp của chủ sở hữu",
            "share_buyback": "2. Tiền chi trả vốn góp cho các chủ sở hữu, mua lại cổ phiếu của doanh nghiệp đã phát hành",
            "borrowings_received": "3. Tiền vay ngắn hạn, dài hạn nhận được",
            "debt_repayment": "4. Tiền chi trả nợ gốc vay",
            "dividends_paid": "8. Cổ tức, lợi nhuận đã trả cho chủ sở hữu",
        },
        # Every key here carries a _direct suffix. The labels are identical to the
        # indirect table's, so without the suffix "cfo" would mean two different
        # columns once the statements are flattened into one panel — and whichever
        # got concatenated last would silently win.
        CASH_FLOW_DIRECT: {
            "cfo_direct": "Lưu chuyển tiền thuần từ hoạt động kinh doanh",
            "cfi_direct": "Lưu chuyển tiền thuần từ hoạt động đầu tư",
            "cff_direct": "Lưu chuyển tiền thuần từ hoạt động tài chính",
            "net_cash_flow_direct": "Lưu chuyển tiền thuần trong kỳ",
            "fx_effect_direct": "Ảnh hưởng của thay đổi tỷ giá hối đoái quy đổi ngoại tệ",
            "cash_begin_direct": "Tiền và tương đương tiền đầu kỳ",
            "cash_end_direct": "Tiền và tương đương tiền cuối kỳ",
            "cash_from_sales": "1. Tiền thu từ bán hàng, cung cấp dịch vụ và doanh thu khác",
            "cash_to_suppliers": "2. Tiền chi trả cho người cung cấp hàng hóa và dịch vụ",
            "cash_to_employees": "3. Tiền chi trả cho người lao động",
            "interest_paid_direct": "4. Tiền chi trả lãi vay",
            "capex_direct": "1. Tiền chi để mua sắm, xây dựng TSCĐ và các tài sản dài hạn khác",
            "stock_issuance_proceeds_direct": "1. Tiền thu từ phát hành cổ phiếu, nhận vốn góp của chủ sở hữu",
            "dividends_paid_direct": "7. Cổ tức, lợi nhuận đã trả cho chủ sở hữu",
        },
    },
    notes={
        "minority_interest": "TT202: nested inside owner_equity. Pre-2015 QĐ15 puts it outside.",
        "treasury_stock": "Carried at cost, not par — cannot be converted to a share count.",
        "cfo": "Also present in CASH_FLOW_DIRECT and the two disagree for some tickers.",
    },
)

# ------------------------------------------------------- CafeF / TCBS stubs
# Fill `fields` with that source's own labels. Leave a key out if the source
# does not have it — .missing() and .signals_supported() will report it.

CAFEF = Source(
    name="cafef",
    unit_vnd=1.0,
    tables={
        BALANCE_SHEET: "cafef_balance_sheet",
        INCOME_STATEMENT: "cafef_income_statement",
        CASH_FLOW: "cafef_cash_flow",
        CASH_FLOW_DIRECT: "cafef_cash_flow_direct",
    },
    fields={s: {} for s in STATEMENTS},
)

TCBS = Source(
    name="tcbs",
    unit_vnd=1e9,
    tables={
        BALANCE_SHEET: "tcbs_balance_sheet",
        INCOME_STATEMENT: "tcbs_income_statement",
        CASH_FLOW: "tcbs_cash_flow",
        CASH_FLOW_DIRECT: "tcbs_cash_flow_direct",
    },
    fields={s: {} for s in STATEMENTS},
    notes={"*": "Values are display-rounded."},
)

SOURCES = {s.name: s for s in (FIREANT, CAFEF, TCBS)}


# ===========================================================================
# 3. CONSUMERS
# ===========================================================================

def widen(df, source: Source, statement: str, keys=None,
          name_col="name", value_col="value", index=("symbol", "period")):
    """
    Long (symbol, period, name, value) -> wide, columns named by canonical key.
    Requested keys the source lacks come back as all-NaN columns, so the frame
    shape is identical across sources.
    """
    keys = list(keys) if keys is not None else source.keys_for(statement)
    label_of = source.labels(statement, keys)
    key_of = {v: k for k, v in label_of.items()}
    wide = (df[df[name_col].isin(key_of)]
            .assign(_k=lambda d: d[name_col].map(key_of))
            .pivot(index=list(index), columns="_k", values=value_col))
    return wide.reindex(columns=keys).rename_axis(columns=None)


def to_vnd(wide, source: Source):
    """Scale a widened frame to plain VND so sources are comparable."""
    return wide * source.unit_vnd


def book_equity(bs_wide):
    return bs_wide["owner_equity"] - bs_wide["minority_interest"].fillna(0)


def shares_issued(bs_wide, source: Source, par_value=PAR_VALUE_VND):
    """Shares *issued* — treasury included. See share_count() for the traded count."""
    return bs_wide["paid_in_capital"] * source.unit_vnd / par_value


# Guards on a vendor-supplied outstanding-share count. Both one-sided, both measured in
# share_count_diagnostics.ipynb; the sensitivity table there shows the -30% one barely
# moves the panel anywhere between -10% and -100%, so it is not a tuned parameter.
SHARE_GAP_TOL = 0.005      # within 0.5% of the par-implied count: the two agree
MAX_TREASURY_FRAC = 0.30   # largest believable buyback; past this the vendor number is wrong

# A list, not a tuple: it is used to select columns, and a tuple would be read as one key.
SHARE_COLS = ["shares_issued", "shares_out", "shares_gap", "shares_source", "shares"]


def share_count(paid_in_capital, shares_out=None, unit_vnd=1.0,
                par_value=PAR_VALUE_VND, tol=SHARE_GAP_TOL,
                max_treasury=MAX_TREASURY_FRAC):
    """
    The share count both market equity and turnover need, with the choice kept beside it.

    Lives here rather than in a notebook because two pipelines must agree on it to the
    share: book_to_market_calculation.ipynb divides price by it and
    tradable_by_turnover_filter.ipynb divides volume by it, and
    price_matching_and_finalize.ipynb asserts the two panels match.

    `paid_in_capital` gives shares *issued*, treasury included. `shares_out` is a vendor
    count of shares outstanding, used only where it survives two guards:

        shares_gap > +tol           outstanding above issued, which cannot happen
        shares_gap < -max_treasury  past any believable buyback

    Rows failing either guard fall back to the issued count, so none is lost. Accepted rows
    are clipped at the issued count: a small positive gap is rounding noise rather than a
    share count, and clipping is what keeps the correction strictly one-sided.

    Note the correction is partial. Where the vendor does not net treasury out at all — 10%
    of the firm-years whose balance sheet records treasury stock — this returns the issued
    count and the old bias survives, which is why callers keep a `has_treasury` flag.
    """
    out = pd.DataFrame(index=paid_in_capital.index)
    out["shares_issued"] = paid_in_capital * unit_vnd / par_value
    out["shares_out"] = (float("nan") if shares_out is None
                         else shares_out.reindex(paid_in_capital.index))

    # Signed, so both guards read off one column: negative is a buyback, positive an error.
    out["shares_gap"] = ((out["shares_out"] - out["shares_issued"])
                         / out["shares_issued"].replace(0, float("nan")))

    usable = (out["shares_out"].notna()
              & out["shares_gap"].le(tol)
              & out["shares_gap"].ge(-max_treasury))
    out["shares_source"] = usable.map({True: "outstanding", False: "par_implied"})
    out["shares"] = (out["shares_out"].clip(upper=out["shares_issued"])
                     .where(usable, out["shares_issued"]))
    return out


def book_to_market(bs_wide, close_raw, source: Source):
    """
    close_raw: unadjusted close in VND at the SAME period end as bs_wide.

    Denominated in shares *issued*, so it overstates market equity — and understates BM —
    for any firm holding treasury stock. See the note on BM_INPUTS above.
    """
    be = book_equity(bs_wide) * source.unit_vnd
    me = close_raw * shares_issued(bs_wide, source)
    bm = be / me
    return bm.where(be > 0)


def sql_in_clause(source: Source, statement: str, keys=None) -> str:
    """Labels as a SQL IN(...) list, for pulling only what you need."""
    labels = source.labels(statement, keys or source.keys_for(statement)).values()
    return ", ".join("'" + s.replace("'", "''") + "'" for s in labels)