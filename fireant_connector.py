"""Connector lấy dữ liệu báo cáo tài chính từ FireAnt (restv2.fireant.vn)."""

import logging
import re
import sqlite3
import time
import warnings

import pandas as pd
import requests

# Nhãn kỳ báo cáo (đồng bộ với cafef_connector): 'Q1-2026' (quý) hoặc '2025' (năm)
_PERIOD_RE = re.compile(r'^(?:Q([1-4])-)?(\d{4})$')

logger = logging.getLogger('fireant')
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)


class Fireant:
    """Client cho API báo cáo tài chính của FireAnt.

    Khác với CafeF (chia page), FireAnt trả toàn bộ các kỳ trong MỘT request:
    truyền year/quarter làm mốc và `limit` là số kỳ lấy về. Response là một
    list phẳng các chỉ tiêu, mỗi chỉ tiêu có `id` (mã), `parentID` (cây phân
    cấp) và `values` (giá trị theo từng kỳ).

    Loại báo cáo (tham số `type`):
        1 -> Bảng cân đối kế toán (Balance sheet)
        2 -> Kết quả kinh doanh (Income statement)
        3 -> Lưu chuyển tiền tệ trực tiếp (Cashflow direct)
        4 -> Lưu chuyển tiền tệ gián tiếp (Cashflow indirect)

    Ngoài 4 báo cáo trên, class còn hỗ trợ endpoint `financial-data` — bộ chỉ
    tiêu tài chính ĐÃ CHUẨN HOÁ "3 trong 1" (CĐKT + KQKD + LCTT, kèm chỉ số
    định giá / tăng trưởng / Piotroski / Altman) — xem `get_financial_data`.

    Example
    -------
    >>> fa = Fireant()
    >>> bs = fa.get_balance_sheet('HPG', limit=4)
    >>> kqkd = fa.get_income_statement('HPG', quarter=4, limit=8)
    >>> fin = fa.get_financials('HPG', '2022-01-01', '2026-12-31', type_time='NAM')
    >>> fd = fa.get_financial_data('HPG', start_year=2009)   # chuẩn hoá, theo năm
    """

    BASE_URL = 'https://restv2.fireant.vn'

    # type -> (tên statement, có chia section theo cây phân cấp hay không)
    _STATEMENTS = {
        'balance_sheet': (1, True),
        'income_statement': (2, False),
        'cash_flow': (4, True),
        'cash_flow_direct': (3, True),
    }

    # Token công khai của frontend FireAnt (hạn dùng tới ~2029). Có thể ghi đè
    # qua tham số `token` trong __init__.
    DEFAULT_TOKEN = (
        'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsIng1dCI6IkdYdExONzViZlZQakdvNERW'
        'djV4QkRITHpnSSIsImtpZCI6IkdYdExONzViZlZQakdvNERWdjV4QkRITHpnSSJ9.eyJp'
        'c3MiOiJodHRwczovL2FjY291bnRzLmZpcmVhbnQudm4iLCJhdWQiOiJodHRwczovL2Fj'
        'Y291bnRzLmZpcmVhbnQudm4vcmVzb3VyY2VzIiwiZXhwIjoxODg5NjIyNTMwLCJuYmYi'
        'OjE1ODk2MjI1MzAsImNsaWVudF9pZCI6ImZpcmVhbnQudHJhZGVzdGF0aW9uIiwic2Nv'
        'cGUiOlsiYWNhZGVteS1yZWFkIiwiYWNhZGVteS13cml0ZSIsImFjY291bnRzLXJlYWQi'
        'LCJhY2NvdW50cy13cml0ZSIsImJsb2ctcmVhZCIsImNvbXBhbmllcy1yZWFkIiwiZmlu'
        'YW5jZS1yZWFkIiwiaW5kaXZpZHVhbHMtcmVhZCIsImludmVzdG9wZWRpYS1yZWFkIiwi'
        'b3JkZXJzLXJlYWQiLCJvcmRlcnMtd3JpdGUiLCJwb3N0cy1yZWFkIiwicG9zdHMtd3Jp'
        'dGUiLCJzZWFyY2giLCJzeW1ib2xzLXJlYWQiLCJ1c2VyLWRhdGEtcmVhZCIsInVzZXIt'
        'ZGF0YS13cml0ZSIsInVzZXJzLXJlYWQiXSwianRpIjoiMjYxYTZhYWQ2MTQ5Njk1ZmJi'
        'YzcwODM5MjM0Njc1NWQifQ.dA5-HVzWv-BRfEiAd24uNBiBxASO-PAyWeWESovZm_hj4a'
        'XMAZA1-bWNZeXt88dqogo18AwpDQ-h6gefLPdZSFrG5umC1dVWaeYvUnGm62g4XS29fj'
        '6p01dhKNNqrsu5KrhnhdnKYVv9VdmbmqDfWR8wDgglk5cJFqalzq6dJWJInFQEPmUs9B'
        'W_Zs8tQDn-i5r4tYq2U8vCdqptXoM7YgPllXaPVDeccC9QNu2Xlp9WUvoROzoQXg25lF'
        'ub1IYkTrM66gJ6t9fJRZToewCt495WNEOQFa_rwLCZ1QwzvL0iYkONHS_jZ0BOhBCdW9'
        'dWSawD6iF1SIQaFROvMDH1rg'
    )

    def __init__(self, token=None, timeout=30, request_sleep=0.3):
        """
        Parameters
        ----------
        token : str           - Bearer token (mặc định dùng DEFAULT_TOKEN)
        timeout : int         - timeout mỗi request (giây)
        request_sleep : float - nghỉ sau MỖI request (giây) để tránh bị chặn
        """
        self.timeout = timeout
        self.request_sleep = request_sleep
        self.session = requests.Session()
        self.session.headers.update({
            'accept': 'application/json, text/plain, */*',
            'authorization': f'Bearer {token or self.DEFAULT_TOKEN}',
            'origin': 'https://fireant.vn',
            'referer': 'https://fireant.vn/',
            'user-agent': (
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/150.0.0.0 Safari/537.36'
            ),
        })

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _fetch(self, symbol, report_type, year, quarter, limit):
        """Gọi API full-financial-reports, trả về list các chỉ tiêu."""
        params = {
            'type': report_type,
            'year': year,
            'quarter': quarter,
            'limit': limit,
        }
        resp = self.session.get(
            f'{self.BASE_URL}/symbols/{symbol}/full-financial-reports',
            params=params, timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if data is None:
            # một số mã không có báo cáo (vd LCTT gián tiếp) -> API trả null;
            # trả list rỗng để df rỗng thay vì raise, các báo cáo khác vẫn lưu
            logger.warning('%s type=%s: API tra null, bo qua', symbol, report_type)
            data = []
        elif not isinstance(data, list):
            raise RuntimeError(f'FireAnt API error for {symbol}: {data}')
        if self.request_sleep:
            time.sleep(self.request_sleep)
        return data

    @staticmethod
    def _period_label(year, quarter):
        """(2026, 1) -> 'Q1-2026'; (2025, 0) -> '2025'."""
        return f'Q{quarter}-{year}' if quarter else str(year)

    @staticmethod
    def _period_key(label):
        """'Q1-2026' -> (2026, 1); '2025' -> (2025, 0); không phải kỳ -> None."""
        m = _PERIOD_RE.match(str(label))
        if not m:
            return None
        quarter, year = m.groups()
        return (int(year), int(quarter) if quarter else 0)

    @classmethod
    def _parse(cls, items, symbol, sectioned):
        """Parse list chỉ tiêu thành DataFrame (mỗi dòng một chỉ tiêu).

        `section` là tên của chỉ tiêu gốc (level 1) trong cây phân cấp — ví dụ
        'TÀI SẢN' / 'NGUỒN VỐN' cho CDKT. Với KQKD (không có cây con)
        section = None để đồng bộ với cafef_connector.
        """
        if not items:  # API trả null/rỗng -> df rỗng nhưng đủ cột meta
            return pd.DataFrame(
                columns=['section', 'code', 'name', 'level', 'symbol'])

        parent = {it['id']: it['parentID'] for it in items}
        names = {it['id']: it['name'] for it in items}

        def root_name(i):
            seen = set()
            while parent.get(i, -1) != -1 and i not in seen:
                seen.add(i)
                i = parent[i]
            return names.get(i)

        rows = []
        for it in items:
            row = {
                'section': root_name(it['id']) if sectioned else None,
                'code': str(it['id']),
                'name': it['name'].replace('\n', ' ').strip(),
                'level': it['level'],
            }
            for v in it['values']:
                row[cls._period_label(v['year'], v['quarter'])] = v['value']
            rows.append(row)

        df = pd.DataFrame(rows)
        df['symbol'] = symbol
        return df

    def _get_statement(self, symbol, report_type, sectioned, year, quarter,
                       limit, start_key=None, end_key=None):
        """Lấy một báo cáo, tùy chọn lọc + sắp xếp cột kỳ theo [start, end]."""
        items = self._fetch(symbol, report_type, year, quarter, limit)
        df = self._parse(items, symbol, sectioned)

        meta_cols = [c for c in df.columns if self._period_key(c) is None]
        period_cols = [c for c in df.columns if self._period_key(c)]
        if start_key is not None:
            period_cols = [
                c for c in period_cols
                if start_key <= self._period_key(c) <= end_key
            ]
        period_cols = sorted(period_cols, key=self._period_key)
        return df[meta_cols + period_cols]

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def get_financials(self, symbol, start_date, end_date, type_time='QUY'):
        """Lấy cả 3 báo cáo tài chính trong khoảng [start_date, end_date].

        Parameters
        ----------
        symbol : str            - mã cổ phiếu, ví dụ 'HPG'
        start_date, end_date    - str 'YYYY-MM-DD' hoặc datetime
        type_time : str         - 'QUY' (theo quý) hoặc 'NAM' (theo năm)

        Returns
        -------
        dict[str, pd.DataFrame] với 4 key:
        'balance_sheet', 'income_statement', 'cash_flow' (gián tiếp),
        'cash_flow_direct' (trực tiếp);
        cột kỳ được sắp theo thứ tự thời gian tăng dần.
        Mã không có báo cáo nào đó -> DataFrame rỗng (không raise).
        """
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        if start > end:
            raise ValueError('start_date phải <= end_date')

        if type_time == 'QUY':
            quarter = int(end.quarter)
            n_periods = (end.year - start.year) * 4 \
                + (end.quarter - start.quarter) + 1
            start_key = (start.year, start.quarter)
            end_key = (end.year, end.quarter)
        elif type_time == 'NAM':
            quarter = 0
            n_periods = end.year - start.year + 1
            start_key = (start.year, 0)
            end_key = (end.year, 0)
        else:
            raise ValueError("type_time phải là 'QUY' hoặc 'NAM'")

        # +1 kỳ đệm phòng khi mốc end vượt quá dữ liệu có sẵn
        limit = max(1, n_periods + 1)
        logger.info(
            '%s %s: lay %s ky trong khoang (limit=%s, year=%s, quarter=%s)',
            symbol, type_time, n_periods, limit, end.year, quarter,
        )

        return {
            name: self._get_statement(
                symbol, report_type, sectioned, end.year, quarter, limit,
                start_key, end_key,
            )
            for name, (report_type, sectioned) in self._STATEMENTS.items()
        }

    def get_balance_sheet(self, symbol, year=None, quarter=0, limit=8):
        """Bảng cân đối kế toán (type=1).

        Parameters
        ----------
        year : int    - năm mốc (mặc định năm hiện tại)
        quarter : int - 0 = báo cáo năm; 1..4 = báo cáo quý
        limit : int   - số kỳ lấy về (page 1 là các kỳ gần nhất)
        """
        year = year or pd.Timestamp.today().year
        return self._get_statement(symbol, 1, True, year, quarter, limit)

    def get_income_statement(self, symbol, year=None, quarter=0, limit=8):
        """Báo cáo kết quả kinh doanh (type=2)."""
        year = year or pd.Timestamp.today().year
        return self._get_statement(symbol, 2, False, year, quarter, limit)

    def get_cash_flow(self, symbol, year=None, quarter=0, limit=8):
        """Báo cáo lưu chuyển tiền tệ gián tiếp (type=4)."""
        year = year or pd.Timestamp.today().year
        return self._get_statement(symbol, 4, True, year, quarter, limit)

    def get_cash_flow_direct(self, symbol, year=None, quarter=0, limit=8):
        """Báo cáo lưu chuyển tiền tệ trực tiếp (type=3)."""
        year = year or pd.Timestamp.today().year
        return self._get_statement(symbol, 3, True, year, quarter, limit)

    # ------------------------------------------------------------------ #
    # SQLite: dùng chung fscore.db, mỗi statement một bảng prefix 'fireant_'
    # (fireant_balance_sheet, fireant_income_statement, fireant_cash_flow,
    #  fireant_cash_flow_direct)
    # ------------------------------------------------------------------ #

    _TABLE_PREFIX = 'fireant_'

    _DB_SCHEMA = """
        CREATE TABLE IF NOT EXISTS {table} (
            symbol    TEXT NOT NULL,
            section   TEXT,
            code      TEXT NOT NULL,
            name      TEXT,
            level     INTEGER,
            period    TEXT NOT NULL,
            value     REAL,
            PRIMARY KEY (symbol, code, period)
        )
    """

    @classmethod
    def _table(cls, statement):
        return f'{cls._TABLE_PREFIX}{statement}'

    def save_financials(self, fin, db_path='fscore.db'):
        """Lưu kết quả của get_financials vào SQLite (dạng long format).

        Mỗi statement ghi vào bảng riêng ('fireant_' + tên statement).
        Dùng INSERT OR REPLACE theo khóa (symbol, code, period) nên chạy
        lại cùng một mã sẽ ghi đè chứ không tạo dòng trùng / xóa mã khác.
        """
        cols = ['symbol', 'section', 'code', 'name', 'level',
                'period', 'value']
        with sqlite3.connect(db_path) as conn:
            for statement, df in fin.items():
                table = self._table(statement)
                conn.execute(self._DB_SCHEMA.format(table=table))
                id_vars = [c for c in ('code', 'section', 'name', 'level',
                                       'symbol') if c in df.columns]
                long_df = df.melt(id_vars=id_vars, var_name='period',
                                  value_name='value')
                if long_df.empty:
                    logger.warning('%s: khong co du lieu, bo qua', table)
                    continue
                conn.executemany(
                    f'INSERT OR REPLACE INTO {table} '
                    f'({", ".join(cols)}) VALUES ({", ".join("?" * len(cols))})',
                    long_df[cols].itertuples(index=False, name=None),
                )
                logger.info('Da luu %s dong (%s) vao %s.%s',
                            len(long_df), long_df['symbol'].iloc[0],
                            db_path, table)

    # tên statement -> tên cột trong DataFrame summary của crawl_to_db
    _SUMMARY_COLS = {
        'balance_sheet': 'BalanceSheet',
        'income_statement': 'IncomeStatement',
        'cash_flow': 'Cash_flow',
        'cash_flow_direct': 'Cash_flow_direct',
    }

    @classmethod
    def _count_periods(cls, df):
        """Số kỳ (cột kỳ) có trong DataFrame của một báo cáo."""
        if df is None or df.empty:
            return 0
        return sum(1 for c in df.columns if cls._period_key(c))

    def crawl_to_db(self, symbols, start_date, end_date, type_time='QUY',
                    db_path='fscore.db', sleep=1.0, skip_existing=True,
                    log_path=None):
        """Crawl nhiều mã vào SQLite, có thể chạy lại tiếp tục nếu đứt giữa chừng.

        Thiết kế cho đợt crawl lớn (vd 1800 mã): mọi lỗi của một mã chỉ được
        GHI NHẬN vào df kết quả, không làm dừng đợt crawl. Ctrl+C cũng không
        mất log — hàm dừng và trả về phần đã chạy.

        Parameters
        ----------
        symbols : iterable   - danh sách mã cổ phiếu
        skip_existing : bool - bỏ qua mã đã có trong DB (mặc định True)
        sleep : float        - nghỉ giữa các mã (giây) để tránh bị chặn
        log_path : str       - nếu có, ghi log ra CSV sau MỖI mã (file bị ghi
                               đè ở đầu mỗi lần chạy) để không mất log nếu
                               process bị kill giữa đợt

        Returns
        -------
        pd.DataFrame - log kết quả từng mã, một dòng một mã, các cột:
        Symbol | IsSuccess | BalanceSheet | IncomeStatement | Cash_flow |
        Cash_flow_direct | ErrorMessage.
        Bốn cột giữa là SỐ KỲ fetch được của từng báo cáo (theo năm hoặc quý
        tùy `type_time`), ví dụ 17 = 17 kỳ; 0 = mã không có báo cáo đó.
        Mã bị bỏ qua vì đã có trong DB: IsSuccess=True, số kỳ = 0 và
        ErrorMessage ghi rõ lý do bỏ qua.
        """
        # Validate tham số một lần: sai thì báo lỗi ngay thay vì để cả 1800 mã
        # cùng fail vì một lý do giống nhau.
        if pd.Timestamp(start_date) > pd.Timestamp(end_date):
            raise ValueError('start_date phải <= end_date')
        if type_time not in ('QUY', 'NAM'):
            raise ValueError("type_time phải là 'QUY' hoặc 'NAM'")

        with sqlite3.connect(db_path) as conn:
            existing = set()
            for statement in self._STATEMENTS:
                table = self._table(statement)
                conn.execute(self._DB_SCHEMA.format(table=table))
                existing |= {row[0] for row in
                             conn.execute(f'SELECT DISTINCT symbol FROM {table}')}

        symbols = list(symbols)
        cols = (['Symbol', 'IsSuccess'] + list(self._SUMMARY_COLS.values())
                + ['ErrorMessage'])
        summary = []
        try:
            for i, symbol in enumerate(symbols, 1):
                row = {'Symbol': symbol, 'IsSuccess': True,
                       'ErrorMessage': None}
                row.update({col: 0 for col in self._SUMMARY_COLS.values()})
                try:
                    if skip_existing and symbol in existing:
                        logger.info('(%s/%s) %s da co trong DB, bo qua',
                                    i, len(symbols), symbol)
                        row['ErrorMessage'] = 'Da co trong DB, bo qua'
                    else:
                        fin = self.get_financials(symbol, start_date, end_date,
                                                  type_time)
                        self.save_financials(fin, db_path)
                        for statement, col in self._SUMMARY_COLS.items():
                            row[col] = self._count_periods(fin.get(statement))
                        logger.info('(%s/%s) %s xong', i, len(symbols), symbol)
                except Exception as exc:  # mã lỗi không làm dừng cả đợt crawl
                    logger.warning('(%s/%s) %s loi: %s', i, len(symbols),
                                   symbol, exc)
                    row['IsSuccess'] = False
                    row['ErrorMessage'] = f'{type(exc).__name__}: {exc}'

                summary.append(row)
                self._write_log(log_path, row, cols, first=len(summary) == 1)
                if i < len(symbols):
                    time.sleep(sleep)
        except KeyboardInterrupt:  # Ctrl+C: dừng nhưng vẫn trả log đã có
            logger.warning('Dung boi nguoi dung sau %s/%s ma',
                           len(summary), len(symbols))

        return pd.DataFrame(summary, columns=cols)

    @staticmethod
    def _write_log(log_path, row, cols, first):
        """Ghi (append) một dòng log ra CSV; lỗi ghi file không làm dừng crawl."""
        if not log_path:
            return
        try:
            pd.DataFrame([row], columns=cols).to_csv(
                log_path, mode='w' if first else 'a', header=first, index=False)
        except Exception as exc:
            logger.warning('Khong ghi duoc log vao %s: %s', log_path, exc)

    @classmethod
    def load_financials(cls, db_path='fscore.db', symbol=None, statement=None):
        """Đọc dữ liệu từ SQLite ra DataFrame (long format).

        Parameters
        ----------
        symbol : str hoặc list  - lọc theo mã (None = tất cả)
        statement : str         - 'balance_sheet', 'income_statement',
                                  'cash_flow', 'cash_flow_direct'
                                  (None = gộp tất cả, thêm cột 'statement')
        """
        statements = [statement] if statement else list(cls._STATEMENTS)
        where, params = '', []
        if symbol is not None:
            symbols = [symbol] if isinstance(symbol, str) else list(symbol)
            where = f' WHERE symbol IN ({", ".join("?" * len(symbols))})'
            params = symbols

        frames = []
        with sqlite3.connect(db_path) as conn:
            for stmt in statements:
                df = pd.read_sql(f'SELECT * FROM {cls._table(stmt)}{where}',
                                 conn, params=params)
                df['statement'] = stmt
                frames.append(df)
        return pd.concat(frames, ignore_index=True)

    # ================================================================== #
    # financial-data: bộ chỉ tiêu tài chính chuẩn hoá "3 trong 1"
    # ================================================================== #
    #
    # GET /symbols/{symbol}/financial-data?type=Y|Q&count=N
    #
    # Trả về list bản ghi, MỖI BẢN GHI LÀ MỘT KỲ (mới nhất trước):
    #   {symbol, year, quarter, companyType, icbCode, icbName,
    #    financialValues: {<tên chỉ tiêu>: <giá trị>, ...}}
    #
    # Đặc điểm quan trọng (đã kiểm chứng bằng cách gọi thử API):
    #   * `count` BẮT BUỘC, không có tham số neo năm — API luôn trả N kỳ gần
    #     nhất rồi tự cắt khi hết dữ liệu. count=100 với HPG chỉ ra 22 kỳ
    #     (2025 -> 2004), nên cứ xin dư rồi lọc theo năm ở phía client.
    #   * `financialValues` đã gộp CĐKT + KQKD + LCTT + chỉ số (P/E, ROE,
    #     tăng trưởng, Piotroski F-Score, Altman Z-Score, số liệu ngành...).
    #   * BỘ TRƯỜNG PHỤ THUỘC `companyType`, và giống hệt nhau giữa các mã
    #     cùng loại hình: General 340 trường, Bank 298, Securities 449,
    #     Insurance 419. Giữa các loại hình có những tên chỉ khác nhau ở chữ
    #     hoa/thường ('ShortTermPrepaidExpense' vs 'ShorttermPrepaidExpense')
    #     — SQLite coi tên cột KHÔNG phân biệt hoa thường nên không thể nhét
    #     chung một bảng => MỖI companyType MỘT BẢNG (fireant_financial_data_
    #     general / _bank / _securities / _insurance).
    #   * financialValues còn lặp lại Year/Quarter/CompanyType/ICBCode/
    #     ICBName của phần meta bên ngoài (ICBName chỉ khác ở khoảng trắng
    #     thừa) nên bảng chỉ thêm 2 cột 'symbol' và 'period', phần còn lại
    #     giữ NGUYÊN TÊN TRƯỜNG của API làm tên cột.

    _FIN_DATA_PREFIX = 'fireant_financial_data_'

    # type_time -> giá trị tham số `type` của API
    _FIN_DATA_TYPE = {'NAM': 'Y', 'QUY': 'Q'}

    # Cột meta tự thêm (không trùng tên với bất kỳ trường nào của API)
    _FIN_DATA_META = ('symbol', 'period')

    # Trường luôn là chuỗi. Khai báo sẵn vì có trường (ICBCode = '55102010')
    # trông như số: nếu để SQLite đoán kiểu REAL thì mã ngành sẽ bị đổi thành
    # 55102010.0, mất số 0 đứng đầu của các mã khác.
    _FIN_DATA_TEXT_FIELDS = frozenset({
        'CompanyType', 'ICBCode', 'ICBName',
        'ManufacturingStatus', 'ManufacturingSPRating',
        'ManufacturingMoodyRating', 'NonManufacturingStatus',
        'NonManufacturingSPRating', 'NonManufacturingMoodyRating',
    })

    def _fetch_financial_data(self, symbol, period_type, count):
        """Gọi API financial-data, trả list bản ghi (mới nhất trước)."""
        resp = self.session.get(
            f'{self.BASE_URL}/symbols/{symbol}/financial-data',
            params={'type': period_type, 'count': count}, timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if data is None:
            data = []
        elif not isinstance(data, list):
            raise RuntimeError(f'FireAnt API error for {symbol}: {data}')
        if self.request_sleep:
            time.sleep(self.request_sleep)
        return data

    @staticmethod
    def _clean_value(value):
        """Chuẩn hoá một giá trị thô: cắt khoảng trắng thừa của chuỗi."""
        return value.strip() if isinstance(value, str) else value

    def get_financial_data(self, symbol, start_year=2009, end_year=None,
                           type_time='NAM', count=None):
        """Bộ chỉ tiêu tài chính chuẩn hoá của một mã, dạng bảng rộng.

        Parameters
        ----------
        symbol : str       - mã cổ phiếu, ví dụ 'HPG'
        start_year : int   - năm bắt đầu (mặc định 2009)
        end_year : int     - năm kết thúc (mặc định năm hiện tại)
        type_time : str    - 'NAM' (theo năm) hoặc 'QUY' (theo quý)
        count : int        - số kỳ xin từ API; None = tự tính từ start_year
                             (dư 1 kỳ đệm) vì API luôn đếm ngược từ kỳ mới nhất

        Returns
        -------
        pd.DataFrame - MỘT DÒNG MỘT KỲ, sắp xếp tăng dần theo thời gian; cột:
        'symbol', 'period' ('2025' hoặc 'Q1-2026') rồi toàn bộ trường của API
        giữ nguyên tên (Year, Quarter, CompanyType, ..., PE, ROE, ...).
        Mã không có dữ liệu -> DataFrame rỗng.
        """
        if type_time not in self._FIN_DATA_TYPE:
            raise ValueError("type_time phải là 'NAM' hoặc 'QUY'")
        end_year = end_year or pd.Timestamp.today().year
        if start_year > end_year:
            raise ValueError('start_year phải <= end_year')

        if count is None:
            # API đếm ngược từ kỳ MỚI NHẤT (không phải từ end_year) nên phải
            # xin đủ số kỳ tính từ hôm nay, +1 kỳ đệm.
            n_years = pd.Timestamp.today().year - start_year + 1
            count = n_years + 1 if type_time == 'NAM' else n_years * 4 + 4
        count = max(1, int(count))

        logger.info('%s financial-data %s: xin %s ky (>= %s)',
                    symbol, type_time, count, start_year)
        records = self._fetch_financial_data(
            symbol, self._FIN_DATA_TYPE[type_time], count)

        rows = []
        for rec in records:
            if not start_year <= rec['year'] <= end_year:
                continue
            row = {
                'symbol': rec['symbol'],
                'period': self._period_label(rec['year'], rec['quarter']),
            }
            row.update({k: self._clean_value(v)
                        for k, v in rec['financialValues'].items()})
            rows.append(row)

        if not rows:
            return pd.DataFrame(columns=list(self._FIN_DATA_META))

        rows.sort(key=lambda r: self._period_key(r['period']))  # cũ -> mới
        # dtype=object khi dựng để không ép số nguyên lớn (tổng tài sản ngành
        # có thể > 2^53) sang float64 rồi mất chữ số; convert_dtypes() sau đó
        # đưa về Int64/Float64/string vẫn giữ nguyên giá trị.
        return pd.DataFrame(rows, dtype=object).convert_dtypes()

    # ------------------------------------------------------------------ #
    # SQLite cho financial-data
    # ------------------------------------------------------------------ #

    @classmethod
    def _fin_data_table(cls, company_type):
        """'General' -> 'fireant_financial_data_general'."""
        slug = re.sub(r'[^0-9a-z]+', '_', str(company_type).lower()).strip('_')
        return f'{cls._FIN_DATA_PREFIX}{slug or "unknown"}'

    @staticmethod
    def _quote(identifier):
        """Bọc tên bảng/cột trong dấu nháy kép cho an toàn."""
        return '"{}"'.format(str(identifier).replace('"', '""'))

    @classmethod
    def _sql_type(cls, name, series):
        """Đoán kiểu cột từ tên trường + các giá trị quan sát được.

        SQLite gán kiểu động nên đây chỉ là type affinity: một cột INTEGER
        vẫn nhận được số thực, cột REAL vẫn nhận được chuỗi. Chỉ những tên
        trong `_FIN_DATA_TEXT_FIELDS` là bắt buộc TEXT để chuỗi dạng số
        không bị ép sang số.
        """
        if name in cls._FIN_DATA_TEXT_FIELDS:
            return 'TEXT'
        values = [v for v in series if cls._to_sql_value(v) is not None]
        if any(isinstance(v, str) for v in values):
            return 'TEXT'
        if values and all(isinstance(cls._to_sql_value(v), int) for v in values):
            return 'INTEGER'
        return 'REAL'

    @classmethod
    def _ensure_fin_data_table(cls, conn, table, df):
        """Tạo bảng nếu chưa có, hoặc ALTER thêm cột cho trường mới xuất hiện."""
        info = list(conn.execute(f'PRAGMA table_info({cls._quote(table)})'))
        if not info:
            defs = ', '.join(
                f'{cls._quote(c)} {cls._sql_type(c, df[c])}'
                + (' NOT NULL' if c in cls._FIN_DATA_META else '')
                for c in df.columns
            )
            conn.execute(
                f'CREATE TABLE {cls._quote(table)} ({defs}, '
                f'PRIMARY KEY ("symbol", "period"))'
            )
            return
        existing = {row[1].lower() for row in info}
        for col in df.columns:
            if col.lower() not in existing:
                logger.info('%s: them cot moi %s', table, col)
                conn.execute(
                    f'ALTER TABLE {cls._quote(table)} ADD COLUMN '
                    f'{cls._quote(col)} {cls._sql_type(col, df[col])}'
                )

    @staticmethod
    def _to_sql_value(value):
        """pd.NA / NaN -> None; numpy scalar -> kiểu Python cho sqlite3."""
        if value is None or value is pd.NA:
            return None
        if isinstance(value, float) and value != value:  # NaN
            return None
        item = getattr(value, 'item', None)  # np.int64/np.float64 -> int/float
        return item() if callable(item) else value

    def save_financial_data(self, df, db_path='fscore.db'):
        """Lưu kết quả get_financial_data vào SQLite, giữ nguyên schema API.

        Mỗi `CompanyType` ghi vào một bảng riêng (xem `_fin_data_table`), tên
        cột = tên trường của API. INSERT OR REPLACE theo khoá
        (symbol, period) nên chạy lại cùng mã sẽ ghi đè chứ không nhân dòng.
        """
        if df is None or df.empty:
            logger.warning('financial-data: khong co du lieu, bo qua')
            return
        if 'CompanyType' not in df.columns:
            raise ValueError("DataFrame thieu cot 'CompanyType'")

        with sqlite3.connect(db_path) as conn:
            for company_type, part in df.groupby('CompanyType', dropna=False):
                table = self._fin_data_table(company_type)
                self._ensure_fin_data_table(conn, table, part)
                cols = ', '.join(self._quote(c) for c in part.columns)
                conn.executemany(
                    f'INSERT OR REPLACE INTO {self._quote(table)} ({cols}) '
                    f'VALUES ({", ".join("?" * len(part.columns))})',
                    (tuple(self._to_sql_value(v) for v in row)
                     for row in part.itertuples(index=False, name=None)),
                )
                logger.info('Da luu %s ky (%s) vao %s.%s', len(part),
                            part['symbol'].iloc[0], db_path, table)

    @classmethod
    def _fin_data_tables(cls, conn):
        """Danh sách bảng financial-data đang có trong DB."""
        return [row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ?",
            (f'{cls._FIN_DATA_PREFIX}%',),
        )]

    def crawl_financial_data_to_db(self, symbols, start_year=2009,
                                   end_year=None, type_time='NAM',
                                   db_path='fscore.db', sleep=1.0,
                                   skip_existing=True, log_path=None):
        """Crawl financial-data cho nhiều mã vào SQLite (chạy lại được).

        Cùng triết lý với `crawl_to_db`: lỗi của một mã chỉ được ghi vào log
        chứ không làm dừng cả đợt, Ctrl+C vẫn trả về phần đã chạy.

        Returns
        -------
        pd.DataFrame - Symbol | IsSuccess | Periods | CompanyType |
        ErrorMessage. `Periods` là số kỳ lấy được (0 = mã không có dữ liệu
        hoặc bị bỏ qua vì đã có trong DB).
        """
        if type_time not in self._FIN_DATA_TYPE:
            raise ValueError("type_time phải là 'NAM' hoặc 'QUY'")
        end_year = end_year or pd.Timestamp.today().year
        if start_year > end_year:
            raise ValueError('start_year phải <= end_year')

        with sqlite3.connect(db_path) as conn:
            existing = set()
            for table in self._fin_data_tables(conn):
                existing |= {row[0] for row in conn.execute(
                    f'SELECT DISTINCT symbol FROM {self._quote(table)}')}

        symbols = list(symbols)
        cols = ['Symbol', 'IsSuccess', 'Periods', 'CompanyType', 'ErrorMessage']
        summary = []
        try:
            for i, symbol in enumerate(symbols, 1):
                row = {'Symbol': symbol, 'IsSuccess': True, 'Periods': 0,
                       'CompanyType': None, 'ErrorMessage': None}
                try:
                    if skip_existing and symbol in existing:
                        logger.info('(%s/%s) %s da co trong DB, bo qua',
                                    i, len(symbols), symbol)
                        row['ErrorMessage'] = 'Da co trong DB, bo qua'
                    else:
                        df = self.get_financial_data(
                            symbol, start_year, end_year, type_time)
                        if df.empty:
                            logger.warning('(%s/%s) %s khong co du lieu',
                                           i, len(symbols), symbol)
                            row['ErrorMessage'] = 'Khong co du lieu'
                        else:
                            self.save_financial_data(df, db_path)
                            row['Periods'] = len(df)
                            row['CompanyType'] = df['CompanyType'].iloc[-1]
                            logger.info('(%s/%s) %s xong (%s ky)',
                                        i, len(symbols), symbol, len(df))
                except Exception as exc:  # một mã lỗi không làm dừng cả đợt
                    logger.warning('(%s/%s) %s loi: %s', i, len(symbols),
                                   symbol, exc)
                    row['IsSuccess'] = False
                    row['ErrorMessage'] = f'{type(exc).__name__}: {exc}'

                summary.append(row)
                self._write_log(log_path, row, cols, first=len(summary) == 1)
                if i < len(symbols):
                    time.sleep(sleep)
        except KeyboardInterrupt:  # Ctrl+C: dừng nhưng vẫn trả log đã có
            logger.warning('Dung boi nguoi dung sau %s/%s ma',
                           len(summary), len(symbols))

        return pd.DataFrame(summary, columns=cols)

    @classmethod
    def load_financial_data(cls, db_path='fscore.db', symbol=None,
                            company_type=None):
        """Đọc financial-data từ SQLite ra DataFrame (giữ dạng bảng rộng).

        Parameters
        ----------
        symbol : str hoặc list - lọc theo mã (None = tất cả)
        company_type : str     - 'General' / 'Bank' / 'Securities' /
                                 'Insurance'; None = gộp mọi loại hình (các
                                 loại hình có bộ chỉ tiêu khác nhau nên cột
                                 không dùng được của loại hình đó sẽ là NaN)
        """
        where, params = '', []
        if symbol is not None:
            symbols = [symbol] if isinstance(symbol, str) else list(symbol)
            where = f' WHERE symbol IN ({", ".join("?" * len(symbols))})'
            params = symbols

        with sqlite3.connect(db_path) as conn:
            tables = ([cls._fin_data_table(company_type)] if company_type
                      else cls._fin_data_tables(conn))
            frames = [pd.read_sql(f'SELECT * FROM {cls._quote(t)}{where}',
                                  conn, params=params) for t in tables]

        frames = [f for f in frames if not f.empty]
        if not frames:
            return pd.DataFrame(columns=list(cls._FIN_DATA_META))
        # Các trường _CUM luôn NULL trong báo cáo năm -> cột toàn NaN khiến
        # pandas 2.x cảnh báo về cách suy ra dtype khi concat. Chỉ ảnh hưởng
        # dtype của đúng những cột rỗng đó nên tắt riêng cảnh báo này.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', category=FutureWarning,
                message='The behavior of DataFrame concatenation with empty '
                        'or all-NA entries')
            df = pd.concat(frames, ignore_index=True)
        return df.sort_values(['symbol', 'Year', 'Quarter']).reset_index(drop=True)
