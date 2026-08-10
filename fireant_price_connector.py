"""Connector lấy giá lịch sử (OHLCV + khối ngoại + tự doanh) từ FireAnt.

Endpoint `symbols/{ticker}/historical-quotes` trả về nến NGÀY dưới dạng list
object, mới nhất trước, phân trang bằng `offset`/`limit`. Khác Simplize (chỉ
cho nến tháng khi lấy lịch sử dài), endpoint này trả nến ngày cho TOÀN BỘ lịch
sử niêm yết trong một lần gọi (mã dài nhất — REE, SAM, VNINDEX — có 6333 phiên
từ 2000-07-28, lấy hết chỉ ~1 giây).

Ba điểm cần lưu ý về dữ liệu:

* Giá (`priceOpen/High/Low/Close`) là giá THÔ chưa điều chỉnh, đơn vị theo cột
  `unit`: 1000 với cổ phiếu (giá tính bằng nghìn đồng) và 1 với chỉ số. Giá
  VND = price * unit.
* `adjRatio` là hệ số điều chỉnh LUỸ KẾ chia tách / cổ tức, dạng hàm bậc thang
  giảm dần về 1.0 ở phiên gần nhất (HPG: 45.87 ở 2010 -> 1.0, đổi 18 lần).
  Giá điều chỉnh = price * unit / adjRatio; khối lượng điều chỉnh =
  volume * adjRatio. Xem `add_adjusted`.
* Vì adjRatio tính tới NGÀY FETCH, một đợt chia tách mới sẽ làm toàn bộ
  adjRatio cũ trong DB lệch đi. Bảng `fireant_prices_meta` ghi `fetched_at`
  của từng mã để biết dữ liệu điều chỉnh theo mốc nào; muốn đồng bộ lại thì
  crawl với mode='full'.

`priceBasic` là giá tham chiếu của phiên (bằng giá đóng cửa phiên trước, trừ
ngày GDKHQ thì đã trừ quyền), tiện để dò ngày có sự kiện quyền.

Token Bearer dùng chung với `fireant_connector.Fireant` (token công khai của
frontend FireAnt, hạn tới ~2029).

Example
-------
>>> fa = FireantPrices()
>>> df = fa.get_prices('HPG', '2009-01-01', '2026-08-07')
>>> adj = FireantPrices.add_adjusted(df)      # thêm cột *_adj (VND)
>>> log = fa.crawl_to_db(symbols, '2009-01-01', '2026-08-07', db_path='fscore.db')
>>> back = FireantPrices.load_prices('fscore.db', symbol='HPG', adjusted=True)
"""

import logging
import sqlite3
import time

import pandas as pd
import requests

from fireant_connector import Fireant

logger = logging.getLogger('fireant_price')
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)


class FireantPrices:
    """Client lấy giá lịch sử theo phiên của FireAnt."""

    BASE_URL = Fireant.BASE_URL

    # API -> tên cột trong DataFrame/DB. Cột nào API không trả về (mã cũ, chỉ
    # số) vẫn được tạo với giá trị NaN để schema DB luôn ổn định.
    _FIELDS = {
        'symbol': 'symbol',
        'date': 'date',
        'priceOpen': 'price_open',
        'priceHigh': 'price_high',
        'priceLow': 'price_low',
        'priceClose': 'price_close',
        'priceAverage': 'price_average',
        'priceBasic': 'price_basic',
        'adjRatio': 'adj_ratio',
        'unit': 'unit',
        'totalVolume': 'total_volume',
        'dealVolume': 'deal_volume',
        'putthroughVolume': 'putthrough_volume',
        'totalValue': 'total_value',
        'putthroughValue': 'putthrough_value',
        'buyForeignQuantity': 'buy_foreign_quantity',
        'buyForeignValue': 'buy_foreign_value',
        'sellForeignQuantity': 'sell_foreign_quantity',
        'sellForeignValue': 'sell_foreign_value',
        'currentForeignRoom': 'current_foreign_room',
        'buyCount': 'buy_count',
        'buyQuantity': 'buy_quantity',
        'sellCount': 'sell_count',
        'sellQuantity': 'sell_quantity',
        'propTradingNetDealValue': 'prop_trading_net_deal_value',
        'propTradingNetPTValue': 'prop_trading_net_pt_value',
        'propTradingNetValue': 'prop_trading_net_value',
    }

    COLUMNS = list(_FIELDS.values())

    # giá thô -> giá điều chỉnh (chia adjRatio); khối lượng thì NHÂN adjRatio
    _PRICE_COLS = ('price_open', 'price_high', 'price_low', 'price_close',
                   'price_average', 'price_basic')
    _VOLUME_COLS = ('total_volume', 'deal_volume', 'putthrough_volume')

    # API không chặn limit (đã thử 10000), nhưng vẫn phân trang cho chắc
    PAGE_SIZE = 1000
    MAX_PAGES = 50  # chặn vòng lặp vô hạn nếu API đổi cách phân trang

    def __init__(self, token=None, timeout=30, request_sleep=0.3):
        """
        Parameters
        ----------
        token : str           - Bearer token (mặc định Fireant.DEFAULT_TOKEN)
        timeout : int         - timeout mỗi request (giây)
        request_sleep : float - nghỉ sau MỖI request (giây) để tránh bị chặn
        """
        self.timeout = timeout
        self.request_sleep = request_sleep
        self.session = requests.Session()
        self.session.headers.update({
            'accept': 'application/json, text/plain, */*',
            'authorization': f'Bearer {token or Fireant.DEFAULT_TOKEN}',
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

    def _fetch_page(self, symbol, start_date, end_date, offset, limit):
        """Gọi API một trang, trả về list dict nến (mới nhất trước)."""
        resp = self.session.get(
            f'{self.BASE_URL}/symbols/{symbol}/historical-quotes',
            params={
                'startDate': start_date,
                'endDate': end_date,
                'offset': offset,
                'limit': limit,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        # mã không tồn tại -> [] (không raise); lỗi nghiệp vụ -> dict
        if not isinstance(data, list):
            raise RuntimeError(f'FireAnt API error for {symbol}: {data}')
        if self.request_sleep:
            time.sleep(self.request_sleep)
        return data

    @classmethod
    def _to_frame(cls, rows, symbol):
        """List dict thô -> DataFrame chuẩn hóa tên cột, sắp theo ngày tăng."""
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=cls.COLUMNS)
        # chỉ giữ field đã biết, thiếu field nào thì thêm NaN
        df = df.reindex(columns=list(cls._FIELDS)).rename(columns=cls._FIELDS)
        df['symbol'] = symbol  # API trả symbol nhưng vá phòng khi thiếu
        df['date'] = pd.to_datetime(df['date']).dt.normalize()
        return (df[cls.COLUMNS]
                .drop_duplicates('date')
                .sort_values('date')
                .reset_index(drop=True))

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def get_prices(self, symbol, start_date, end_date=None, page_size=None):
        """Lấy nến ngày của một mã trong khoảng [start_date, end_date].

        Parameters
        ----------
        symbol : str            - mã cổ phiếu ('HPG') hoặc chỉ số ('VNINDEX')
        start_date, end_date    - str 'YYYY-MM-DD' hoặc datetime; end_date
                                  None = hôm nay
        page_size : int         - số nến mỗi request (mặc định PAGE_SIZE)

        Returns
        -------
        pd.DataFrame - một dòng một phiên, cột theo `COLUMNS`, ngày tăng dần.
                       Giá là giá THÔ (đơn vị theo cột `unit`) kèm `adj_ratio`;
                       dùng `add_adjusted` để ra giá điều chỉnh VND.
                       Mã không tồn tại / không có phiên nào -> DataFrame rỗng.
        """
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date) if end_date is not None \
            else pd.Timestamp.today().normalize()
        if start > end:
            raise ValueError('start_date phải <= end_date')
        limit = page_size or self.PAGE_SIZE
        start_s, end_s = start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')

        rows, offset = [], 0
        for _ in range(self.MAX_PAGES):
            page = self._fetch_page(symbol, start_s, end_s, offset, limit)
            rows += page
            if len(page) < limit:
                break
            offset += limit
        else:
            logger.warning('%s: cham tran MAX_PAGES=%s, du lieu co the thieu',
                           symbol, self.MAX_PAGES)

        df = self._to_frame(rows, symbol)
        if df.empty:
            logger.warning('%s %s..%s: khong co du lieu', symbol, start_s, end_s)
            return df
        logger.info('%s: %s phien, %s -> %s', symbol, len(df),
                    df['date'].iloc[0].date(), df['date'].iloc[-1].date())
        return df

    def get_prices_many(self, symbols, start_date, end_date=None, sleep=0.5):
        """Lấy giá nhiều mã, gộp thành một DataFrame.

        Mã lỗi chỉ được ghi log cảnh báo, không làm dừng cả vòng lặp. Với số
        lượng mã lớn (vd cả sàn) nên dùng `crawl_to_db` để có log chi tiết và
        chạy tiếp được khi đứt giữa chừng.
        """
        frames = []
        symbols = list(symbols)
        for i, symbol in enumerate(symbols, 1):
            try:
                frames.append(self.get_prices(symbol, start_date, end_date))
            except Exception as exc:  # mã lỗi không làm dừng cả đợt
                logger.warning('(%s/%s) %s loi: %s', i, len(symbols),
                               symbol, exc)
            if i < len(symbols):
                time.sleep(sleep)
        if not frames:
            return pd.DataFrame(columns=self.COLUMNS)
        return pd.concat(frames, ignore_index=True)

    @classmethod
    def add_adjusted(cls, df, drop_raw=False):
        """Thêm cột giá/khối lượng đã điều chỉnh chia tách & cổ tức.

        * `*_adj` (price_open_adj, ..., price_basic_adj): VND, tính bằng
          price * unit / adj_ratio.
        * `*_volume_adj`: khối lượng quy về số cổ phiếu sau chia tách, tính
          bằng volume * adj_ratio.

        Mốc điều chỉnh là phiên gần nhất tại thời điểm FETCH (adj_ratio = 1),
        không phải thời điểm gọi hàm này.
        """
        out = df.copy()
        if out.empty:
            return out
        ratio = out['adj_ratio'].replace(0, pd.NA)
        for col in cls._PRICE_COLS:
            out[f'{col}_adj'] = out[col] * out['unit'] / ratio
        for col in cls._VOLUME_COLS:
            out[f'{col}_adj'] = out[col] * out['adj_ratio']
        return out

    # ------------------------------------------------------------------ #
    # SQLite: dùng chung fscore.db
    # ------------------------------------------------------------------ #

    _TABLE = 'fireant_prices'
    _META_TABLE = 'fireant_prices_meta'

    _DB_SCHEMA = f"""
        CREATE TABLE IF NOT EXISTS {_TABLE} (
            symbol                      TEXT NOT NULL,
            date                        TEXT NOT NULL,
            price_open                  REAL,
            price_high                  REAL,
            price_low                   REAL,
            price_close                 REAL,
            price_average               REAL,
            price_basic                 REAL,
            adj_ratio                   REAL,
            unit                        REAL,
            total_volume                REAL,
            deal_volume                 REAL,
            putthrough_volume           REAL,
            total_value                 REAL,
            putthrough_value            REAL,
            buy_foreign_quantity        REAL,
            buy_foreign_value           REAL,
            sell_foreign_quantity       REAL,
            sell_foreign_value          REAL,
            current_foreign_room        REAL,
            buy_count                   REAL,
            buy_quantity                REAL,
            sell_count                  REAL,
            sell_quantity               REAL,
            prop_trading_net_deal_value REAL,
            prop_trading_net_pt_value   REAL,
            prop_trading_net_value      REAL,
            PRIMARY KEY (symbol, date)
        )
    """

    # adj_ratio luỹ kế tới ngày fetch -> ghi lại mốc để biết dữ liệu điều chỉnh
    # theo thời điểm nào, và để crawl lần sau biết cần cập nhật từ đâu
    _META_SCHEMA = f"""
        CREATE TABLE IF NOT EXISTS {_META_TABLE} (
            symbol     TEXT PRIMARY KEY,
            first_date TEXT,
            last_date  TEXT,
            bars       INTEGER,
            fetched_at TEXT
        )
    """

    @classmethod
    def _init_db(cls, conn):
        conn.execute(cls._DB_SCHEMA)
        conn.execute(cls._META_SCHEMA)

    def save_prices(self, df, db_path='fscore.db'):
        """Lưu DataFrame của get_prices vào SQLite (bảng `fireant_prices`).

        Dùng INSERT OR REPLACE theo khóa (symbol, date) nên chạy lại cùng một
        mã sẽ ghi đè chứ không tạo dòng trùng. Đồng thời cập nhật
        `fireant_prices_meta` (khoảng ngày, số phiên, thời điểm fetch).
        """
        if df.empty:
            logger.warning('DataFrame rong, khong luu gi vao %s', db_path)
            return
        out = df[self.COLUMNS].copy()
        out['date'] = out['date'].dt.strftime('%Y-%m-%d')
        placeholders = ', '.join('?' * len(self.COLUMNS))
        with sqlite3.connect(db_path) as conn:
            self._init_db(conn)
            conn.executemany(
                f'INSERT OR REPLACE INTO {self._TABLE} '
                f'({", ".join(self.COLUMNS)}) VALUES ({placeholders})',
                out.itertuples(index=False, name=None),
            )
            now = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            # meta lấy theo toàn bộ dữ liệu đã có trong DB, không chỉ lần này
            conn.executemany(
                f'INSERT OR REPLACE INTO {self._META_TABLE} '
                f'(symbol, first_date, last_date, bars, fetched_at) '
                f'SELECT symbol, MIN(date), MAX(date), COUNT(*), ? '
                f'FROM {self._TABLE} WHERE symbol = ? GROUP BY symbol',
                [(now, s) for s in out['symbol'].unique()],
            )
        logger.info('Da luu %s dong (%s ma) vao %s.%s', len(out),
                    out['symbol'].nunique(), db_path, self._TABLE)

    def crawl_to_db(self, symbols, start_date, end_date=None, db_path='fscore.db',
                    sleep=0.5, mode='skip', log_path=None):
        """Crawl giá nhiều mã vào SQLite, chạy lại tiếp tục được nếu đứt giữa chừng.

        Thiết kế cho đợt crawl lớn (vd 1800 mã): mọi lỗi của một mã chỉ được
        GHI NHẬN vào df kết quả, không làm dừng đợt crawl. Ctrl+C cũng không
        mất log — hàm dừng và trả về phần đã chạy.

        Parameters
        ----------
        symbols : iterable   - danh sách mã cổ phiếu
        start_date, end_date - khoảng ngày; end_date None = hôm nay
        mode : str           - cách xử lý mã ĐÃ CÓ trong DB:
                               'skip'   bỏ qua (mặc định)
                               'update' chỉ lấy thêm từ phiên cuối đã có
                                        (adj_ratio phiên cũ giữ nguyên, có thể
                                        lệch nếu mã vừa chia tách)
                               'full'   lấy lại toàn bộ khoảng, ghi đè —
                                        dùng khi muốn đồng bộ lại adj_ratio
        sleep : float        - nghỉ giữa các mã (giây) để tránh bị chặn
        log_path : str       - nếu có, ghi log ra CSV sau MỖI mã (file bị ghi
                               đè ở đầu mỗi lần chạy) để không mất log nếu
                               process bị kill giữa đợt

        Returns
        -------
        pd.DataFrame - log kết quả từng mã, một dòng một mã, các cột:
        Symbol | IsSuccess | Bars | FirstDate | LastDate | ErrorMessage.
        Bars là số phiên lấy được trong LẦN CHẠY NÀY; 0 = mã không có dữ liệu,
        bị bỏ qua, hoặc (mode='update') đã cập nhật tới phiên mới nhất.
        """
        if mode not in ('skip', 'update', 'full'):
            raise ValueError("mode phải là 'skip', 'update' hoặc 'full'")
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date) if end_date is not None \
            else pd.Timestamp.today().normalize()
        if start > end:
            raise ValueError('start_date phải <= end_date')

        with sqlite3.connect(db_path) as conn:
            self._init_db(conn)
            existing = {row[0]: row[1] for row in conn.execute(
                f'SELECT symbol, last_date FROM {self._META_TABLE}')}

        symbols = list(symbols)
        cols = ['Symbol', 'IsSuccess', 'Bars', 'FirstDate', 'LastDate',
                'ErrorMessage']
        summary = []
        try:
            for i, symbol in enumerate(symbols, 1):
                row = {'Symbol': symbol, 'IsSuccess': True, 'Bars': 0,
                       'FirstDate': None, 'LastDate': None,
                       'ErrorMessage': None}
                try:
                    last = existing.get(symbol)
                    sym_start = start
                    if last is not None and mode == 'skip':
                        logger.info('(%s/%s) %s da co trong DB, bo qua',
                                    i, len(symbols), symbol)
                        row['ErrorMessage'] = 'Da co trong DB, bo qua'
                        summary.append(row)
                        self._write_log(log_path, row, cols,
                                        first=len(summary) == 1)
                        continue  # không gọi API -> không cần sleep
                    if last is not None and mode == 'update':
                        # lấy lại phiên cuối để bắt luôn dữ liệu bị sửa
                        sym_start = max(start, pd.Timestamp(last))

                    df = self.get_prices(symbol, sym_start, end)
                    self.save_prices(df, db_path)
                    row['Bars'] = len(df)
                    if not df.empty:
                        row['FirstDate'] = df['date'].iloc[0].date()
                        row['LastDate'] = df['date'].iloc[-1].date()
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
    def load_prices(cls, db_path='fscore.db', symbol=None, start_date=None,
                    end_date=None, adjusted=False):
        """Đọc giá từ SQLite ra DataFrame.

        Parameters
        ----------
        symbol : str hoặc list  - lọc theo mã (None = tất cả)
        start_date, end_date    - lọc theo ngày (None = không lọc)
        adjusted : bool         - True thì thêm cột `*_adj` (xem `add_adjusted`)
        """
        query = f'SELECT * FROM {cls._TABLE} WHERE 1=1'
        params = []
        if symbol is not None:
            symbols = [symbol] if isinstance(symbol, str) else list(symbol)
            query += f' AND symbol IN ({", ".join("?" * len(symbols))})'
            params += symbols
        if start_date is not None:
            query += ' AND date >= ?'
            params.append(pd.Timestamp(start_date).strftime('%Y-%m-%d'))
        if end_date is not None:
            query += ' AND date <= ?'
            params.append(pd.Timestamp(end_date).strftime('%Y-%m-%d'))
        query += ' ORDER BY symbol, date'
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql(query, conn, params=params)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            if adjusted:
                df = cls.add_adjusted(df)
        return df

    @classmethod
    def load_meta(cls, db_path='fscore.db'):
        """Đọc bảng meta: mã nào đã crawl, khoảng ngày, số phiên, mốc fetch."""
        with sqlite3.connect(db_path) as conn:
            return pd.read_sql(
                f'SELECT * FROM {cls._META_TABLE} ORDER BY symbol', conn)


if __name__ == '__main__':
    fa = FireantPrices()
    df = fa.get_prices('HPG', '2009-01-01')
    print('HPG:', len(df), 'phien')
    print(FireantPrices.add_adjusted(df)[
        ['date', 'price_close', 'adj_ratio', 'price_close_adj', 'total_volume']
    ].head())
