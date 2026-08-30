# Project & Experiment Handoff — Piotroski F-Score Capstone

## 1. Mục tiêu project

Project kiểm định Piotroski F-Score trên ba thị trường US, Japan và Vietnam, với hai câu hỏi tách biệt:

1. **Selection effect:** F-Score có chọn được cổ phiếu tạo hiệu quả tốt hơn các basket ngẫu nhiên và các rule đơn giản khác không?
2. **Construction effect:** Sau khi danh sách cổ phiếu đã cố định, Equal Weight, GMV và sector-constrained GMV cho kết quả khác nhau thế nào?

Đây là một **portfolio backtest với Monte Carlo benchmark**, không phải mô hình dự báo F-Score ở cấp công ty và không phải thiết kế causal.

Biến kết quả chính của nghiên cứu là **gross Sharpe ratio**. Các biến phụ gồm annualized return, annualized volatility, maximum drawdown, turnover, cost drag, net Sharpe, effective number of holdings, percentile/p-value so với random portfolios và Fama–French alpha.

## 2. Cấu trúc repository

| Thành phần | Vai trò |
|---|---|
| `src/fscore/data/` | Chuẩn hóa fundamentals, prices, universe, sector và point-in-time alignment |
| `src/fscore/signal/piotroski.py` | Tính chín Piotroski signals và F-Score 0–9 |
| `src/fscore/selection/baskets.py` | Tạo F-Score, random, value, market-cap và liquidity-matched baskets |
| `src/fscore/construction/weights.py` | Equal Weight, long-only GMV, sector-GMV và RMT covariance cleaning |
| `src/fscore/evaluation/` | Performance metrics, Monte Carlo placement, benchmarks và FF3 regression |
| `src/fscore/pipeline.py` | Runner của main study: `run_study()` |
| `src/fscore/grid.py` | Runner của robustness/grid study: `run_grid()` |
| `notebooks/` | Entry points chạy nghiên cứu và sinh bảng/biểu đồ |
| `scripts/` | Build data, build/execute notebooks và sensitivity analyses |
| `results/` | CSV và figures đã chạy |
| `tests/` | Các test cho signal, PIT, GMV, turnover, delisting, tie-break và significance |

Luồng thực nghiệm tổng quát:

```text
Fundamentals năm trước formation + lịch sử giá trước formation
    -> universe đủ điều kiện
    -> tính/xếp hạng F-Score
    -> chọn basket
    -> EW / GMV / sector-GMV
    -> mua ngày 1/7 năm T
    -> buy-and-hold đủ 12 tháng
    -> rebalance vào formation tiếp theo
    -> nối các holding years
    -> tính Sharpe và so với phân phối random portfolios
```

Covariance cho GMV sử dụng 36 tháng daily returns kết thúc trước formation. Covariance được Marchenko–Pastur denoise; detoning mặc định tắt. Portfolio được để trọng số drift trong năm, không rebalance hàng ngày.

## 3. Các file chạy thực nghiệm chính

### Main study

- `notebooks/03_us_full_study.ipynb`: main study US.
- `notebooks/04_japan_full_study.ipynb`: main study Japan.
- `src/fscore/pipeline.py`: triển khai `run_year()` và `run_study()`.
- `scripts/build_notebooks.py`: tạo/chạy main notebooks.

### Robustness/grid study

- `notebooks/grid/us_grid.ipynb`: grid US.
- `notebooks/grid/japan_grid.ipynb`: grid Japan.
- `notebooks/grid/vietnam_grid.ipynb`: grid Vietnam — chín cell giống hệt US/Japan, thay cho notebook `vietnam_k25_mc1000.ipynb` cũ.
- `src/fscore/grid.py`: triển khai `run_grid_year()` và `run_grid()`.
- `scripts/build_grid_notebooks.py`: tạo/chạy grid notebooks cho cả ba thị trường.

### Sensitivity/robustness khác

- `scripts/full_period.py`: chạy theo toàn bộ khoảng thời gian khả dụng của từng nước.
- `scripts/eq_offer_sensitivity.py`: sensitivity cho cách xác định equity issuance.
- `notebooks/01_us_fscore_single_year.ipynb` và `02_japan_fscore_single_year.ipynb`: demo synthetic một năm, không phải kết quả nghiên cứu chính.

## 4. Hai họ thực nghiệm không được trộn lẫn

### 4.1 Main study — `pipeline.py`

Thiết kế mặc định hiện tại:

- Lấy tối đa 150 cổ phiếu thanh khoản.
- Giữ 40% cổ phiếu có book-to-market cao nhất (`value_quantile=0.4`).
- Chọn top 30 F-Score.
- So sánh F-Score EW với random EW, value EW, market-cap EW và liquidity-matched EW.
- Áp dụng GMV và sector-GMV lên basket F-Score.
- Random control cũng được chạy qua cùng construction method.
- US chạy formation 2012–2024; Japan main study chỉ chạy 2023–2024.

Null hypothesis chính của selection test:

> Sharpe của F-Score EW không cao hơn Sharpe của random EW baskets trong high-B/M universe.

Do random main study nằm trong value subset, phép so sánh dự kiến đo phần đóng góp của F-Score **sau khi đã kiểm soát value exposure**.

### 4.2 Grid study — `grid.py`

Grid xếp hạng trên toàn bộ scoreable universe có đủ price history, không lọc high-B/M trước khi chọn F-Score.

Grid (cả ba thị trường):

- Basket size `k in {20, 25, 30}`.
- Monte Carlo draws `N in {1000, 2000, 5000}`.
- Formation 2012–2024.
- Fresh random baskets mỗi formation year.
- Có thêm strict `F >= 8` portfolio.
- Có full-universe EW/GMV controls.
- Có random pool loại các F-Score picks.
- Có long top-k/short bottom-k ở cả ba thị trường. Ở Vietnam đây là **giả định, không phải danh mục giao dịch được**: bán khống cổ phiếu thường không khả dụng trên HOSE/HNX, nên `fscore_LS` chỉ dùng để tách xem đầu *thấp* của bảng xếp hạng có mang thông tin hay không, chứ không phải một chiến lược ai đó có thể chạy. `fscore.markets.is_hypothetical_short` đánh dấu điều này và mọi báo cáo in dòng đó phải nói kèm.

Grid có direct synergy statistic:

```text
D = Sharpe(GMV) - Sharpe(EW)
```

Sau đó đặt `D_FScore` vào phân phối `D` của random baskets. Đây mới là phép kiểm định trực tiếp cho câu hỏi optimizer có tạo hiệu quả đặc biệt khi kết hợp với F-Score hay không.

Các N=1000/2000/5000 dùng cùng seed và là các sample lồng nhau. N chỉ làm tăng độ phân giải của p-value; chúng không phải ba replication độc lập.

## 5. Định nghĩa metric trong code

Trong `src/fscore/evaluation/backtest.py`, metric được tính như sau:

```text
CAGR = compounded total return annualized
annualized volatility = daily standard deviation * sqrt(252)
Sharpe = (CAGR - annual risk-free rate) / annualized volatility
```

Mặc định `rf_annual = 0`. Vì numerator dùng CAGR thay vì annualized arithmetic mean, metric nên được diễn giải chính xác là **CAGR-to-volatility ratio**, dù code gọi là Sharpe.

Monte Carlo one-sided p-value:

```text
p = tỷ lệ random portfolios có statistic >= statistic của F-Score portfolio
```

Mức ý nghĩa cố định là 5%.

Chi phí giao dịch:

```text
annual cost drag = 2 * one-way turnover * 0.20% per side
```

Long-short còn chịu stock-borrow fee 1% trên short notional. Gross performance là headline; net performance là sensitivity.

Effective number of holdings:

```text
effective_n = 1 / sum(w_i^2)
```

## 6. Kết quả hiện tại

### 6.1 US main study

Từ `results/us_summary.csv` và `results/us_mc_placement.csv`:

- F-Score EW Sharpe: khoảng 0.811.
- Random Sharpe percentile: 47.3%.
- Selection p-value: 0.527, không significant.
- Value EW Sharpe: khoảng 0.857.
- SPY Sharpe: khoảng 0.847.
- F-Score GMV Sharpe: khoảng 0.736.
- F-Score sector-GMV Sharpe: khoảng 0.918.
- GMV-vs-random-GMV p-value: 0.030.
- Sector-GMV-vs-random-sector-GMV p-value: khoảng 0.003.

Diễn giải quan trọng:

```text
D_GMV = 0.736 - 0.811 = -0.075
```

GMV F-Score significant so với random-GMV không có nghĩa GMV cải thiện chính basket F-Score. Trong US main study, plain GMV làm Sharpe F-Score giảm. Sector-GMV làm Sharpe tăng, nhưng portfolio tập trung mạnh; effective N khoảng 8.8 trên nominal k=30.

### 6.2 US/Japan grid

Không cell nào vượt ngưỡng 5% trong grid US/Japan.

- US F-Score EW ở khoảng percentile 16–36%, dưới random median.
- US `D` âm ở mọi k: khoảng -0.216, -0.197 và -0.178.
- Japan percentile khoảng 47–74%, không significant.
- Japan `D` khoảng -0.203, -0.086 và +0.015, không significant.

Kết luận từ grid: selection edge không ổn định và optimizer không tạo synergy ổn định với F-Score.

### 6.3 Vietnam artefacts hiện có

Đọc từ `results/grid/vietnam_k25_mc1000_*` (k=25, formation 2012–2024, 13 holding year đủ 12 tháng). Toàn bộ notebook Vietnam đã được regenerate và chạy lại; xem `VIETNAM_RERUN_NEEDED.md`.

- F-Score EW Sharpe: 1.318 (ann return 20.9%).
- F-Score EW gross Sharpe p-value so với random full universe: 0.010, **significant**.
- Gross Sharpe p-value so với random non-F-Score pool: 0.011, **significant**.
- Annual-return p-value: 0.038, **significant**.
- F-Score GMV Sharpe: 1.769.
- Sector-GMV Sharpe: 1.508.
- Universe-EW Sharpe: 1.285. Universe-GMV Sharpe: 2.492.
- `D_FScore = 0.451`.
- Synergy p-value: 0.14, không significant.
- **Long-short (`fscore_LS`)**: Sharpe 0.461 gross, 0.362 net; ann return 8.1%. Ổn định trên cả chín cell của grid (0.41–0.46 gross, 0.30–0.36 net).

Diễn giải:

> Vietnam F-Score EW có gross Sharpe cao hơn random ở mức 5% một cách ổn định, nhưng vẫn chưa có bằng chứng GMV tạo synergy đặc biệt với F-Score (`D_p = 0.14`). Universe-GMV vẫn có Sharpe cao hơn F-Score GMV, nên phải so sánh với control này trước khi kết luận F-Score có giá trị thực tiễn. Ở `k=30` thì F-Score EW (1.184) **thấp hơn** universe EW (1.285) và không còn significant (p = 0.105) — kết luận nhạy với k, không nhạy với N.

> `fscore_LS` là thị trường duy nhất mà spread high-minus-low sống sót sau chi phí (net 0.36, so với −0.29 ở US và 0.24 ở Japan). Nhưng đây là **giả định, không giao dịch được** — xem mục 4.2.