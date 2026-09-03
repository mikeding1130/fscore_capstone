# Piotroski F-Score Capstone Project

This repository contains the source code and supporting materials for our MScFE 690 Capstone Project on the Piotroski F-Score, high book-to-market stock selection, and portfolio construction in the United States, Japan, and Vietnam.

## Final Submission

The project changed over the course of the earlier milestones, so earlier submissions, notebooks, and experimental code are not identical to the final design. For the final submission, please use the **`consolidated_report` branch** and the final country pipelines and results referenced there. Where an earlier file differs from the final report, the **final report and consolidated results should be treated as the source of truth**.

The main final-report files are located under:

- `consolidated_report/` — final report-generation notebooks and supporting code
- `consolidated_report/resources/` — final tables and figures used in the report
- `results/` — final country-level portfolio and Monte Carlo outputs
- `src/` — reusable project source code, including the common F-Score pipeline and the Vietnam-specific data-processing path
- `tests/` — reproducibility and implementation checks

The main GitHub repository has the URL https://github.com/mikeding1130/fscore_capstone

## Final Methodology

The final study first restricts each market to a high book-to-market universe, then ranks stocks using the Piotroski F-Score. Portfolios are constructed using equal weighting, RMT-denoised global minimum variance, and sector-capped global minimum variance. The main analysis uses `K = 30`, with `K = 20` and `K = 25` used as robustness cases. Matched random portfolios from the same annual high-B/M pool (the *null portfolios*) are used as Monte Carlo null benchmarks.

Portfolios are formed annually on July 1 and held through June 30 the following year. Performance is reported gross of transaction costs, with turnover reported as well but separately.

## Running the Project

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

The final report can be reproduced from the notebooks in `consolidated_report/` after the required prepared country inputs and result files are placed in their expected directories. Country-specific data preparation is documented in the corresponding source folders and notebooks.

Some raw inputs, particularly licensed Bloomberg data and locally collected Vietnam data, are not redistributed. Prepared or derived inputs required for the submitted analysis are included where permitted.

## Authors

- Ta Tan Phat
- Zhicheng Ding
- Zu Yao Teoh
