# Historical OHLCV benchmark

Market india, model regularized_logistic_all_blend_1, horizon 5 sessions.
Train through 2018-12-31, validate through 2019-12-31, test 2020-01-01 to 2021-01-01.

| Model | Accuracy | Log loss | Brier | ECE |
|---|---:|---:|---:|---:|
| class_prior | 0.417 | 1.0835 | 0.6563 | 0.0193 |
| momentum | 0.317 | 1.2612 | 0.7865 | 0.2831 |
| selected_raw | 0.413 | 1.1760 | 0.6915 | 0.0880 |
| selected_calibrated | 0.413 | 1.1230 | 0.6727 | 0.0365 |

Improves on class-prior log loss and Brier score: False.
Research candidate only; requires recent, untouched data and verified price basis.
Previously inspected test periods are development comparisons; reserve fresh dates/universe for confirmation.
Validation-selected model weight 1.0, temperature 1.5.

## Scope and assumptions

- Selected historical universe can have survivorship and selection bias.
- Source corporate-action adjustment basis is unverified.
- Price/volume only; historical news and point-in-time fundamentals excluded.
- Test labels overlap for classification; effective sample size is lower than row count.
- Strategy uses non-overlapping long/cash trades and exit-only drawdown, excludes dividends and overnight gaps between trades.
- Historic test performance does not establish current-market performance.
