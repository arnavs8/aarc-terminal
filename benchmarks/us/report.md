# Historical OHLCV benchmark

Market us, model logistic, horizon 5 sessions.
Train through 2012-12-31, validate through 2014-12-31, test 2015-01-02 to 2017-11-10.

| Model | Accuracy | Log loss | Brier | ECE |
|---|---:|---:|---:|---:|
| class_prior | 0.470 | 1.0546 | 0.6370 | 0.0408 |
| momentum | 0.338 | 1.2382 | 0.7697 | 0.2621 |
| selected_raw | 0.471 | 1.0435 | 0.6284 | 0.0472 |
| selected_calibrated | 0.471 | 1.0372 | 0.6241 | 0.0166 |

Improves on class-prior log loss and Brier score: True.
Research candidate only; requires recent, untouched data and verified price basis.

## Scope and assumptions

- Selected historical universe can have survivorship and selection bias.
- Source corporate-action adjustment basis is unverified.
- Price/volume only; historical news and point-in-time fundamentals excluded.
- Test labels overlap for classification; effective sample size is lower than row count.
- Strategy uses non-overlapping long/cash trades and exit-only drawdown, excludes dividends and overnight gaps between trades.
- Historic test performance does not establish current-market performance.
