# Historical training and benchmarks

This workflow measures a five-session OHLCV classifier. It also exposes its actual probabilities in the dashboard when an operator configures a local model artifact. The committee still performs its own risk analysis. Historical price data cannot evaluate the news debate without contemporaneous headlines.

## Data

Download the ZIP archives from:

- https://www.kaggle.com/datasets/hk7797/stock-market-india
- https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs

Review the source's terms before redistributing its data. Raw archives, normalized data and binary models are excluded from Git. Source files and normalized CSVs are SHA-256 fingerprinted in a manifest.

The Indian archive contains minute bars. The importer aggregates in Asia/Kolkata calendar sessions, using first open, maximum high, minimum low, last close and summed volume. All-empty placeholder rows are omitted and counted, including affected session counts. Partial/nonfinite rows and contradictory prices reject the series. Aggregates can therefore contain incomplete sessions. The source's corporate-action adjustment basis remains unverified. US files are daily bars. Their snapshot ends in 2017; the Indian sample ends in January 2021.

## Install

Use Python 3.14 with the pinned benchmark dependencies in backend/requirements-historical.txt for the recorded experiments, or the existing backend virtual environment. The full dashboard additionally needs its README dependencies.

```sh
python -m pip install -r backend/requirements-historical.txt
python -m backend.historical --help
```

## Ingest and benchmark

```sh
python -m backend.historical ingest --archive india.zip --market india --output datasets/india --symbols RELIANCE TCS HDFCBANK INFY ICICIBANK ITC SBIN HINDUNILVR LT MARUTI
python -m backend.historical ingest --archive us.zip --market us --output datasets/us --symbols AAPL MSFT AMZN GOOGL NVDA JPM XOM JNJ WMT PG
python -m backend.historical benchmark --data datasets/india --train-end 2018-12-31 --validation-end 2019-12-31 --output benchmark-runs/india
python -m backend.historical benchmark --data datasets/us --train-end 2012-12-31 --validation-end 2014-12-31 --output benchmark-runs/us
```

Omit --symbols to ingest the whole archive, or use --limit for a deterministic initial subset. The initial twenty-stock universe is a convenience sample of recognizable surviving companies and is subject to selection/survivorship bias. Results apply to that sample. Inspect manifest.json for rejected series and missing bars before using a run.

Each market has its own pooled model. Features reuse the repository's past-only technical indicators. Signals are formed after the close. Labels measure next-session open to the fifth subsequent session close, with DOWN/FLAT/UP thresholds based on trailing volatility. Labels crossing training or validation boundaries are removed. All symbols share calendar boundaries, preventing later market dates from training a prediction on an earlier date in another ticker.

Model selection now uses three expanding-window folds entirely within the training interval. Labels crossing each fold boundary are purged. Candidates include logistic regression, stronger regularization, 252-session and 756-session training windows, and a constrained seven-leaf LightGBM model. Average per-symbol log loss across these folds chooses the candidate, giving each symbol equal weight in the selection score.

The later validation interval chooses how much to blend the learned probabilities with the training class-frequency baseline and selects a temperature. The untouched baseline is always an eligible fallback. Ties favor less model influence. This ensures the selected combination is at least as good as the baseline on validation macro log loss; it makes no guarantee about later observations. The model emits ABSTAIN when the baseline alone is selected or the largest forecast probability is below 0.5. A baseline-only forecast always holds cash in the strategy illustration.

The test period is used for reporting, and never passed to model selection. These particular test periods were inspected during the first implementation, so the version-two results are development comparisons. Reserve fresh dates or a separately defined new universe for independent confirmation. A new universe from the same dates still shares the same market regime.

Reports include class-prior and momentum baselines, raw and calibrated model results, per-symbol results, macro-symbol log loss, accuracy, multiclass Brier score and ECE. Classification labels overlap, so row count overstates independent evidence.

The strategy illustration uses non-overlapping, fully allocated long/cash trades independently for each symbol, entry next open and exit at the horizon close. It charges 30 basis points per active round trip by default. Its comparator is always-long on the same trading schedule, including the same costs. It reports exit-only drawdown and omits overnight returns between trades and dividends. It is an execution illustration; the committee sizing and stops are not simulated.

Each run writes report.md, report.json, predictions.csv, trades.csv, data-manifest.json and a frozen model.joblib with JSON metadata. The report says whether log loss AND Brier improve over the class-prior baseline. Results alone never automatically promote a model.

## Retraining for later observations

```sh
python -m backend.historical train --data datasets/india --through 2021-01-01 --output backend/artifacts/historical-india.joblib
python -m backend.historical train --data datasets/us --through 2017-11-10 --output backend/artifacts/historical-us.joblib
python -m backend.historical predict --model benchmark-runs/india/model.joblib --csv datasets/india/RELIANCE.csv --market india --as-of 2020-12-31
```

Train reserves the latest 20% of signal dates for blending/calibration and purges boundary-crossing training labels. The earlier portion supplies the three temporal selection folds. It produces a deployment candidate. Extend normalized data by ingesting an updated source archive, benchmark against a new frozen future interval, and retrain with an explicit cutoff. The command rejects training dates beyond available observations.

Inference requires an observation later than every training/calibration observation and a matching market. Failed or unavailable forecasts do not stop the committee. The output carries training date, model age, prediction horizon and actual probabilities. Models trained only on these old snapshots need current validation before a present-day deployment. An old snapshot cannot establish performance in the current regime.

## Dashboard

Set AARC_MODEL_INDIA and/or AARC_MODEL_US to absolute paths of locally trained model.joblib files, then restart the backend. These are trusted local pickle artifacts; never load an untrusted downloaded artifact. The dashboard shows the optional historical forecast panel, its training cutoff and age. Without configuration it displays an unavailable explanation. Its probabilities remain separate from committee confidence.

The live Yahoo price basis may differ from the Kaggle source. Validate adjustment consistency before promoting a candidate. The news and fundamentals pipeline is excluded from this price-only benchmark.

## Tests

```sh
python -m pytest backend/tests -q
```

Regression coverage checks causal feature invariance, entry timing, boundary purging, minute aggregation, quality counters, duplicate rejection, non-overlapping trades, transaction costs, zero profit for cash, model market/date guards and fallback behavior.

## Recorded initial results

The initial [US report](benchmarks/us/report.md) and [India report](benchmarks/india/report.md) are retained. Updated [US version-two report](benchmarks/us-v2/report.md) and [India version-two report](benchmarks/india-v2/report.md) include the temporal selection audit in their JSON files. Source manifests are beside each report.

| Market | Initial model log loss | Updated model log loss | Class-prior log loss |
|---|---:|---:|---:|
| US | 1.0372 | 1.0372 | 1.0546 |
| India | 1.1283 | 1.1230 | 1.0835 |

Lower log loss is better. The US result is unchanged. India improves slightly with stronger regularization but remains worse than its baseline. Both remain research candidates. The sample contains ten equities per market. Broader market or future performance has not been established.

For a separate frontend/backend port pair, set NEXT_PUBLIC_API_BASE_URL before starting or building Next.js. For example, a backend on port 8001 requires http://localhost:8001.
