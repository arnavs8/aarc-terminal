# Crisis replay: evidence, decisions and outcomes

Open `/stress-test` in the running frontend. It includes all scheduled decisions for February–June 2020 and the full 2022 rate-hike year, in US and Indian instruments. Choose a market, period and decision to inspect the evidence, exact committee answer and next five price bars. Download the standalone offline report from that page.

Download the [complete frozen evidence archive](benchmarks/evidence/AARC-crisis-evidence.zip) from the fork (GitHub: View raw / Download). Archive SHA-256: `65723645950fad76733db7372104f7acaf9d55810a3e7a03c0d20f388ff2305f`.

Read [the results](benchmarks/stress/report.md), [machine-readable summary](benchmarks/stress/report.json), [independent reconciliation](benchmarks/stress/verification.json) and [trade ledger](benchmarks/stress/trade-ledger.csv). Results are simulations after stated costs, not actual fills.

## What was tested

Each historical date runs production GARCH, regime estimation, Isolation Forest and LSTM on past prices, actual FinBERT on eligible curated official macro summaries, and the production rule-based committee plus final evidence reviewer. Gemini was unavailable and is explicitly **not evaluated**. The original graph and revised graph receive identical quant/news evidence and isolated, frozen calibration priors. This compares committee logic; it is not a reconstruction of the original author's entire historical deployment.

There are 356 regularly spaced decisions, eight selected surviving instruments, four separately funded market/period portfolios. COVID signals are five sessions apart; 2022 signals are ten sessions apart. The test always reserves five future bars and never gives those bars to the model. No-trades remain in the denominator and are not counted as winning trades.

Three different questions are measured separately:

1. **Is the answer internally supported?** Specific checks cover stated versus actual allocation, risk caps, unsupported claims, score disclosure and information cutoffs. These are targeted checks, not a universal fact-checking score.
2. **Did the recommendation make money?** Next-open entry, fixed stops with adverse opening-gap fills, dividends, 0.30% round-trip costs and assumed 3% annual short borrow. Idle money remains cash. Compare against cash, buy-and-hold, always-long at matched sizing, and no-shorts.
3. **Did the separate price classifier predict well?** Accuracy and probability losses versus a baseline fitted before the test. These classifier probabilities currently do not drive the CRO trade decision.

The macro-news-removal comparison is explicitly exploratory: it was added after observing FinBERT labels, not predeclared as a confirmation test. It does not change production behavior. A positive sentiment label on a rate-hike summary is not proof of a positive economic impact.

## Training and model changes

Four research classifier artifacts are in `benchmarks/stress/models/`, with selection and calibration metadata. COVID models use data ending in December 2019; 2022 models use data ending in December 2021. Model selection occurs inside those earlier data. These are frozen historical research artifacts, not automatic replacements for the live models.

The production improvements being measured were made before this crisis run: evidence-grounded bull/bear arguments, final answer/allocation reconciliation, risk-cap enforcement, explicit unavailable news, heuristic-score disclosure, and test isolation for persistent calibration. This release adds a reproducible evaluation and an inspectable report. It does not claim that higher profits have been proved merely because output consistency improved.

## Reproduce from the repository root

Use the project's existing Python dependencies, plus `matplotlib` for the offline report. Exact runtime versions and the cached FinBERT revision are recorded in `benchmarks/stress/environment.json`. Results depend on installed model/library versions; downloaded prices may be revised. The supplied local evidence ZIP preserves the exact data, models, plan, cases and hashes from this run. Extract its `stress/` directory under `benchmark-runs/` for exact source-data inspection.

```bash
python -m backend.stress_data download
python -m backend.stress_data train
python -m backend.stress_benchmark news
python -m backend.stress_benchmark run
python -m backend.stress_ablation
python -m backend.stress_report
python -m backend.stress_verify
python -m backend.stress_export
python -m pytest backend/tests -q
```

The runner resumes saved cases when fingerprints match, and caches expensive quant snapshots. Do not change the frozen config inside an existing run. Reporting refuses missing cases or recorded failures. `stress_verify` independently reconciles original/current scalar trade returns and compounded portfolio endpoints against the frozen price tape; it does not reuse the portfolio-accounting functions.

The offline export embeds all 356 compact cases and four comparison charts. The local ZIP additionally contains full streamed answers, raw normalized CSV inputs and trained artifacts. It is written to `../../outputs` relative to the repository root. Public compact evidence is also in `frontend/public/stress-cases-*.json`.

## Interpretation limits

Selected instruments and known crises are not a representative market sample. Yahoo data is retrospectively delivered and split-adjusted; the curated news set is not an exhaustive point-in-time company-news archive. Current FinBERT weights are not a vintage-2020 deployment. Hypothetical fractional positions, borrow availability, Indian overnight shorts and actual trading liquidity have not been verified. Taxes, margin funding, FX and cash interest are omitted. See the report for the complete assumptions.

Do not turn a heuristic rating or this small sample's win rate into a claimed probability of future profit. Further rule changes should be judged on a new untouched period, not selected and advertised on these same crisis outcomes.
