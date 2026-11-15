# Latest upstream versus our fork

Dashboard: **`/upstream-comparison`**. Reviewed original repository commit: [`49bf9c043d84d779c077f27570ec8243178e3457`](https://github.com/akkansh/aarc-terminal/commit/49bf9c043d84d779c077f27570ec8243178e3457). A subsequent upstream commit, `94bd421f72c4b5d3bb31a532b064dc94fd69d82f`, adds a low-volume scanner but changes none of the trading-model files used in this replay. We reviewed that update too. Previous fork release: `582e3b0ec420fc4b69a82f6e0397f64c9adae040`.

## Conclusion

The newer original code does **not** establish a stronger trading model. On the same 356 frozen COVID/2022 decisions, our prior fork earned more in three of four portfolios. Upstream earned slightly more in India during 2022, with higher exposure and drawdown. Applying our risk limits to its proposals produced the same returns as our previous fork.

We did find a useful execution idea: an exit target at **twice the stop distance (2R)**. Applying that to our existing entries and allocations improved three of four portfolios, without increasing the maximum drawdown in these tests. We implemented it as an **experimental paper exit plan**, alongside two correctness fixes. India COVID performance decreased slightly; this is not a universal improvement or a blind out-of-sample validation.

The full upstream feature replay completed all 356 cases and confirms the conclusion. Directions were unchanged versus matched inputs; three allocation sizes differed by 0.01–0.02 percentage points. Independent accounting reconciled 1,564 active trades across 24 portfolios. The release passed 97 tests plus build and desktop/mobile checks.

## P&L on identical inputs

Each portfolio starts at 10,000 in its own currency. These are **period returns, not annual returns**. USD profits must not be added to INR profits.

| Period / market | Our previous fork | Latest upstream, matched inputs | Updated fork, 2R exit |
|---|---:|---:|---:|
| COVID / US | +0.0954% ($9.54) | -0.2305% (-$23.05) | **+0.1643% ($16.43)** |
| COVID / India | **+0.2252% (₹22.52)** | -0.0967% (-₹9.67) | +0.2061% (₹20.61) |
| 2022 hikes / US | +0.0146% ($1.46) | -0.4260% (-$42.60) | **+0.1195% ($11.95)** |
| 2022 hikes / India | +0.5411% (₹54.11) | **+0.6449% (₹64.49)** | +0.5556% (₹55.56) |

The risk-capped upstream variant equals our previous fork in all four portfolios. Its isolated India advantage is not evidence for increasing position sizes indiscriminately. At 100 bps round-trip cost, all four updated single-stock portfolios lose money. The dashboard exposes that sensitivity.

The **13.59% annualized ETF result** remains a separate six-year allocation experiment; none of these small single-stock returns should be presented as that result. No parameters of that ETF policy were changed here.

## What was actually compared

- **356 frozen cases** across US and Indian stocks in the COVID and 2022 rate-hike episodes. Same date schedule, actual future OHLC/dividend tape, initial capital, non-overlapping holding schedule and costs.
- **Matched-input test:** both repository versions receive the same pre-signal quant snapshots and dated macro-news sentiment. This isolates committee/calibration/decision differences.
- **Full upstream feature replay:** upstream independently recomputes its indicators, GARCH, HMM and anomaly models from the same past-only 252-bar windows. Only the data downloader is substituted with frozen real bars. No synthetic fallback is needed. The report records completion separately, rather than silently dropping cases.
- **Final-output check:** each version's actual `stream_committee` generator renders the final SSE payload, including postprocessing, narratives and probability fields. We test that payload, not merely an internal action label.
- **No hosted LLM calls:** both versions use their deterministic paths, with isolated empty calibration state. The historical news scores are fixed. This does not measure the quality of a private Gemini session running on Akansh's device, or prove historical Qwen profitability. Our live Qwen provider remains configured and unchanged.
- **Common execution:** enter at the next session's open; fixed stop with adverse gap fills; maximum five sessions; dividends for existing holders; assumed 3% annual short borrow; 30-bps round-trip trading costs. No trades count as neither wins nor losses. Cost sensitivity uses 0/30/100 bps.
- **2R candidate:** the target multiple comes from upstream's documented default, not a parameter search over these test outcomes. When stop and target both lie within a daily candle, assume the stop hit first. The stop and target are anchored to actual next-open entry, not the preceding signal close.

The current test periods have been inspected before. Adoption is a retrospective engineering experiment; future paper tracking is still needed. It does not establish statistical significance or future profitability.

## Changes adopted

1. **Experimental 2R exit plan.** The main committee now includes an explicit paper execution contract: next-open entry, fixed stop, take-profit twice that distance away and a fifth-session exit. The UI labels it experimental. The benchmark executor consumes that same contract. Directions and allocation sizes remain identical to the previous fork in the recorded replay: gains come from exits, not higher leverage or inflated risk caps.
2. **Reject conflicting generated verdicts.** Adapted from upstream [`819dd33`](https://github.com/akkansh/aarc-terminal/commit/819dd33). A generated action that contradicts the configured rules is rejected and recomputed. We preserve our additional unsupported-signal rejection and final risk audit; upstream's unconditional default-long behavior is not adopted.
3. **Chronological input handling.** Adapted from upstream's ordering guarantee. Sort before computing indicators, and reject duplicate/invalid session dates. Missing real data continues to fail explicitly.

The optional Gemini-proposal fix is covered with controlled conflicting outputs; the reproducible P&L replay exercises the deterministic branch. It is a correctness improvement, not credited with the 2R return gain.

4. **Daily-volume diagnostic from 94bd421.** We adapted relative volume and low-volume price-move flags using the preceding 20 completed sessions. Zero volume is preserved; zero-variance Z-scores remain unavailable. It is advisory only and never claims to measure order-book depth from daily candles. No synthetic-price fallback or uncalibrated liquidity probability was adopted.

## Why upstream demo results are not accepted as model proof

These are concrete code paths, not claims about the author's intent:

- [COVID illustration](https://github.com/akkansh/aarc-terminal/blob/49bf9c043d84d779c077f27570ec8243178e3457/backend/eval/covid_backtest.py#L347) opens the hedge on a literal March 2, 2020 date and [closes it on March 24](https://github.com/akkansh/aarc-terminal/blob/49bf9c043d84d779c077f27570ec8243178e3457/backend/eval/covid_backtest.py#L361). These are scenario instructions, not decisions reconstructed from the live committee.
- [Signal-backtest loader](https://github.com/akkansh/aarc-terminal/blob/49bf9c043d84d779c077f27570ec8243178e3457/backend/eval/backtest_signals.py#L346) can substitute synthesized forward prices. Such outcomes cannot establish real-market P&L. We used frozen actual prices instead.
- [Quant fallback](https://github.com/akkansh/aarc-terminal/blob/49bf9c043d84d779c077f27570ec8243178e3457/backend/tools/quant_math.py#L441) supplies fixed historical-return numbers when future bars are absent. Those are not observed outcomes. Our P&L uses a separately held actual future tape; the invented labels are flagged by output checks and never used as realized returns.
- [News producer](https://github.com/akkansh/aarc-terminal/blob/49bf9c043d84d779c077f27570ec8243178e3457/backend/data/news_stream.py#L345) cycles a built-in headline catalog. We retain actual dated RSS headlines, source URLs and Groq evidence validation rather than treating that catalog as live regulatory news.
- The [final stream postprocessor](https://github.com/akkansh/aarc-terminal/blob/49bf9c043d84d779c077f27570ec8243178e3457/backend/graph.py#L558) assigns constant class probabilities and conformal sets from action/confidence. Those are not fitted probability estimates. Our fork continues to disclose heuristic scores and avoids fabricated attribution percentages.

Upstream also adds UI, broker and portfolio-scenario features. They are useful feature directions, but not proof of higher returns. This integration does not connect broker accounts, submit orders or replace the verified dashboard with demo paths.

## Evidence and reproduction

`benchmarks/upstream-comparison/report.json` contains all portfolio curves, cost sensitivities and check counts. `cases.json.gz` contains every final answer and trade outcome. `verification.json` independently reconstructs exit prices, dividend/borrow/cost arithmetic and final balances. `AARC-upstream-comparison-evidence.zip` preserves feature outputs, input hashes and source manifests.

The original input tape is in `benchmarks/evidence/AARC-crisis-evidence.zip`. Extract only its `stress/` subtree beneath `benchmark-runs/`, giving `benchmark-runs/stress/`; do not overwrite current code with the older archived `backend/` or `benchmarks/` copies. Extract the new upstream evidence archive at the repository root. It omits duplicate chart candles but preserves model features, final answers, and both packed/original hashes. Use a separate checkout/worktree pinned to upstream `49bf9c0`; do not import an arbitrary unreviewed branch as executable code.

```bash
python backend/upstream_worker.py --repository ../aarc-upstream-review --input-root benchmark-runs/stress --output benchmark-runs/upstream-comparison/upstream --recompute-quant
python backend/upstream_stream_review.py --repository ../aarc-upstream-review --folder benchmark-runs/upstream-comparison/upstream
python -m backend.upstream_comparison
python -m backend.upstream_verify
python -m pytest backend/tests -q
```

The evidence archive supplies matched and prior-fork runs for recomputing the report. To independently regenerate those, run the worker and final-stream renderer against the pinned previous fork and upstream checkouts without `--recompute-quant`. `--refresh` explicitly reruns saved worker cases. The updated fork uses the same frozen quant/news inputs and its current stream renderer. Runtime state lives in temporary directories; the user's credentials and demo recordings are excluded.
