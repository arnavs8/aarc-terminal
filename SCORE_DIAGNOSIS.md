# Score discrepancy: 40 versus 80–90

The 40-point score was caused by contaminated local calibration state, not a new ceiling or a change from percentages to points.

The existing API test posted one artificial HIT and one heavily penalized MISS for TCS.NS into production storage each time the suite ran. Five repetitions exactly reproduce the observed state: global and TCS alpha=1.3956, beta=7.826, updates=10. Other tickers inherit that global posterior, so the effect extended to RELIANCE. The stored prior-adjusted reliability was approximately 15.1%. These were synthetic test events, not ten observed market outcomes. Earlier verification runs triggered this bug.

Identical saved RELIANCE.NS evidence as of 2026-11-13, deterministic rules mode:

| Version | Calibration state | Raw score | Final score | Allocation |
|---|---|---:|---:|---:|
| Original checkout dcae346 | Contaminated | 88 | 40 | 5.49% |
| Revised fork | Contaminated | 88 | 40 | 5.49% |
| Original checkout dcae346 | Clean prior | 88 | 84 | 10.97% |
| Revised fork | Clean prior | 88 | 84 | 9.54% |

This isolates the local cause. It does not inspect Akansh's device, API configuration or market inputs. The original and fork scoring formulas are unchanged. The fork continues to enforce the 9.54% risk cap even where the original would exceed it.

## Fix

- All backend tests now use a fresh temporary AARC_STATE_DIR per test. API requests, rolling prediction logs, Bayesian updates and demo-cache writes are isolated.
- Default calibration storage resolves relative to the backend rather than the shell's working directory. Explicit storage paths are still supported for benchmarks and deliberate isolated experiments.
- Default storage refuses pytest execution without an isolation directory.
- Regression tests exercise the real confidence-update endpoint and verify that the production file remains byte-for-byte unchanged. A clean-state regression confirms that scores above 80 remain possible.
- The UI shows both raw and final heuristic scores, plus the number of calibration updates.

The contaminated local Bayesian file and rolling log were backed up under ignored benchmark-runs/state-repair-2026-11-15 before repair. The Bayesian state was restored to its existing Beta(7,3) prior only after reproducing the exact synthetic sequence. Four identifiable unresolved fixture predictions were quarantined; the real pending RELIANCE prediction was preserved. No real resolved losses were discarded, and no score floor was raised.

A clean prior is an assumption, not evidence of 70% accuracy. A final 84/100 remains a heuristic score, not an 84% probability of profit. Genuine panic, choppy conditions, tail risk or subsequently verified poor performance can still reduce it.

If another checkout is affected, inspect and back up its state before repair; do not blindly reset calibrated market outcomes merely to obtain a higher score.
