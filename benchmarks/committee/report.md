# Whole-committee replay benchmark

Production quant + full streamed rules-fallback committee + final output. Gemini and archived-news truth untested.

Same causal snapshots; fresh calibration state; next-open entry; five-session fixed-stop exit; no outcome tuning.

| Market | Version | Cases | Trades | Net win rate | Mean allocated 5-session return |
|---|---|---:|---:|---:|---:|
| india | before | 40 | 23 | 69.6% | 0.069% |
| india | after | 40 | 17 | 82.4% | 0.069% |
| us | before | 40 | 33 | 60.6% | 0.027% |
| us | after | 40 | 21 | 71.4% | 0.055% |

## Final-output consistency (passes / cases)

- india before: allocation_within_risk_cap: 29/40; summary_matches_allocation: 12/40; zero_exposure_is_no_trade: 23/40; no_unsupported_institutional_flow_claim: 0/40; no_unfitted_probability_claim: 0/40; score_disclosed_as_heuristic: 0/40
- india after: allocation_within_risk_cap: 40/40; summary_matches_allocation: 40/40; zero_exposure_is_no_trade: 40/40; no_unsupported_institutional_flow_claim: 40/40; no_unfitted_probability_claim: 40/40; score_disclosed_as_heuristic: 40/40
- us before: allocation_within_risk_cap: 39/40; summary_matches_allocation: 30/40; zero_exposure_is_no_trade: 33/40; no_unsupported_institutional_flow_claim: 0/40; no_unfitted_probability_claim: 0/40; score_disclosed_as_heuristic: 0/40
- us after: allocation_within_risk_cap: 40/40; summary_matches_allocation: 40/40; zero_exposure_is_no_trade: 40/40; no_unsupported_institutional_flow_claim: 40/40; no_unfitted_probability_claim: 40/40; score_disclosed_as_heuristic: 40/40

## What these numbers mean

A win is a positive return after the assumed execution costs on an allocated trade. No-trade decisions are excluded from win rate, but remain in coverage and mean allocated return. A check passing means a specific contradiction or unsupported claim was absent; it is not proof that the recommendation will make money.

## Coverage limits

- Selected surviving large caps, four dates per symbol: pilot, not a representative market estimate.
- No archived news, Gemini calls, measured market impact, borrow constraints, dividends or verified corporate-action adjustments.
- Consistency checks test specific known failure classes; they do not measure all factual truth.
- Mean allocated returns are per-decision outcomes, not a compounded portfolio backtest.
- Snapshots are paired/correlated. Win-rate intervals treat trades independently and are descriptive only.