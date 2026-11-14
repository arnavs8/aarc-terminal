# Evaluating the complete committee answer

Open `/benchmark` in the frontend to compare the recorded old and revised committee outputs, their evidence checks, and simulated outcomes. The terminal links to this page.

The evaluation has two separate questions:

1. **Does the answer agree with its evidence and final risk controls?** Six checks catch allocation-cap breaches, written allocation contradictions, directional approvals with zero exposure, unsupported institutional-flow claims, fixed probabilities presented as calibrated, and undisclosed heuristic confidence scores.
2. **Does the allocated recommendation work later?** Enter at the next session's open, hold at most five sessions, apply a fixed stop with adverse gap fills, and deduct 30 basis points per round trip. Report coverage, net trade win rate, and mean allocation-weighted return across all cases. No trade earns zero; it is never counted as a winning prediction.

Example: a 10% position with a -10% trade return contributes -1% to capital before costs. A zero-sized SHORT before a decline earns zero. Correct direction alone does not establish profitable execution.

## What changed in the product

- Added a deterministic final evidence/risk reviewer (`backend/committee_audit.py`) after the CRO proposal. Both the streaming and consolidated endpoints use it.
- Rebuild the final written verdict after sizing and regime gates. Clamp allocation to the base and regime-adjusted cap, with no minimum allocation allowed to override the cap. Zero allocation becomes `REJECT_TRADE`, while `proposed_action` preserves the earlier direction for inspection.
- Reject a default LONG when the asset is below SMA20 and there is no positive news evidence. Reject missing/nonfinite essential risk inputs or invalid stop distances.
- Replace free-form bull/bear generation with evidence templates that distinguish observations from hypotheses. This deliberately reduces narrative freedom: there is no evidence for inferring institutional buying/selling from an anomaly count. Gemini can still propose the CRO decision when configured, subject to existing rules and the final reviewer.
- Remove the stream's fixed UP/FLAT/DOWN probabilities, invented feature-impact percentages and unvalidated conformal prediction sets. Label the remaining committee score as a heuristic, not a success probability. The consolidated endpoint retains `calibrated_confidence` as null and adds `heuristic_score` and `confidence_kind`.
- Label actual generation mode, remove hardcoded Gemini and GPU claims, preserve news source URLs, and reject stale (>30 days), future or undated news. Date-only historical news uses midnight UTC conservatively; this can omit relevant same-day pre-close stories.
- Streamed regime labels now use the actual HMM result instead of fabricated volatility-based defaults.
- Live outcome tracking logs the final approved direction only, using the evidence date. Historical requests no longer write simulated predictions into today's live outcome log. The benchmark additionally isolates all calibration state from user data.
- Clarify that the terminal's time-machine outcome is a closing-price market move, not realized strategy P&L. Execution simulation lives in the benchmark.

This improves output integrity and some risk decisions. It does not constitute retraining Gemini or prove superior future investment returns. The optional trained historical classifier remains a separate experimental signal; these committee replays run without that optional artifact, as in the unconfigured production fallback.

## Reproduce

Use the normalized datasets from `HISTORICAL_BENCHMARK.md` and the existing backend dependencies. From the repository root:

```powershell
$env:OMP_NUM_THREADS='2'
$env:PYTHONUTF8='1'
python -m backend.committee_benchmark --data-root ../datasets --output benchmarks/committee
python -m pytest backend/tests -q
```

The default baseline is the trusted fork commit `6087be7a01ac416e8a79b3cd544c6f453efd00e5`. The replay loads that revision's graph as Python code, so only use trusted baseline refs. Both versions receive exactly the same saved quant snapshots. Production GARCH, anomaly ensemble, HMM and LSTM computations run on past bars only; future bars stay in the evaluator. The old graph's entire streamed payload is evaluated, including its former probability fields.

Outputs:

- `benchmarks/committee/report.json`: metrics, source hashes, code hashes, seed, dates and limits.
- `benchmarks/committee/report.md`: readable results.
- `frontend/public/committee-benchmark.json`: paired final answers consumed by `/benchmark`.
- `benchmark-runs/committee/cases.jsonl`: full events, quant evidence, future bars and execution results (local, ignored).
- `benchmark-runs/committee/snapshots/`: cached production quant outputs. Move this directory aside before rerunning after any quant/model/dependency changes; the cache is intended to preserve paired evidence, not track code changes automatically.

The benchmark uses a cold-start calibration state with no resolved outcome history. No historical prediction is fed into today's live calibration file. It does not test an online learning loop. Production calibration formulas still require separate validation before their internal values can be interpreted probabilistically.

## How to read the result honestly

This is an 80-case pilot: 20 selected surviving large-cap stocks, four predetermined dates each. The dates include stress and quieter periods, but are not a random or comprehensive sample. Results are correlated across stocks and dates. Descriptive Wilson intervals for trade win rates do not account for that clustering.

The six consistency checks are targeted regression checks, not a universal factual-truth grader. Their pass rate must not be called investment accuracy. The historical tables are independent outcome measurements; returns can get worse even while the text becomes more honest.

The datasets contain prices and volume. They cannot validate archived headlines, causality, institutional flows, company fundamentals, or Gemini reasoning. Historical news is explicitly unavailable in the replay; FinBERT is not secretly given present-day headlines. Actual Gemini output quality remains unevaluated because this run makes no Gemini calls.

Other limits: price corporate-action adjustments are unverified; no dividends, realistic market impact, financing or short-borrow constraints; stops are fixed because the original API supplies a distance but no executable trailing-stop rule. The fully invested long reference is context, not a risk-matched benchmark. Per-case mean returns are not annualized portfolio returns.

## Next evidence needed

A stronger release gate needs an untouched period and a broader point-in-time universe, archived timestamped news with source documents, human-labelled factual and inference claims, and real configured-model runs. Score citation support, omitted adverse evidence, numerical accuracy, decision stability across repeated model calls, latency and cost, as well as financial outcomes. Train or tune only on earlier development cases, then evaluate once on a held-out period. Do not optimize prompts against the same examples used to claim improvement.
