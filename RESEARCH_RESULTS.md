Follow-up: [latest upstream comparison and experimental 2R exits](UPSTREAM_COMPARISON.md). The historical results below preserve the previous release; the ETF policy is unchanged.

# Returns, flash-crash replay and Groq news

Open **`/validation`** for the judge-facing dashboard. It has annual profits, account-value charts, the real 2010 flash-crash replay, nine volume-injection scenarios, daily committee outputs, model comparisons and a current paper allocation button. `/model-lab` preserves the unsuccessful Timestral experiment; `/stress-test` preserves the earlier COVID/2022 evidence.

## What clears 10%, and what does not

The **new, separate USD ETF allocation policy** returned **13.59% annualized after simulated costs from 2020 through 2025**. $10,000 became **$21,437.47**, a **$11,437.47 profit** and **114.37% cumulative return**. Annualized return is the compounded average, not a promise or a win rate.

| Year | Allocation policy | SPY buy-and-hold |
|---|---:|---:|
| 2020 | 17.47% | 17.14% |
| 2021 | 18.87% | 28.14% |
| 2022 | -14.45% | -17.73% |
| 2023 | 7.17% | 24.87% |
| 2024 | 27.78% | 23.58% |
| 2025 | 31.06% | 16.53% |

SPY earned **14.26% annualized**, more than the policy. The policy's worst daily closing drawdown was **19.48%**, versus **33.69%** for SPY. It exceeded 10% in four of six years, lost money in 2022, and missed 10% in 2023. Its advantage in this experiment is a smaller historical decline, not proven excess return.

**How it was selected:** six candidates (126/252-session momentum, 15%/25%/100% per-asset volatility targets) were compared only on 2011–2019 using CAGR minus half the maximum drawdown. The winning 252-session/15% policy was written to `frozen-policy.json` before calculating test results. At each month’s first session it ranks SPY, QQQ, TLT and GLD using only prior completed bars and dividend-inclusive returns; it can hold up to two positive-momentum ETFs, at most 50% of capital each. Volatility scales those weights downward. The rest is cash. There is no leverage, no shorting and no outcome-selected test-period tuning.

This is retrospective research. The universe was selected today, and 2024/2025 market behavior had already been examined in other experiments. Therefore this is a chronological parameter holdout, **not a pristine blind or prospective test**. It is not an India-stock result or the original single-stock committee’s P&L. There is no claim that Groq generated these returns.

Execution uses next-session opens, fractional shares, dividend cash for existing holders, 0.15% on actual turnover per side, and final-close liquidation. Cash earns zero; taxes, FX and market impact are excluded. The dashboard includes 0/15/50-bps cost sensitivity. Independent reconciliation checks every daily balance and all **75 transactions** across strategy and baseline. Frozen prices and exact target weights are supplied in the evidence archive.

## Flash crash: real data, honest scope

We retrieved **14,040 five-second SPY observations** across May 5–7, 2010 from [Stefan Voigt's SPY dataset](https://huggingface.co/datasets/voigtstefan/sp500), pinned revision `07322c1747957fb0e9ab45dc9dcc67d33d80293b`. Attribution: Menkveld, Hautsch and Voigt (2026), *Market Responses to a VIX Shock*. The dataset declares MIT licensing. Raw files, hashes and its card are included.

The [publisher's extraction code](https://github.com/voigtstefan/market-responses-to-a-vix-shock/blob/main/01-lobster-extract-data.R) establishes signed volume in shares, total volume in USD and spreads in basis points. This differs from parts of the card's prose. Session wall-clock timestamps were encoded with a UTC tag; we preserve their 09:30–16:00 New York session interpretation rather than shifting them four hours. We do not pretend these aggregates are consolidated executable tick quotes. The [SEC/CFTC investigation](https://www.sec.gov/files/marketevents-report.pdf) supplies historical context, not inputs known to the model before the crash.

**Scenario:** an explicitly assumed $10,000 long at 13:55; compare holding, a fixed 2% stop and a new intraday risk guard. This is not an entry predicted by the daily model.

| May 6 response | Final return | Worst intraday account decline |
|---|---:|---:|
| Hold | -2.02% | 7.39% |
| Fixed 2% stop | -2.33% | 2.33% |
| New guard | -2.36% | 2.36% |

The guard cut the drawdown but missed the rebound; it did **not** beat the simple stop or holding on final P&L. It halved exposure at 14:06:25 after visible bid depth deteriorated, then requested exit at 14:43:35 after an adverse one-minute move. These observations support risk sensitivity, not crash prediction.

The guard uses only the current/past tape: exit on a 1% adverse one-minute move or 10-bps spread; halve on exit-side depth below 25% of the preceding five-minute median, or at least 3x volume with 60% adverse signed flow. Favorable flow never increases risk. Rules were fixed before running the event replay, but designed with knowledge of flash-crash mechanisms. The test includes surrounding days, 5/30/60-second delays and 0/10/50-bps extra slippage. Sales are capped at 10% of observed best-bid depth per interval; final sampled-quote liquidation is an explicit assumption. Fees are 15 bps each side. No queue position or endogenous market impact is simulated.

The actual quant → bull → bear → CRO stages also ran on six pre-cutoff daily snapshots around the event. Five rejected a trade; May 3 approved a 3.77% long allocation. May 5's decision was **no position**, not a successful crash forecast. Each case includes the final narrative, risk checks and next-five-session outcome. These cases overlap and must not be added as independent portfolio returns. Models trained after those dates are unavailable; missing historical news is disclosed rather than fabricated or replaced with modern headlines.

## Mid-trade bullish/bearish volume injections

At **14:00 for five minutes**, the replay tests 2x, 5x and 10x volume under bullish, bearish and unsigned controls. Directional shocks set signed pressure to ±80% of the modified dollar volume, converted to signed shares. Historical prices, spreads and depth remain unchanged. This is **input sensitivity**, not a claim that real injected orders would leave prices unchanged.

- Bearish 5x volume led to earlier risk reduction and a **-2.23%** final return; bearish 10x produced **-2.26%**.
- Bullish, unsigned and bearish 2x scenarios finished at the unmodified guard's **-2.36%**. Volume multiples are relative to original bars, while the trigger compares with the preceding five-minute baseline; doubling does not guarantee a trigger.
- Inference: sufficiently strong adverse signed flow changes risk handling. Total volume alone cannot identify direction, and bullish volume does not justify overriding risk caps.

All nine scenarios, exact decision times, reasons, account paths and transaction ledgers are saved. **27 replays** passed cash/share reconciliation. Full five-second paths are in `benchmarks/flash-crash/full-replay.json.gz`; the UI downsamples charts to one minute. No intraday return is annualized.

## Groq instead of FinBERT

**Groq Qwen 3.8 27B is now the default live news provider.** The user's key stays in ignored root `.env`, server-side. FinBERT remains an explicit offline option and comparison baseline. Qwen reads supplied public headlines and dates; returns relevance, exact quotes and explanations; and abstains on ambiguous macro evidence. Invalid quotes, missing credentials or provider failures are unavailable, with no silent substitution. It does not output invented profit probabilities.

On the same frozen **120 balanced financial tweets**:

| Model | Sentiment accuracy | Macro F1 |
|---|---:|---:|
| FinBERT | 66.67% | 0.666 |
| FinSense ModernBERT | 61.67% | 0.615 |
| Groq Qwen | **87.50%** | **0.872** |

All 120 planned examples are included. Provider JSON-generation errors occurred; unchanged failed requests were retried with a bounded policy and recorded. Successful responses were cached. The labels were never sent to Qwen. This small balanced diagnostic may overlap foundation-model pretraining and tests general text sentiment, while production additionally filters company relevance. It is not a trading win rate or proof of news truth. The separate full 2,388-row local-model comparison remains available. All twelve synthetic engineering cases passed with production prompt version 3 after a quota-limited run was resumed; that earlier HTTP 429 is retained in the diagnostic record. Production spaces requests to conserve the free-tier quota and retries only JSON-generation failures, at most twice.

## What is live versus experimental

- `/`: live single-stock committee with default Groq news and explicit uncertainty.
- `/validation`: recorded proof plus current **paper** ETF allocation using the frozen policy. It places no orders. `GET /api/portfolio/recommendation` uses completed daily bars.
- `market_guard.assess`: tested intraday guard, also enforced by final committee audit when a `market_guard` snapshot is supplied. **The daily terminal has no streaming order-book feed**, so it does not continuously monitor live positions. The flash replay is its demonstrated application.
- `/model-lab`: Timestral research only. Its four annual portfolios missed the 10% target; it was not promoted.

## Reproduce and inspect

Extract `benchmarks/evidence/AARC-research-evidence.zip` **at the repository root**. It restores exact frozen inputs under `benchmark-runs/` and includes a SHA-256 manifest. Install the normal project requirements and `backend/requirements-challenger.txt`. Use a separate `HF_HOME` cache if desired.

```bash
python -m backend.portfolio_research
python -m backend.portfolio_verify
python -m backend.flash_research
python -m backend.flash_committee
python -m backend.challenger_benchmark report
python -m backend.challenger_verify
python -m backend.research_setup
python -m backend.sentiment_challenger_benchmark
python -m backend.qwen_sentiment_benchmark
python -m pytest backend/tests -q
```

Hosted Qwen uses the user's Groq quota. `GROQ_API_KEY` and `AARC_NEWS_PROVIDER=qwen` belong in root `.env`, never Git. To explicitly opt into local FinBERT, set `AARC_NEWS_PROVIDER=local` and `AARC_LOCAL_SENTIMENT_MODEL=finbert`.

The credible demo claim is: **a measured allocation policy, evidence-linked news interpretation and inspectable risk stress tests**. It is not “AI guarantees 10%” or “we predicted the flash crash.”
