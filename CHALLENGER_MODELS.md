# Timestral, ModernBERT and hosted Qwen

Open `/model-lab` for the annual-return experiment, model comparisons and a live Timestral research forecast. The earlier `/stress-test` evidence remains unchanged.

The follow-up [portfolio and flash-crash results](RESEARCH_RESULTS.md) are shown at `/validation`. Groq Qwen scored 87.5% on all 120 shared examples; provider failures and limitations are recorded.

## What changed in the running product

- The news reader supports **Qwen 3.8 27B through Groq**. It receives public headlines, ticker and publication dates. It returns company relevance, sentiment, exact evidence quotes and a short explanation. Quotes and source IDs are validated; unrelated and macro-only headlines cannot manufacture a company-specific directional signal.
- The output uses strict JSON Schema, then application-level validation. Network errors, rate limits, missing credentials and invalid evidence are explicitly unavailable. There is no silent model substitution or paid-provider fallback. Results are cached by input and prompt version to conserve quota.
- Qwen does not invent a numeric confidence. The UI displays the actual model, evidence quotes and explanations. An uncertain interpretation is distinct from neutral sentiment or missing news.
- FinSense ModernBERT is available as a pinned local alternative. The existing FinBERT baseline remains selectable because the evaluation does not show a clear all-metric replacement winner.
- Timestral runs in a separate research endpoint and policy. Unlike the earlier classifier panel, this policy actually uses the forecast to approve or reject a position. It is **not promoted to the main trading committee**, because the recorded annual returns did not meet the target.

## What the tests show

Read [annual results](benchmarks/challenger/report.md), [exact results and curves](benchmarks/challenger/report.json), [independent reconciliation](benchmarks/challenger/verification.json), [sentiment evaluation](benchmarks/challenger/sentiment-report.json), [Qwen comparison](benchmarks/challenger/qwen-comparison.json) and [Qwen engineering diagnostics](benchmarks/challenger/qwen-diagnostics.json).

The Timestral experiment froze its policy before revealing 2024–2025 outcomes. It evaluates 788 dated forecasts across eight instruments. Initial capital is 10,000 per market/year; weekly next-open rebalancing; long-only, no leverage; actual turnover fees at 0.15% per side; cash dividends; final-close liquidation. Four equal starting sleeves can invest up to their full value under a 20% annual volatility target. This uses much more capital than the earlier committee. The policy requires a positive 200-day trend and a Timestral median five-session forecast above the 0.3% hurdle. An identical trend/volatility policy without the forecast is the component baseline.

Timestral returns: US 2024 **+6.53%**, India 2024 **−1.00%**, US 2025 **+0.48%**, India 2025 **0%**. It cleared 10% in **zero of four** portfolios. The simpler trend/volatility baseline returned +23.76% in US 2024 but was weak in the other groups. We do not select that isolated positive year as proof of a generally successful strategy.

On all 2,388 rows actually present in the financial-tweet validation CSV, FinBERT scored 72.53% accuracy and 0.6682 macro-F1; FinSense scored 73.62% and 0.6664. FinSense's accuracy gain is not a clear improvement across classes. The existing lexicon gate is separately scored. The dataset README lists a different validation count; the report records the actual CSV row count and hash rather than assuming the card count.

The Qwen shared-sample test is separate: 120 fixed, balanced rows evaluated by every model. This prompt measures general financial sentiment. Production Qwen additionally performs company-relevance filtering, so the two tasks must not be conflated. The twelve synthetic diagnostics test integration behavior only and are not a representative accuracy estimate.

**No Qwen P&L is claimed.** A sufficiently complete archived company-news tape was not available for the annual portfolio experiment. Current foundation models may have seen these historical markets or public text datasets during pretraining; this is not a reconstructed vintage deployment or a guaranteed unseen-data benchmark.

## Run and configure

Install the existing runtime and `python -m pip install -r backend/requirements-challenger.txt`. Timestral's 28M checkpoint is pinned; both local sentiment checkpoints are pinned. Models download from their official/publisher Hugging Face repositories and are not copied into Git. `model-sources.json` records exact revisions. The Timestral weights use the [included Modified MIT license](benchmarks/challenger/TIMESTRAL-LICENSE.txt); attribution: **2026 — Mistral AI**.

For hosted Qwen, set these in the root `.env` (Git-ignored):

```dotenv
GROQ_API_KEY=your_own_key
AARC_NEWS_PROVIDER=qwen
```

Create a free-tier Groq account/key at <https://console.groq.com/keys>. Verified model ID: `qwen/qwen3.8-27b`. The published free-tier limits at evaluation time are 30 requests/minute, 1,000 requests/day, 8,000 tokens/minute and 200,000 tokens/day; account limits and availability can change. Check your account's plan. Paid Groq accounts have separate pricing; the application cannot verify billing-plan status. Do not enable paid billing merely to run the demo. Qwen is a preview model, and provider errors can occur. The key stays server-side and is never sent to the browser.

For a fully local reader:

```dotenv
AARC_NEWS_PROVIDER=local
AARC_LOCAL_SENTIMENT_MODEL=finbert
# Or: finsense
```

The live research route is `GET /api/research/forecast?ticker=AAPL`. It downloads current daily prices and returns the five-session quantiles and the experimental decision. It places no orders and explicitly reports `production_enabled=false`.

## Reproduce

```bash
python -m backend.challenger_benchmark download
python -m backend.challenger_benchmark generate
python -m backend.challenger_benchmark report
python -m backend.challenger_verify
python -m backend.research_setup
python -m backend.sentiment_challenger_benchmark
python -m backend.qwen_sentiment_benchmark
python -m pytest backend/tests -q
```

`sentiment_challenger_benchmark` uses the pinned validation CSV and model manifest under `benchmark-runs/challenger`. The evidence archive supplies those inputs. Its source is [Twitter Financial News Sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment), revision `ccbe24de388e287beb92dd393a335c376b350ac3`; file `sent_valid.csv`; labels 0 negative, 1 positive, 2 neutral. No training or prompt selection uses its labels. Hosted Qwen requests consume the user's provider quota and resume cached successful batches. Failures must be resolved or reported as incomplete, never dropped from the denominator.

Price CSVs, cases and provider caches are runtime evidence, not production calibration. Tests use isolated state and disable hosted credentials. Reports contain public data and model results, never API keys.

## Sources checked for this implementation

- [Mistral Timestral inference repository](https://github.com/mistralai/timestral), code commit `a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff`.
- [Timestral-28M model card](https://huggingface.co/mistralai/Timestral-28M).
- [FinSense ModernBERT card](https://huggingface.co/AnkitAI/FinSense-ModernBERT-Financial-News-Sentiment-Analysis): its own revised card does not claim a demonstrated out-of-corpus win over FinBERT.
- [ModernFinBERT alternative](https://huggingface.co/neoyipeng/ModernFinBERT-base): investigated but not loaded because its published config exposes generic labels and a multi-label task inconsistent with the proposed three-class drop-in contract. We do not guess label mappings.
- [Groq Qwen model documentation](https://console.groq.com/docs/model/qwen/qwen3.8-27b), [structured outputs](https://console.groq.com/docs/structured-outputs), and [rate limits](https://console.groq.com/docs/rate-limits).

General time-series benchmark rankings, news sentiment accuracy, and profitable trading are different claims. This release keeps them separate.
