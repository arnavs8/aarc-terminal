# What this benchmark actually tells us

All 356 planned decisions completed: 156 COVID cases and 200 rate-hike-year cases, across four US and four Indian instruments. This evaluates the actual production quant pipeline, FinBERT, rule-based committee and final reviewer. Gemini was not configured or tested.

## The answer is more consistent, but profitable prediction remains unproved

Across all cases, the revised committee stayed within its risk cap and matched its written allocation to its actual allocation 356/356 times. The original committee passed those checks 254/356 and 186/356 times respectively. The revised outputs also avoided the particular unsupported institutional-flow and unfitted-probability claims checked here. These are checks for known errors, not a measure of universal truth.

Of 249 approved revised-committee trades, 122 were profitable after the stated costs (49.0%). The other 107 decisions stayed in cash and are not counted as wins. A heuristic rating of 80 does not mean an 80% chance of profit.

## Starting with 10,000 in each separate portfolio

| Period and market | Revised committee | Original committee | Buy and hold | Always buy, same sizing |
|---|---:|---:|---:|---:|
| COVID, US dollars | 10,009.54 | 9,976.95 | 8,887.46 | 10,016.78 |
| COVID, Indian rupees | 10,022.52 | 9,990.33 | 9,994.55 | 9,999.48 |
| 2022, US dollars | 10,001.46 | 9,957.40 | 10,328.55 | 9,997.00 |
| 2022, Indian rupees | 10,054.11 | 10,064.49 | 11,259.32 | 10,004.39 |

The revised committee beat the original in three of four portfolios. It did not beat the original in India during 2022. Buy-and-hold beat it in both 2022 baskets. The US basket contains SPY, AAPL, JPM and XOM; it is not the S&P 500 alone.

Mean allocation per recommendation was only 2.27%–4.70% of each instrument's sleeve. Most money stayed in cash. That explains much of the shallow drawdown and also why the committee missed gains from holding stocks. In US COVID, simply buying with the same small sizing did slightly better than the committee. Lower exposure should not be sold as proven forecasting skill.

## The trained price models are research artifacts, not promoted winners

The four separately trained classifiers matched or trailed the pre-period frequency baseline on classification accuracy in every group: 32.5% versus 40.0% (US COVID), 44.7% versus 44.7% (India COVID), 34.0% versus 35.0% (US 2022), and 42.0% versus 42.0% (India 2022). The full report also includes log loss and Brier scores, which assess probability quality.

These models were trained before their respective test periods, and are preserved with their metadata. They were not automatically promoted to the live model paths. Their probabilities currently appear in a separate panel and do not control the CRO's final trade decision.

## What to improve next

1. Evaluate economic event meaning separately from headline sentiment. FinBERT marked several rate-hike summaries positive. Removing the curated macro summaries changed 65 actions, but improved returns in only three of four portfolios. This was an exploratory diagnostic, not a validated new strategy.
2. Define how a calibrated price forecast should influence the committee, then test that integration on a new untouched period. These crisis outcomes must not become both the tuning data and the proof of improvement.
3. Obtain a point-in-time company-news archive and evaluate Gemini separately if that is the intended demo mode. The current test cannot support claims about unavailable reasoning or unobserved news.
4. Test realistic position sizes, short availability, fees and funding for the intended market. The small rupee balances and overnight short trades here are mathematical comparisons, not verified executable orders.

## Proof and verification

- `report.json`: exact results, daily balances, costs, risk-threshold coverage, classifier scores, data manifests and source links.
- `trade-ledger.csv`: every original/current recommendation, including no-trades.
- `verification.json`: 356 source-tape checks and 533 original/current active trades independently reconciled, plus compounded portfolio endpoints.
- `/stress-test`: inspect the complete chain from historical evidence to final answer and next five price bars.
- The accompanying local evidence ZIP contains full case files, normalized prices, model artifacts and source hashes.

76 backend tests passed, and the production frontend build passed. This benchmark measures specific past behavior on selected surviving instruments, not future performance or universal factual accuracy.
