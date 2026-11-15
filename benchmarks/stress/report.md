# AARC through COVID and the 2022 rate hikes

356 scheduled decisions, eight instruments, two episodes. Every decision is preserved; none is silently discarded.

## Read this first

This asks what the present committee would have said using only information available at each historical date. It does not claim the committee predicted COVID or interest-rate policy. Gemini is not active in this run.

Start with 10,000 units of money in each market separately: dollars for the US, rupees for India. Divide it equally across four instruments. Each recommendation invests only its approved fraction of that instrument's share; the remainder stays in cash. Profits and losses carry into later decisions.

| Episode / market | Current committee | Original committee | Buy and hold | Always buy with same sizing | No-short committee |
|---|---:|---:|---:|---:|---:|
| COVID crash and early recovery / us | 10,009.54 | 9,976.95 | 8,887.46 | 10,016.78 | 10,014.66 |
| COVID crash and early recovery / india | 10,022.52 | 9,990.33 | 9,994.55 | 9,999.48 | 10,025.82 |
| 2022 Federal Reserve rate-hike year / us | 10,001.46 | 9,957.40 | 10,328.55 | 9,997.00 | 10,002.57 |
| 2022 Federal Reserve rate-hike year / india | 10,054.11 | 10,064.49 | 11,259.32 | 10,004.39 | 10,037.19 |

Cash remains 10,000. Buy-and-hold takes much more market exposure: beating it during a crash can result from holding cash rather than forecasting well. The same-sizing long comparison helps test whether the chosen direction adds value.

## Wins, losses and risk

| Episode / market | Recommendations | Executed trades | Profitable after costs | Worst closing-balance decline | Mean allocation per recommendation |
|---|---:|---:|---:|---:|---:|
| covid / us | 80 | 49 | 22/49 | 0.39% | 2.27% |
| covid / india | 76 | 50 | 26/50 | 0.53% | 2.59% |
| hikes / us | 100 | 73 | 36/73 | 0.50% | 4.16% |
| hikes / india | 100 | 77 | 38/77 | 0.50% | 4.70% |

A no-trade recommendation earns zero; it is not counted as a successful trade. A short is a bet that the price falls, with assumed borrowing. Stops limit losses imperfectly because prices can jump past them.

## How the model was tested

1. Download and hash daily prices, dividends and split records from Yahoo Finance. Check dates, missing values and OHLC bounds.
2. Train separate price classifiers through 2019-12-31 for COVID and 2021-12-31 for 2022. Select models only within those earlier samples.
3. At each scheduled close, compute the actual production GARCH, HMM, Isolation Forest and LSTM outputs from at most 252 past bars.
4. Feed only already-published curated macro summaries to the actual FinBERT model. If none exist within 30 days, disclose unavailable news.
5. Run the original and revised full streamed committees on identical evidence and clean isolated calibration state. Record every argument and final decision.
6. Only after that decision, reveal the next five sessions to the execution evaluator. Enter at the next open, apply the fixed stop with gap handling, dividends, 30-basis-point round-trip costs, and 3% annual short borrowing.
7. Track daily closing balances, retain cash between decisions, and compare with the predefined alternatives. Repeat accounting at 0 and 100 basis points of trading costs.

The separate price predictor is shown and scored too, but the production CRO does not use its probabilities to choose the trade. It would be misleading to credit that predictor for committee returns.

## Inspect the proof

- `/stress-test` in the website: every dated case, original/current answers, sources, five future bars and trade arithmetic.
- `trade-ledger.csv`: all original and current recommendations, including zero-exposure cases.
- `report.json`: exact portfolio curves, costs, classification scores, evidence checks, source manifests and hashes.
- Downloadable local evidence ZIP: original CSV inputs, trained artifacts, every complete case, frozen plan, news inputs and code hashes.

## Limits

- A retrospective stress test of the present rules, not proof the system existed or predicted the crisis at the time.
- Selected surviving instruments and two known crisis periods; no claim of market-wide or future predictive accuracy.
- Gemini is not configured. Its reasoning is not tested. Bull/bear narratives use production evidence templates.
- News is a small curated set of official macro-event summaries. Missing company news, India-specific policy news and wording choices limit conclusions.
- Historical Yahoo prices are split-adjusted and may be revised. They are not an immutable point-in-time exchange feed.
- Hypothetical short trades assume borrow and instrument availability, with 3% annual borrowing cost. India overnight cash shorts may require a separate lending/derivative arrangement; the no-short variant avoids that assumption.
- Fractional mathematical positions are assumed; whole-share or lot-size constraints and small-account executability are not tested.
- Fixed stops can fill at worse opening prices. Daily bars do not model intraday liquidity, spread spikes or actual order fills.
- Closing-balance drawdown misses worse intraday losses. Cash earns zero; margin funding, taxes and FX are omitted.
- Calibration stays at the clean starting prior; this is not a test of online outcome learning.
- A score is a rule-based rating, not a measured probability. Trade wins, price classification and output consistency are separate measurements.
- FinBERT weights are the current downloaded model; this is a retrospective component test, not a vintage-2020 deployment reconstruction.

## Evidence consistency, separate from profit

| Check | Original passed | Current passed |
|---|---:|---:|
| allocation within risk cap | 254/356 | 356/356 |
| summary matches allocation | 186/356 | 356/356 |
| zero exposure is no trade | 298/356 | 356/356 |
| no unsupported institutional flow claim | 0/356 | 356/356 |
| no unfitted probability claim | 0/356 | 356/356 |
| score disclosed as heuristic | 0/356 | 356/356 |
| no future prices | 356/356 | 356/356 |
| classifier trained before signal | 356/356 | 356/356 |
| news available before signal | 356/356 | 356/356 |

These targeted checks detect known failure modes. They do not establish universal truth, profit probabilities, or the quality of untested Gemini responses.

## Exploratory macro-news diagnostic

The no-macro-news comparison was added after inspecting sentiment labels. It is a diagnostic, not an independently validated improvement or a production change.

| Episode / market | Current committee | Without macro summaries |
|---|---:|---:|
| covid / us | 10,009.54 | 10,028.61 |
| covid / india | 10,022.52 | 10,000.61 |
| hikes / us | 10,001.46 | 10,034.54 |
| hikes / india | 10,054.11 | 10,055.47 |

## Primary sources

- 2020-03-03: [Federal Reserve](https://www.federalreserve.gov/newsevents/pressreleases/monetary20200303a.htm) — Federal Reserve cuts rates by half a percentage point as coronavirus threatens economic activity.
- 2020-03-11: [WHO](https://www.who.int/news-room/speeches/item/who-director-general-s-opening-remarks-at-the-media-briefing-on-covid-19---11-march-2020) — WHO characterizes COVID-19 as a pandemic amid alarming spread and severity.
- 2020-03-15: [Federal Reserve](https://www.federalreserve.gov/newsevents/pressreleases/monetary20200315a.htm) — Federal Reserve cuts its target rate to zero to 0.25 percent as coronavirus disrupts economic activity.
- 2022-03-16: [Federal Reserve](https://www.federalreserve.gov/newsevents/pressreleases/monetary20220316a.htm) — Federal Reserve raises rates to 0.25 to 0.50 percent as inflation remains elevated and anticipates further increases.
- 2022-05-04: [Federal Reserve](https://www.federalreserve.gov/newsevents/pressreleases/monetary20220504a.htm) — Federal Reserve raises rates to 0.75 to 1 percent and announces balance-sheet reduction while inflation remains elevated.
- 2022-06-15: [Federal Reserve](https://www.federalreserve.gov/newsevents/pressreleases/monetary20220615a.htm) — Federal Reserve raises rates to 1.50 to 1.75 percent as inflation and pressure on economic activity persist.
- 2022-07-27: [Federal Reserve](https://www.federalreserve.gov/newsevents/pressreleases/monetary20220727a.htm) — Federal Reserve raises rates to 2.25 to 2.50 percent while spending and production soften and inflation remains elevated.
- 2022-09-21: [Federal Reserve](https://www.federalreserve.gov/newsevents/pressreleases/monetary20220921a.htm) — Federal Reserve raises rates to 3 to 3.25 percent and anticipates further increases to address elevated inflation.
- 2022-11-02: [Federal Reserve](https://www.federalreserve.gov/newsevents/pressreleases/monetary20221102a.htm) — Federal Reserve raises rates to 3.75 to 4 percent and anticipates further tightening to bring inflation down.
- 2022-12-14: [Federal Reserve](https://www.federalreserve.gov/newsevents/pressreleases/monetary20221214a.htm) — Federal Reserve raises rates to 4.25 to 4.50 percent as inflation remains elevated and ongoing increases are anticipated.