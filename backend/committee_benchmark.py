"""Replay the final streamed committee output on frozen, local market snapshots.

python -m backend.committee_benchmark --data-root ../datasets --output benchmarks/committee
The Git baseline is loaded as code: only use trusted repository revisions.
"""
import argparse
import asyncio
import contextlib
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tempfile
import types
import numpy as np
import pandas as pd
import torch
from backend import graph
from backend.tools.quant_math import analyze_ticker_quant
from backend.calibration.rolling_tracker import RollingAccuracyTracker
from backend.calibration.bayesian_calibrator import BetaBinomialConfidenceCalibrator

BASELINE = '6087be7a01ac416e8a79b3cd544c6f453efd00e5'
DATES = {'india': ['2020-02-03', '2020-03-02', '2020-06-01', '2020-10-01'],
         'us': ['2015-08-03', '2016-01-04', '2016-06-01', '2017-10-02']}
NEWS = {'headlines': [], 'aggregate_sentiment_score': None, 'news_available': False,
        'news_error': 'Price-only dataset: dated news evidence unavailable.'}


def checks(q, events):
    v = events['cro']; allocation = v['recommended_allocation_pct']
    cap = q['half_kelly_allocation_pct'] * q.get('hmm_regime', {}).get('kelly_scale_factor', 1)
    text = v['verdict_summary']
    quoted = re.findall(r'(?:approved at|allocated at|permitted at|final allocation) ([\d.]+)%', text)
    arguments = events['bull']['bull_thesis'] + events['bear']['bear_thesis']
    return {
        'allocation_within_risk_cap': 0 <= allocation <= cap + .005,
        'summary_matches_allocation': all(abs(float(x)-allocation) <= .005 for x in quoted),
        'zero_exposure_is_no_trade': allocation > 0 or v['action'] == 'REJECT_TRADE',
        'no_unsupported_institutional_flow_claim': not bool(re.search(r'reflecting institutional accumulation|flag institutional distribution', arguments)),
        'no_unfitted_probability_claim': not bool(v.get('calibrated_probabilities')),
        'score_disclosed_as_heuristic': v.get('confidence_kind') == 'heuristic_score_not_probability',
    }


def execution(verdict, future, cost_bps=30):
    """Next-open entry, fixed stop with adverse open gaps, fifth-close exit.

    Stops are fixed distances because the API supplies no executable trailing rule.
    Shorts assume borrow availability; fees/borrow not included beyond flat costs.
    """
    allocation = verdict['recommended_allocation_pct'] / 100
    if allocation <= 0 or verdict['action'] == 'REJECT_TRADE':
        return {'traded': False, 'net_trade_return': 0., 'weighted_return': 0., 'stop_hit': False}
    direction = 1 if verdict['action'] == 'LONG' else -1
    entry = float(future.iloc[0]['Open'])
    stop = entry * (1 - direction * verdict['stop_loss_pct']/100)
    exit_price = float(future.iloc[-1]['Close']); stopped = False
    for _, bar in future.iterrows():
        if direction == 1 and bar.Low <= stop:
            exit_price = min(stop, float(bar.Open)); stopped = True; break
        if direction == -1 and bar.High >= stop:
            exit_price = max(stop, float(bar.Open)); stopped = True; break
    net = direction * (exit_price / entry - 1) - cost_bps / 10000
    return {'traded': True, 'net_trade_return': net, 'weighted_return': allocation * net,
            'stop_hit': stopped}


async def replay(module, q):
    q = deepcopy(q)  # Stream metadata must not mutate the paired input or the other version.
    module.analyze_ticker_quant = lambda *args: q
    module.analyze_ticker_sentiment = lambda *args: dict(NEWS)
    return {event['node']: event['data'] async for event in module.stream_committee(q['ticker'], q['as_of_date'])}


def summarize(rows, variant):
    entries = [r[variant] for r in rows]
    trades = [e['execution'] for e in entries if e['execution']['traded']]
    n = len(trades); hits = sum(t['net_trade_return'] > 0 for t in trades)
    if n:
        p = hits/n; z = 1.96; d = 1+z*z/n
        center = (p+z*z/(2*n))/d
        width = z*((p*(1-p)/n+z*z/(4*n*n))**.5)/d
        interval = [center-width, center+width]
    else:
        interval = None
    return {'cases': len(entries), 'trades': n, 'coverage': n/len(entries),
            'net_win_rate': hits/n if n else None, 'win_rate_95pct_wilson': interval,
            'mean_allocated_five_session_return_pct': 100*float(np.mean([e['execution']['weighted_return'] for e in entries])),
            'mean_net_trade_return_pct': 100*float(np.mean([t['net_trade_return'] for t in trades])) if n else None,
            'stop_hits': sum(t['stop_hit'] for t in trades),
            'checks_passed': {k: sum(e['checks'][k] for e in entries) for k in entries[0]['checks']}}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('benchmarks/committee'))
    parser.add_argument('--baseline-ref', default=BASELINE)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    cache = Path('benchmark-runs/committee/snapshots'); cache.mkdir(parents=True, exist_ok=True)
    source = subprocess.check_output(['git', 'show', f'{args.baseline_ref}:backend/graph.py']).decode('utf8')
    baseline = types.ModuleType('committee_baseline'); baseline.__file__ = graph.__file__
    exec(compile(source, 'trusted_baseline_graph.py', 'exec'), baseline.__dict__)
    torch.set_num_threads(2)
    rows = []; hashes = {}
    # Isolate all replay calibration from user logs, and keep a constant cold-start state.
    with tempfile.TemporaryDirectory() as tmp:
        class ReplayTracker(RollingAccuracyTracker):
            def __init__(self): super().__init__(str(Path(tmp)/'rolling.json'))
            def log_prediction(self, **kwargs): pass
        for module in (baseline, graph):
            module._get_gemini_client = lambda: None
            module.RollingAccuracyTracker = ReplayTracker
            module.BetaBinomialConfidenceCalibrator = lambda: BetaBinomialConfidenceCalibrator(storage_path=str(Path(tmp)/'bayes.json'))
        for market, dates in DATES.items():
            for path in sorted((args.data_root/market).glob('*.csv')):
                hashes[f'{market}/{path.name}'] = hashlib.sha256(path.read_bytes()).hexdigest()
                frame = pd.read_csv(path, index_col='Date', parse_dates=True)
                for date in dates:
                    future = frame.loc[frame.index > pd.Timestamp(date)].head(5)
                    if len(future) != 5: continue
                    symbol = path.stem + ('.NS' if market == 'india' else '')
                    key = hashlib.sha256((hashes[f'{market}/{path.name}']+date+'quant-v1-seed42').encode()).hexdigest()[:20]
                    snapshot = cache/f'{key}.json'
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        if snapshot.exists(): q = json.loads(snapshot.read_text())
                        else:
                            np.random.seed(42); torch.manual_seed(42)
                            # Only past bars enter the production quant pipeline. Future is held by evaluator.
                            past = frame.loc[frame.index <= pd.Timestamp(date)]
                            q = analyze_ticker_quant(symbol, date, price_data=past)
                            snapshot.write_text(json.dumps(q))
                        row = {'market': market, 'ticker': symbol, 'date': date, 'quant': q,
                               'future_bars': json.loads(future.reset_index().to_json(orient='records', date_format='iso')),
                               'buy_and_hold_return_pct': float((future.iloc[-1].Close/future.iloc[0].Open-1-.003)*100)}
                        for name, module in [('before', baseline), ('after', graph)]:
                            events = asyncio.run(replay(module, q))
                            if 'cro' not in events: raise RuntimeError(events)
                            row[name] = {'events': events, 'checks': checks(q, events), 'execution': execution(events['cro'], future)}
                    rows.append(row)
                    print(f'{len(rows)}: {market} {symbol} {date}', flush=True)
    report = {'baseline_commit': args.baseline_ref, 'seed': 42, 'round_trip_cost_bps': 30,
              'scope': 'Production quant + full streamed rules-fallback committee + final output. Gemini and archived-news truth untested.',
              'selection': DATES, 'source_sha256': hashes,
              'code_hash_line_endings': 'LF',
              'code_sha256': {p.as_posix(): hashlib.sha256(p.read_text(encoding='utf8').encode('utf8')).hexdigest() for p in [Path('backend/graph.py'), Path('backend/committee_audit.py'), Path('backend/tools/quant_math.py')]},
              'method': 'Same causal snapshots; fresh calibration state; next-open entry; five-session fixed-stop exit; no outcome tuning.',
              'limitations': ['Selected surviving large caps, four dates per symbol: pilot, not a representative market estimate.',
                              'No archived news, Gemini calls, measured market impact, borrow constraints, dividends or verified corporate-action adjustments.',
                              'Consistency checks test specific known failure classes; they do not measure all factual truth.',
                              'Mean allocated returns are per-decision outcomes, not a compounded portfolio backtest.',
                              'Snapshots are paired/correlated. Win-rate intervals treat trades independently and are descriptive only.'],
              'results': {market: {v: summarize([r for r in rows if r['market']==market],v) for v in ('before','after')} for market in DATES}}
    (cache.parent/'cases.jsonl').write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf8')
    public = Path('frontend/public/committee-benchmark.json')
    public.parent.mkdir(parents=True, exist_ok=True)
    thin_cases = []
    for row in rows:
        thin = {k: row[k] for k in ('market', 'ticker', 'date', 'buy_and_hold_return_pct')}
        for variant in ('before', 'after'):
            entry = row[variant]
            thin[variant] = {'checks': entry['checks'], 'execution': entry['execution'],
                'events': {k: entry['events'][k] for k in ('bull', 'bear')}}
            thin[variant]['events']['cro'] = {k: entry['events']['cro'][k] for k in
                ('action', 'recommended_allocation_pct', 'verdict_summary', 'confidence')}
        thin_cases.append(thin)
    public.write_text(json.dumps({'report': report, 'cases': thin_cases}, separators=(',', ':')), encoding='utf8')
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    lines = ['# Whole-committee replay benchmark', '', report['scope'], '', report['method'], '',
             '| Market | Version | Cases | Trades | Net win rate | Mean allocated 5-session return |',
             '|---|---|---:|---:|---:|---:|']
    for market, variants in report['results'].items():
        for version, r in variants.items():
            win = f"{r['net_win_rate']:.1%}" if r['net_win_rate'] is not None else 'unavailable'
            lines.append(f"| {market} | {version} | {r['cases']} | {r['trades']} | {win} | {r['mean_allocated_five_session_return_pct']:.3f}% |")
    lines += ['', '## Final-output consistency (passes / cases)', '']
    for market, variants in report['results'].items():
        for version, r in variants.items():
            lines.append(f"- {market} {version}: " + '; '.join(f"{k}: {n}/{r['cases']}" for k,n in r['checks_passed'].items()))
    lines += ['', '## What these numbers mean', '',
              'A win is a positive return after the assumed execution costs on an allocated trade. No-trade decisions are excluded from win rate, but remain in coverage and mean allocated return. A check passing means a specific contradiction or unsupported claim was absent; it is not proof that the recommendation will make money.', '',
              '## Coverage limits', ''] + ['- '+x for x in report['limitations']]
    (args.output/'report.md').write_text('\n'.join(lines),encoding='utf8')

if __name__ == '__main__': main()
