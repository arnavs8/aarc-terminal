"""Frozen crisis replay using production model stages and a separately held future tape."""
import argparse
import asyncio
from copy import deepcopy
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import types
import numpy as np
import pandas as pd

ROOT=Path('benchmark-runs/stress')
PUBLIC=Path('benchmarks/stress')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def config(): return json.loads((PUBLIC/'config.json').read_text())

def schedule(frame, period):
    """Sample regular trading-session intervals, requiring five future sessions in-period."""
    dates=frame.loc[period['start']:period['end']].index
    return list(dates[:-5:period['stride_sessions']])


def plan():
    cfg=config();tasks=[]
    for episode,period in cfg['periods'].items():
        for market,symbols in cfg['symbols'].items():
            for ticker in symbols:
                path=ROOT/'data'/market/(ticker+'.csv');frame=pd.read_csv(path,index_col='Date',parse_dates=True)
                for date in schedule(frame,period):
                    tasks.append({'id':f'{episode}-{ticker}-{date.date()}','episode':episode,'market':market,'ticker':ticker,'date':str(date.date()),'data_sha256':sha(path),'model_sha256':sha(ROOT/'models'/f'{episode}-{market}.joblib')})
    result={'config_sha256':sha(PUBLIC/'config.json'),'tasks':tasks}
    p=ROOT/'plan.json'
    if p.exists() and json.loads(p.read_text())!=result: raise ValueError('Frozen plan changed: create a new run directory')
    p.write_text(json.dumps(result,indent=2),encoding='utf8')
    return tasks


def prepare_news():
    from backend.tools.sentiment_nlp import FinBertSentimentAnalyzer
    import torch
    torch.set_num_threads(2)
    tasks=plan();sources=json.loads((PUBLIC/'news-sources.json').read_text());analyzer=FinBertSentimentAnalyzer.get_instance()
    groups={};result={}
    for date in sorted({t['date'] for t in tasks}):
        cutoff=pd.Timestamp(date)
        eligible=[s for s in sources if pd.Timestamp(s['available_from'])<=cutoff and pd.Timestamp(s['date'])>=cutoff-pd.Timedelta(days=30)]
        key=tuple(s['id'] for s in eligible)
        if key not in groups:
            if not eligible:
                value={'aggregate_sentiment_score':None,'headlines':[],'news_available':False,'news_error':'No curated macro announcement in the trailing 30 days.'}
            else:
                items=[{'title':s['title'],'published_at':s['date'],'source_url':s['url']} for s in eligible]
                value={**analyzer.score_headlines(items),'news_available':True}
            value['source_ids']=list(key);value['coverage']='Curated primary-source macro summaries; not company-specific archived headlines.'
            groups[key]=value
        result[date]=groups[key]
    output={'sources_sha256':sha(PUBLIC/'news-sources.json'),'finbert_model':'ProsusAI/finbert','by_date':result}
    (ROOT/'news.json').write_text(json.dumps(output,indent=2),encoding='utf8')
    print('Prepared',len(tasks),'decisions;',len(groups),'distinct news contexts; no future news supplied',flush=True)


def trade_path(verdict, future, cost_bps=30, borrow_rate=.03):
    """Executable fixed-stop accounting, including gap fills, dividends and assumed borrow.

    Returns on the invested amount. Allocation is applied later by the portfolio ledger.
    """
    allocation=float(verdict.get('recommended_allocation_pct',0))/100
    action=verdict.get('action')
    if allocation<=0 or action not in ('LONG','SHORT'):
        return {'traded':False,'action':action,'allocation':0.,'net_return':0.,'weighted_return':0.,'stop_hit':False,'path':[{'date':str(d.date()),'return':0.} for d in future.index]}
    if not 0<allocation<=1: raise ValueError('Invalid final allocation')
    direction=1 if action=='LONG' else -1
    entry=float(future.iloc[0].Open);entry_date=future.index[0];stop_pct=float(verdict['stop_loss_pct'])
    if not 0<stop_pct<100: raise ValueError('Invalid stop')
    stop=entry*(1-direction*stop_pct/100)
    target_r=verdict.get('execution_plan',{}).get('take_profit_r_multiple')
    if target_r is not None and (not np.isfinite(target_r) or target_r<=0):raise ValueError('Invalid reward-to-risk target')
    target=entry*(1+direction*stop_pct/100*target_r) if target_r is not None else None
    target_hit=False
    path=[];dividends=0.;closed=False;stop_hit=False;exit_price=None;exit_date=None;net=0.;gross=0.;borrow=0.
    for date,bar in future.iterrows():
        if not closed:
            # Buying at the ex-dividend open does not earn that day's dividend.
            if date > entry_date: dividends+=float(bar.get('Dividends',0))
            hit=(bar.Low<=stop) if direction==1 else (bar.High>=stop)
            profit_hit=target is not None and ((bar.High>=target) if direction==1 else (bar.Low<=target))
            exit_now=hit or profit_hit or date==future.index[-1]
            price=(min(stop,float(bar.Open)) if direction==1 else max(stop,float(bar.Open))) if hit else float(bar.Close)
            if profit_hit and not hit:
                price=max(target,float(bar.Open)) if direction==1 else min(target,float(bar.Open));target_hit=True
            borrow=(max(1,(date-entry_date).days+1)/365*borrow_rate) if direction==-1 else 0.
            gross=direction*(price/entry-1)+direction*dividends/entry
            net=gross-borrow-cost_bps/20000-(cost_bps/20000 if exit_now else 0)
            if exit_now: closed=True;stop_hit=bool(hit);exit_price=price;exit_date=date
        path.append({'date':str(date.date()),'return':float(net)})
    return {'traded':True,'action':action,'allocation':allocation,'entry_date':str(entry_date.date()),'entry_price':entry,
            'exit_date':str(exit_date.date()),'exit_price':exit_price,'stop_price':stop,'stop_hit':stop_hit,
            'target_price':target,'target_hit':target_hit,
            'gross_return':gross,'dividends_per_share':dividends,'borrow_cost_fraction':borrow,'round_trip_cost_fraction':cost_bps/10000,
            'net_return':float(net),'weighted_return':float(allocation*net),'path':path}


def replay_nodes(module,q,news):
    module.analyze_ticker_quant=lambda *args:deepcopy(q)
    module.analyze_ticker_sentiment=lambda *args:deepcopy(news)
    async def collect(): return {e['node']:e['data'] async for e in module.stream_committee(q['ticker'],q['as_of_date'])}
    events=asyncio.run(collect())
    if 'cro' not in events: raise RuntimeError(events)
    return events


def run(shard_index=0,shard_count=1):
    import torch
    from backend import graph
    from backend.tools.quant_math import analyze_ticker_quant
    from backend.calibration.bayesian_calibrator import BetaBinomialConfidenceCalibrator as Cal
    from backend.calibration.rolling_tracker import RollingAccuracyTracker
    from backend.committee_benchmark import checks
    torch.set_num_threads(2);tasks=plan();cfg=config();news=json.loads((ROOT/'news.json').read_text())
    assert news['sources_sha256']==sha(PUBLIC/'news-sources.json')
    original=types.ModuleType('stress_original');original.__file__=graph.__file__
    exec(subprocess.check_output(['git','show',cfg['original_graph_commit']+':backend/graph.py']).decode('utf8'),original.__dict__)
    code_files=sorted(set(Path('backend').rglob('*.py'))-set(Path('backend/tests').rglob('*.py')))
    production_files=[p for p in code_files if not p.name.startswith('stress_')]
    code_hashes={p.as_posix():hashlib.sha256(p.read_text(encoding='utf8').encode()).hexdigest() for p in production_files}
    signature=hashlib.sha256(json.dumps(code_hashes,sort_keys=True).encode()).hexdigest()
    (ROOT/'code-manifest.json').write_text(json.dumps(code_hashes,indent=2),encoding='utf8')
    snapshots=ROOT/'snapshots';snapshots.mkdir(exist_ok=True)
    cases=ROOT/'cases';cases.mkdir(exist_ok=True)
    failures=[]
    with tempfile.TemporaryDirectory() as temp:
        class QuietTracker(RollingAccuracyTracker):
            def __init__(self):super().__init__(str(Path(temp)/'rolling.json'))
            def log_prediction(self,**kwargs):pass
        for module in (graph,original):
            module._get_gemini_client=lambda:None
            module.RollingAccuracyTracker=QuietTracker
            module.BetaBinomialConfidenceCalibrator=lambda:Cal(storage_path=str(Path(temp)/'bayes.json'))
        for i,t in enumerate(tasks,1):
            if (i-1) % shard_count != shard_index: continue
            started=time.perf_counter();out=cases/(t['id']+'.json')
            quant_fingerprint=hashlib.sha256(json.dumps([t,signature],sort_keys=True).encode()).hexdigest()
            fingerprint=hashlib.sha256(json.dumps([t,signature,news['sources_sha256'],sha(ROOT/'news.json'),sha(__file__),sha(PUBLIC/'config.json')],sort_keys=True).encode()).hexdigest()
            if out.exists() and json.loads(out.read_text()).get('fingerprint')==fingerprint:
                print(f'{i}/{len(tasks)} cached {t["id"]}',flush=True);continue
            try:
                frame=pd.read_csv(ROOT/'data'/t['market']/(t['ticker']+'.csv'),index_col='Date',parse_dates=True)
                assert sha(ROOT/'data'/t['market']/(t['ticker']+'.csv'))==t['data_sha256']
                date=pd.Timestamp(t['date']);past=frame.loc[:date].copy();future=frame.loc[frame.index>date].head(5).copy()
                assert len(future)==5 and future.index[-1]<=pd.Timestamp(cfg['periods'][t['episode']]['end'])
                model=ROOT/'models'/f'{t["episode"]}-{t["market"]}.joblib'
                os.environ[f'AARC_MODEL_{t["market"].upper()}']=str(model.resolve())
                snap=snapshots/(quant_fingerprint+'.json')
                with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                    if snap.exists():q=json.loads(snap.read_text())
                    else:
                        np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
                        q=analyze_ticker_quant(t['ticker'],t['date'],price_data=past)
                        snap.write_text(json.dumps(q),encoding='utf8')
                    assert q['time_machine_actual_return_pct'] is None
                    assert q['as_of_date']==t['date']
                    assert q['learned_forecast']['available'], q['learned_forecast']
                    assert q['learned_forecast']['training_data_end']<t['date']
                    actual=float(future.iloc[-1].Close/future.iloc[0].Open-1)
                    threshold=float(np.clip(.5*past.Close.pct_change().tail(20).std()*np.sqrt(5),.01,.15))
                    target='DOWN' if actual < -threshold else ('UP' if actual>threshold else 'FLAT')
                    row={**t,'fingerprint':fingerprint,'quant':q,'news':news['by_date'][t['date']],
                        'future_bars':json.loads(future.reset_index().to_json(orient='records',date_format='iso')),
                        'classification_target':target,'classification_threshold':threshold,'future_price_return':actual}
                    for label,module in [('original',original),('current',graph)]:
                        events=replay_nodes(module,q,row['news']);v=events['cro']
                        audit=checks(q,events)
                        audit['no_future_prices']=q['time_machine_actual_return_pct'] is None
                        audit['classifier_trained_before_signal']=q['learned_forecast']['training_data_end']<t['date']
                        audit['news_available_before_signal']=all(s['published_at']<t['date'] for s in row['news']['headlines'])
                        row[label]={'events':events,'checks':audit,'trade':trade_path(v,future,cfg['round_trip_bps']),
                            'cost_sensitivity':{str(c):trade_path(v,future,c)['weighted_return'] for c in cfg['sensitivity_costs_bps']}}
                    current=row['current']['events']['cro']
                    row['matched_long']=trade_path({**current,'action':'LONG'},future,cfg['round_trip_bps'])
                    row['no_short']=trade_path({**current,'recommended_allocation_pct':0} if current['action']=='SHORT' else current,future,cfg['round_trip_bps'])
                row['elapsed_seconds']=round(time.perf_counter()-started,3)
                pending=out.with_suffix('.pending')
                pending.write_text(json.dumps(row),encoding='utf8')
                pending.replace(out)
                print(f'{i}/{len(tasks)} {t["id"]} {current["action"]} {row["elapsed_seconds"]}s',flush=True)
            except Exception as e:
                failures.append({'id':t['id'],'error':repr(e)})
                print('FAILED',t['id'],repr(e),flush=True)
    (ROOT/f'failures-{shard_index}.json').write_text(json.dumps(failures,indent=2),encoding='utf8')
    if failures: raise RuntimeError(f'{len(failures)} cases failed; these must not be silently excluded')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['news','run']);p.add_argument('--shard-index',type=int,default=0);p.add_argument('--shard-count',type=int,default=1);args=p.parse_args()
    if not 0<=args.shard_index<args.shard_count:p.error('Invalid shard index/count')
    if args.command=='news':prepare_news()
    else:run(args.shard_index,args.shard_count)
