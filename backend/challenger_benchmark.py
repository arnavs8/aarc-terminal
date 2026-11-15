"""Frozen annual portfolio test for a Timestral-driven policy, with full ledgers."""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from backend.models.timestral_forecast import forecast_batch,decide,MODEL_ID,REVISION
from backend.stress_data import validate_bars

ROOT=Path('benchmark-runs/challenger');OUT=Path('benchmarks/challenger')


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def protocol():return json.loads((OUT/'protocol.json').read_text())


def download():
    import yfinance as yf
    cfg=protocol();records=[]
    for market,symbols in cfg['markets'].items():
        folder=ROOT/'prices'/market;folder.mkdir(parents=True,exist_ok=True)
        for ticker in symbols:
            p=folder/(ticker+'.csv')
            if not p.exists():
                frame=yf.download(ticker,start=cfg['price_start'],end=cfg['price_end_exclusive'],auto_adjust=False,actions=True,threads=False,progress=False)
                if isinstance(frame.columns,pd.MultiIndex):frame.columns=frame.columns.get_level_values(0)
                frame.index=pd.DatetimeIndex(frame.index).tz_localize(None);frame.index.name='Date';validate_bars(frame)
                frame.to_csv(p,float_format='%.10g')
            frame=pd.read_csv(p,index_col='Date',parse_dates=True);validate_bars(frame)
            records.append({'market':market,'ticker':ticker,'sha256':digest(p),'rows':len(frame),'first':str(frame.index[0].date()),'last':str(frame.index[-1].date()),'source':f'https://finance.yahoo.com/quote/{ticker}/history/'})
            print('Frozen prices',ticker,len(frame),flush=True)
    (ROOT/'price-manifest.json').write_text(json.dumps({'retrieved_utc':datetime.now(timezone.utc).isoformat(),'records':records,'basis':'Yahoo split-adjusted OHLC as delivered, separate cash dividends; revisions possible'},indent=2),encoding='utf8')


def read_prices(market):return {s:pd.read_csv(ROOT/'prices'/market/(s+'.csv'),index_col='Date',parse_dates=True) for s in protocol()['markets'][market]}


def generate():
    import torch
    torch.set_num_threads(2);cfg=protocol();cases=ROOT/'forecast-cases';cases.mkdir(exist_ok=True)
    signature=digest(OUT/'protocol.json');code=digest('backend/models/timestral_forecast.py');count=0
    for year in cfg['years']:
        for market in cfg['markets']:
            frames=read_prices(market);first=next(iter(frames.values()));dates=first.loc[f'{year}-01-01':f'{year}-12-31'].index
            if any(not f.loc[f'{year}-01-01':f'{year}-12-31'].index.equals(dates) for f in frames.values()):raise ValueError('Market calendars differ')
            for date in dates[:-5:cfg['signal_stride']]:
                paths=[cases/f'{year}-{market}-{s}-{date.date()}.json' for s in frames]
                fingerprint=hashlib.sha256(json.dumps([signature,code,MODEL_ID,REVISION,str(date),[(s,digest(ROOT/'prices'/market/(s+'.csv'))) for s in frames]],sort_keys=True).encode()).hexdigest()
                if all(p.exists() and json.loads(p.read_text())['fingerprint']==fingerprint for p in paths):continue
                pasts=[f.loc[:date] for f in frames.values()]
                forecasts=forecast_batch([p.Close.to_numpy()[-cfg['context_sessions']:] for p in pasts])
                for (ticker,frame),past,forecast,path in zip(frames.items(),pasts,forecasts,paths):
                    assert past.index[-1]==date
                    row={'id':path.stem,'year':year,'market':market,'ticker':ticker,'date':str(date.date()),'fingerprint':fingerprint,
                         'price_sha256':digest(ROOT/'prices'/market/(ticker+'.csv')),'forecast':forecast,
                         'timestral':decide(past,forecast,True,cfg['forecast_cost_hurdle'],cfg['annual_risk_target']),
                         'trend_vol':decide(past,forecast,False,cfg['forecast_cost_hurdle'],cfg['annual_risk_target'])}
                    # Reveal outcomes only after recording both decisions from past prices.
                    future=frame.loc[frame.index>date].head(5);assert len(future)==5
                    row['actual_future']=json.loads(future.reset_index().to_json(orient='records',date_format='iso'))
                    row['actual_5d_close_return']=float(future.Close.iloc[-1]/past.Close.iloc[-1]-1)
                    path.write_text(json.dumps(row),encoding='utf8');count+=1
                print('Forecasted',year,market,str(date.date()),'new cases',count,flush=True)


def rebalance(cash,shares,price,weight,fee):
    if not 0<=weight<=1 or price<=0 or not 0<=fee<1:raise ValueError('Invalid position target')
    value=cash+shares*price;old=shares*price;buy=weight*value>=old
    after=(value+fee*old)/(1+fee*weight) if buy else (value-fee*old)/(1-fee*weight)
    new_shares=weight*after/price;cost=fee*abs(new_shares-shares)*price
    new_cash=after-new_shares*price
    assert new_cash>=-1e-7 and abs(after-(value-cost))<1e-7
    return max(0.,new_cash),new_shares,cost


def simulate(frame,rows,year,variant,fee=.0015,initial=2500):
    bars=frame.loc[f'{year}-01-01':f'{year}-12-31'];signals={pd.Timestamp(r['date']):r for r in rows}
    cash=initial;shares=0.;pending=None;curve=[];ledger=[];started=False;costs=0.;dividends=0.;exposures=[]
    for i,(date,bar) in enumerate(bars.iterrows()):
        income=shares*float(bar.get('Dividends',0));cash+=income;dividends+=income
        if pending is not None and variant!='cash':
            weight=1. if variant=='buy_hold' else pending[variant]['sleeve_weight']
            if variant!='buy_hold' or not started:
                before=cash+shares*bar.Open;old_shares=shares
                cash,shares,cost=rebalance(cash,shares,float(bar.Open),weight,fee);costs+=cost
                ledger.append({'signal_date':pending['date'],'execution_date':str(date.date()),'price':float(bar.Open),'target_weight':weight,'shares_before':old_shares,'shares_after':shares,'fee':cost,'equity_before':before,'cash_after':cash})
                started=True
        pending=None
        if i==len(bars)-1 and shares:
            old=shares;before=cash+shares*bar.Close
            cash,shares,cost=rebalance(cash,shares,float(bar.Close),0,fee);costs+=cost
            ledger.append({'signal_date':None,'execution_date':str(date.date()),'price':float(bar.Close),'target_weight':0,'shares_before':old,'shares_after':0,'fee':cost,'equity_before':before,'cash_after':cash,'reason':'Final-close liquidation'})
        value=float(cash+shares*bar.Close);curve.append({'date':str(date.date()),'balance':value});exposures.append(float(shares*bar.Close/value))
        if date in signals:pending=signals[date]
    assert len(curve)==len(bars)
    return {'ending_balance':curve[-1]['balance'],'curve':curve,'ledger':ledger,'fees':costs,'dividends':dividends,'mean_exposure':float(np.mean(exposures))}


def report():
    cfg=protocol();groups={};allcases=[];full_ledger=[]
    for year in cfg['years']:
        for market in cfg['markets']:
            frames=read_prices(market);rows=[]
            for ticker,frame in frames.items():
                dates=frame.loc[f'{year}-01-01':f'{year}-12-31'].index[:-5:cfg['signal_stride']]
                for date in dates:
                    p=ROOT/'forecast-cases'/f'{year}-{market}-{ticker}-{date.date()}.json'
                    r=json.loads(p.read_text());assert r['price_sha256']==digest(ROOT/'prices'/market/(ticker+'.csv'));rows.append(r)
            allcases.extend(rows);variants={}
            for v in cfg['variants']:
                parts=[simulate(f,[r for r in rows if r['ticker']==s],year,v,cfg['one_way_cost_bps']/10000,cfg['initial_capital']/4) for s,f in frames.items()]
                values=np.array([[p['balance'] for p in x['curve']] for x in parts]).sum(axis=0);ret=values[-1]/cfg['initial_capital']-1
                dd=1-np.r_[cfg['initial_capital'],values]/np.maximum.accumulate(np.r_[cfg['initial_capital'],values])
                variants[v]={'ending_balance':round(float(values[-1]),2),'pnl':round(float(values[-1]-cfg['initial_capital']),2),'return_pct':float(ret*100),'above_10_percent':bool(ret>.1),
                             'max_drawdown_pct':float(dd.max()*100),'mean_exposure_pct':float(np.mean([p['mean_exposure'] for p in parts])*100),'fees':sum(p['fees'] for p in parts),'dividends':sum(p['dividends'] for p in parts),
                             'curve':[{'date':d['date'],'balance':float(x)} for d,x in zip(parts[0]['curve'],values)]}
                for ticker,p in zip(frames,parts):
                    for entry in p['ledger']:full_ledger.append({'year':year,'market':market,'ticker':ticker,'variant':v,**entry})
            costs={}
            for bps in cfg['cost_sensitivity_one_way_bps']:
                end=sum(simulate(f,[r for r in rows if r['ticker']==s],year,'timestral',bps/10000,cfg['initial_capital']/4)['ending_balance'] for s,f in frames.items())
                costs[str(bps)]={'ending_balance':end,'return_pct':100*(end/cfg['initial_capital']-1)}
            actual=np.array([r['actual_5d_close_return'] for r in rows]);pred=np.array([r['forecast']['median_return'] for r in rows])
            covered=[r['forecast']['quantile_prices']['0.1'][-1]<=r['actual_future'][-1]['Close']<=r['forecast']['quantile_prices']['0.9'][-1] for r in rows]
            groups[f'{year}-{market}']={'year':year,'market':market,'cases':len(rows),'variants':variants,'cost_sensitivity':costs,
                'forecast_metrics':{'median_absolute_return_error':float(np.median(abs(pred-actual))),'random_walk_median_absolute_return_error':float(np.median(abs(actual))),
                                    'direction_accuracy':float(np.mean((pred>0)==(actual>0))),'nominal_80_interval_coverage':float(np.mean(covered))}}
    summary={'protocol':cfg,'model':{'id':MODEL_ID,'revision':REVISION},'cases':len(allcases),'groups':groups,
             'price_manifest':json.loads((ROOT/'price-manifest.json').read_text()),'protocol_sha256':digest(OUT/'protocol.json'),
             'code_sha256':{p:digest(p) for p in ['backend/challenger_benchmark.py','backend/models/timestral_forecast.py']},
             'promoted_to_production':False,'news_attribution':'Qwen and ModernBERT sentiment are separately evaluated; no archived company-news tape was available for this portfolio test.'}
    OUT.mkdir(exist_ok=True);(OUT/'report.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    pd.DataFrame(full_ledger).to_csv(OUT/'trade-ledger.csv',index=False)
    public=Path('frontend/public');(public/'challenger-summary.json').write_text(json.dumps(summary),encoding='utf8')
    (public/'challenger-cases.json').write_text(json.dumps(allcases),encoding='utf8')
    lines=['# Timestral annual portfolio experiment','',cfg['target'],'','No parameters were selected on these results. This is a retrospective test of a 2026 foundation model; its pretraining overlap is unknown.','','| Year / market | Timestral P&L | Return | Trend/vol baseline | Buy-and-hold | 10% target met? |','|---|---:|---:|---:|---:|---|']
    for key,g in groups.items():
        v=g['variants'];t=v['timestral'];lines.append(f"| {key} | {t['pnl']:.2f} | {t['return_pct']:.2f}% | {v['trend_vol']['return_pct']:.2f}% | {v['buy_hold']['return_pct']:.2f}% | {'YES' if t['above_10_percent'] else 'NO'} |")
    lines+=['','All accounts start at 10,000; USD for US and INR for India. Profits are hypothetical after 0.15% costs on each traded amount, including final liquidation. Fractional shares, long-only, no leverage. Money stays invested between weekly rebalances; this differs from the earlier low-allocation committee.','','The trend/volatility baseline uses identical sizing without Timestral. It isolates whether the forecast gate improves results. The passive baseline begins on the same first executable date.','','News was unavailable for this portfolio experiment: do not credit Qwen or ModernBERT for these returns.','','## Limitations','',cfg['limitations']]
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps({k:{v:round(x['return_pct'],3) for v,x in g['variants'].items()} for k,g in groups.items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['download','generate','report']);args=p.parse_args();globals()[args.command]()
