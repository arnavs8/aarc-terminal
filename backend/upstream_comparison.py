"""Matched-data comparison, with common executable accounting and no synthetic prices."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import numpy as np
import pandas as pd
from backend.committee_audit import finalize_verdict
from backend.committee_benchmark import checks
from backend.stress_benchmark import trade_path,config

ROOT=Path('benchmark-runs/upstream-comparison');OUT=Path('benchmarks/upstream-comparison')


def target_trade(verdict,future,cost_bps=30):
    """Adapt upstream's fixed 2R target; retain next-open entry, fees and dividends."""
    base=trade_path(verdict,future,cost_bps)
    if not base['traded']:return base
    direction=1 if verdict['action']=='LONG' else -1
    entry=base['entry_price'];stop=base['stop_price'];target=entry*(1+direction*2*verdict['stop_loss_pct']/100)
    dividends=0.;closed=False;path=[];net=0.;hit_stop=False;reason='Fifth close'
    for date,bar in future.iterrows():
        if not closed:
            if date>future.index[0]:dividends+=float(bar.get('Dividends',0))
            stop_hit=bar.Low<=stop if direction==1 else bar.High>=stop
            target_hit=bar.High>=target if direction==1 else bar.Low<=target
            price=float(bar.Close);exit_now=stop_hit or target_hit or date==future.index[-1]
            # A stop takes priority if both thresholds lie inside one daily bar.
            if stop_hit:price=min(stop,float(bar.Open)) if direction==1 else max(stop,float(bar.Open));reason='Stop';hit_stop=True
            elif target_hit:price=max(target,float(bar.Open)) if direction==1 else min(target,float(bar.Open));reason='2R target'
            borrow=(max(1,(date-future.index[0]).days+1)/365*.03) if direction==-1 else 0.
            gross=direction*(price/entry-1)+direction*dividends/entry
            net=gross-borrow-cost_bps/20000-(cost_bps/20000 if exit_now else 0)
            if exit_now:closed=True;exit_date=str(date.date());exit_price=price
        path.append({'date':str(date.date()),'return':net})
    return {**base,'net_return':net,'weighted_return':base['allocation']*net,'gross_return':gross,'dividends_per_share':dividends,'borrow_cost_fraction':borrow,'path':path,'target_price':target,'stop_hit':hit_stop,'exit_date':exit_date,'exit_price':exit_price,'exit_reason':reason}


def portfolio(rows,variant,episode,market,cost=30):
    cfg=config();parts=[]
    for ticker in cfg['symbols'][market]:
        subset=sorted([r for r in rows if r['ticker']==ticker],key=lambda r:r['date'])
        frame=pd.read_csv(Path('benchmark-runs/stress/data')/market/(ticker+'.csv'),index_col='Date',parse_dates=True)
        start=frame.index[frame.index>pd.Timestamp(subset[0]['date'])][0];dates=frame.loc[start:cfg['periods'][episode]['end']].index
        initial=cfg['starting_capital']/len(cfg['symbols'][market]);curve=pd.Series(initial,index=dates,dtype=float);balance=initial;last_exit=None
        for r in subset:
            verdict=r['variants'][variant]['events']['cro'];future=pd.DataFrame(r['future_bars']);future['Date']=pd.to_datetime(future.Date);future=future.set_index('Date')
            trade=(target_trade if variant=='fork_2r_target' else trade_path)(verdict,future,cost)
            if not trade['traded']:continue
            assert future.index[0]>pd.Timestamp(r['date'])
            assert last_exit is None or future.index[0]>last_exit
            for point in trade['path']:curve.loc[pd.Timestamp(point['date'])]=balance*(1+trade['allocation']*point['return'])
            balance*=1+trade['weighted_return'];last_exit=future.index[-1];curve.loc[curve.index>last_exit]=balance
        parts.append(curve)
    total=pd.concat(parts,axis=1).ffill().sum(axis=1);values=np.r_[10000.,total.to_numpy()]
    return {'ending_balance':float(total.iloc[-1]),'pnl':float(total.iloc[-1]-10000),'return_pct':float((total.iloc[-1]/10000-1)*100),'max_drawdown_pct':float((1-values/np.maximum.accumulate(values)).max()*100),'curve':[{'date':str(d.date()),'balance':float(v)} for d,v in total.items()]}


def main():
    paths=sorted(Path('benchmark-runs/stress/cases').glob('*.json'));n=len(paths)
    full_ready=all((ROOT/'upstream'/p.name).exists() for p in paths)
    after_ready=all((ROOT/'fork-after'/p.name).exists() for p in paths)
    variants=['fork_before','upstream_matched','upstream_risk_capped','fork_2r_target']
    if full_ready:variants.append('upstream_full')
    if after_ready:variants.append('fork_after')
    rows=[]
    for p in paths:
        source=json.loads(p.read_text());fork=json.loads((ROOT/'fork-before'/p.name).read_text());up=json.loads((ROOT/'upstream-matched'/p.name).read_text())
        guarded=deepcopy(up);guarded['events']['cro']=finalize_verdict(up['events']['cro'],source['quant'],source['news'])
        guarded['events']['cro'].pop('execution_plan',None) # Isolate risk caps from the new exit policy.
        inputs={'fork_before':fork,'upstream_matched':up,'upstream_risk_capped':guarded,'fork_2r_target':fork}
        if full_ready:inputs['upstream_full']=json.loads((ROOT/'upstream'/p.name).read_text())
        if after_ready:inputs['fork_after']=json.loads((ROOT/'fork-after'/p.name).read_text())
        future=pd.DataFrame(source['future_bars']);future['Date']=pd.to_datetime(future.Date);future=future.set_index('Date')
        row={k:source[k] for k in ('id','date','ticker','episode','market','future_bars')};row['variants']={}
        for name,data in inputs.items():
            v=data['events']['cro'];trade=(target_trade if name=='fork_2r_target' else trade_path)(v,future)
            audit=checks(data['quant'],data['events']);audit['no_invented_future_label']=data['quant'].get('time_machine_actual_return_pct') is None
            row['variants'][name]={'events':{k:data['events'][k] for k in ('bull','bear','cro')},'trade':trade,'checks':audit}
        rows.append(row)
    groups={}
    for episode in config()['periods']:
        for market in config()['symbols']:
            subset=[r for r in rows if r['episode']==episode and r['market']==market];group={}
            for v in variants:
                trades=[r['variants'][v]['trade'] for r in subset];active=[t for t in trades if t['traded']]
                group[v]={**portfolio(subset,v,episode,market),'trades':len(active),'wins':sum(t['net_return']>0 for t in active),'mean_allocation_pct':float(np.mean([t['allocation'] for t in trades])*100),'cost_sensitivity':{str(c):{k:x for k,x in portfolio(subset,v,episode,market,c).items() if k!='curve'} for c in [0,30,100]}}
            groups[episode+'-'+market]={'cases':len(subset),'variants':group}
    totals={v:{k:sum(r['variants'][v]['checks'][k] for r in rows) for k in rows[0]['variants'][v]['checks']} for v in variants}
    report={'upstream_commit':'49bf9c0','fork_before_commit':'582e3b0ec420fc4b69a82f6e0397f64c9adae040','cases':n,'full_upstream_complete':full_ready,'fork_after_complete':after_ready,'groups':groups,'checks_passed':totals,'scope':['356 dated COVID/2022 cases, eight instruments, identical frozen market tape and dated macro-news scores.','No external language-model generation in this reproducible replay; compares deterministic production paths and calibration defaults.','Each source package runs in an isolated process with empty calibration state; no live state or provider keys affect comparison.','Matched-input variants share quant snapshots; full upstream variant recomputes its own features from only past bars.','Same next-open entry, five-session maximum hold, stops with adverse gaps, dividends, 3% assumed annual short borrow and 30-bps round-trip costs.','No leverage, same starting capital and sleeve accounting. Zero-exposure cases do not count as wins.','These are period returns, not annualized; not comparable to the separate six-year ETF policy.','2R target is a prespecified upstream execution idea, not fitted on these outcomes.']}
    OUT.mkdir(exist_ok=True,parents=True);(OUT/'report.json').write_text(json.dumps(report,indent=2))
    with gzip.open(OUT/'cases.json.gz','wt',encoding='utf8') as stream:json.dump(rows,stream)
    Path('frontend/public/upstream-comparison.json').write_text(json.dumps(report))
    public=[]
    for row in rows:
        public.append({**{k:row[k] for k in ('id','date','ticker','episode','market')},'variants':{
            name:{'action':x['events']['cro']['action'],'allocation_pct':x['events']['cro']['recommended_allocation_pct'],'summary':x['events']['cro']['verdict_summary'],'trade_return_pct':x['trade']['net_return']*100,'portfolio_contribution_pct':x['trade']['weighted_return']*100,'checks':x['checks']} for name,x in row['variants'].items() if name in ('fork_before','upstream_matched','upstream_full','fork_after')}})
    Path('frontend/public/upstream-cases.json').write_text(json.dumps(public))
    print(json.dumps({k:{v:round(x['return_pct'],4) for v,x in g['variants'].items()} for k,g in groups.items()},indent=2))


if __name__=='__main__':main()
