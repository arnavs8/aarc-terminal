"""Actual five-second flash-crash replay and explicitly synthetic volume shocks."""
import hashlib
import gzip
import json
from pathlib import Path
import numpy as np
import pandas as pd
from backend.market_guard import assess

ROOT=Path('benchmark-runs/flash-crash');OUT=Path('benchmarks/flash-crash')
REVISION='07322c1747957fb0e9ab45dc9dcc67d33d80293b'


def inject(frame,direction,multiplier,start='14:00:00',seconds=300):
    result=frame.copy(deep=True)
    if direction=='none':return result
    begin=pd.Timestamp(str(frame.index[0].date())+' '+start);mask=(frame.index>=begin)&(frame.index<begin+pd.Timedelta(seconds=seconds))
    result.loc[mask,'trading_volume']*=multiplier
    if direction in ('bullish','bearish'):
        sign=1 if direction=='bullish' else -1
        result.loc[mask,'signed_volume']=sign*.8*result.loc[mask,'trading_volume']/result.loc[mask,'midquote']
    # Historical prices, spreads and depths remain unchanged: sensitivity, not market impact.
    return result


def replay(frame,mode='guard',latency=5,slippage_bps=0,allocation=1.):
    begin=pd.Timestamp(str(frame.index[0].date())+' 13:55:00');start=frame.index.searchsorted(begin)
    cash=10000.;shares=0.;pending=None;ledger=[];curve=[];decisions=[];target=allocation;entry=None
    for i in range(start,len(frame)):
        date=frame.index[i];bar=frame.iloc[i];mid=float(bar.midquote);half=float(bar.spread)/20000
        if i==start:
            price=mid*(1+half+slippage_bps/10000);shares=10000*allocation/(price*1.0015);cash-=shares*price*1.0015;entry=price
            ledger.append({'time':str(date),'side':'BUY','shares':shares,'price':price,'fee':shares*price*.0015,'cash_after':cash})
        if pending and date>=pending['time']:
            # Only 10% of observed best bid depth is assumed executable per five-second bar.
            want=max(0.,shares-pending['shares']);qty=min(want,max(0.,float(bar.depth0_bid))*.1)
            price=mid*(1-half-slippage_bps/10000);cash+=qty*price*.9985;shares-=qty
            if qty>0:ledger.append({'time':str(date),'side':'SELL','shares':qty,'price':price,'fee':qty*price*.0015,'cash_after':cash,'reason':pending['reason']})
            if shares<=pending['shares']+1e-9:pending=None
        if mode=='guard':
            guard=assess(frame.iloc[:i+1]);scale=guard['exposure_multiplier']
            if scale<target and pending is None:
                target=scale;equity=cash+shares*mid
                pending={'time':date+pd.Timedelta(seconds=latency),'shares':equity*scale/mid,'reason':guard['reason']}
                decisions.append({'time':str(date),**guard})
        elif mode=='stop' and mid<=entry*.98 and pending is None and shares>0:
            pending={'time':date+pd.Timedelta(seconds=latency),'shares':0.,'reason':'2% fixed stop observed at sampled quote'}
        if i==len(frame)-1 and shares>0:
            # Last-quote liquidation is a valuation assumption, not a guaranteed auction fill.
            price=mid*(1-half-slippage_bps/10000);cash+=shares*price*.9985
            ledger.append({'time':str(date),'side':'SELL','shares':shares,'price':price,'fee':shares*price*.0015,'cash_after':cash,'reason':'Final sampled quote liquidation'});shares=0
        curve.append({'time':str(date),'balance':cash+shares*mid,'midquote':mid})
    values=np.r_[10000.,[x['balance'] for x in curve]]
    return {'ending_balance':float(values[-1]),'pnl':float(values[-1]-10000),'return_pct':100*(values[-1]/10000-1),'max_drawdown_pct':100*float((1-values/np.maximum.accumulate(values)).max()),'decisions':decisions,'ledger':ledger,'curve':curve}


def load(day):
    f=pd.read_parquet(ROOT/(day+'.parquet'))
    # The publisher encodes New York session wall-clock times with a UTC label.
    # Its extraction code uses UTC origin for seconds since local midnight.
    f.index=pd.DatetimeIndex(f.pop('ts')).tz_localize(None)
    assert len(f)==4680 and f.index.is_monotonic_increasing and not f.index.has_duplicates
    assert (f.index.to_series().diff().dropna()==pd.Timedelta(seconds=5)).all()
    return f


def verify(result):
    cash=10000.;shares=0.
    for row in result['ledger']:
        sign=1 if row['side']=='BUY' else -1
        cash-=sign*row['shares']*row['price']+row['fee'];shares+=sign*row['shares']
        assert abs(cash-row['cash_after'])<1e-6 and shares>=-1e-8 and cash>=-1e-6
    assert abs(shares)<1e-8 and abs(cash-result['ending_balance'])<1e-6


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cfg={'entry':'13:55:00','injection':'14:00:00 for 300 seconds','volume_multipliers':[2,5,10],'directions':['bullish','bearish','unsigned'],'flow_imbalance':.8,'latency_seconds':[5,30,60],'extra_slippage_bps':[0,10,50],'fee_bps_per_side':15,'maximum_participation_best_bid_depth':.1,'guard_rules':'One-minute adverse move >=1% or spread >=10bps -> exit; depth <=25% prior median or volume >=3x and adverse signed imbalance >=60% -> halve; no reentry','freeze_note':'Rules fixed before event replay; designed with knowledge of flash-crash mechanism, not blind discovery'}
    (OUT/'protocol.json').write_text(json.dumps(cfg,indent=2))
    results={};count=0
    for day in ['2010-05-05','2010-05-06','2010-05-07']:
        f=load(day);out={}
        for mode in ['hold','stop','guard']:
            value=replay(f,mode);verify(value);out[mode]=value;count+=1
        results[day]=out;print('Replayed',day,flush=True)
    f=load('2010-05-06');scenarios=[]
    for direction in cfg['directions']:
        for multiplier in cfg['volume_multipliers']:
            changed=inject(f,direction,multiplier);value=replay(changed);verify(value);count+=1
            scenarios.append({'direction':direction,'multiplier':multiplier,**value});print('Injected',direction,multiplier,flush=True)
    sensitivity=[]
    for latency in cfg['latency_seconds']:
        for slip in cfg['extra_slippage_bps']:
            value=replay(f,latency=latency,slippage_bps=slip);verify(value);count+=1
            sensitivity.append({'latency_seconds':latency,'extra_slippage_bps':slip,**{k:v for k,v in value.items() if k not in ('curve','ledger','decisions')}})
    source={'dataset':'voigtstefan/sp500','revision':REVISION,'url':'https://huggingface.co/datasets/voigtstefan/sp500','license':'MIT','citation':'Menkveld, Hautsch and Voigt (2026), Market Responses to a VIX Shock','extraction_code':'https://github.com/voigtstefan/market-responses-to-a-vix-shock/blob/main/01-lobster-extract-data.R','unit_note':'Extraction code establishes signed volume in shares, total volume in USD, spread in bps; dataset card prose has inconsistent unit/window descriptions. Session wall-clock encoded as UTC was preserved as New York time, not shifted.','sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.parquet')}}
    report={'protocol':cfg,'source':source,'historical':results,'synthetic_volume_scenarios':scenarios,'execution_sensitivity':sensitivity,'verification':{'passed':True,'replays':count},'limitations':['Explicit hypothetical pre-existing long position at 13:55; this is not an entry predicted by the daily committee.','Five-second aggregated quotes are not full consolidated tick-by-tick executable fills.','Spread/depth are time-weighted aggregates; estimated execution includes fees, latency and depth caps but not queue position or market impact.','Volume injections alter inputs only; they do not reconstruct how other traders would change prices.','These scenario returns are intraday, never annualized.','Daily committee replay is reported separately; a daily model cannot claim to have forecast a five-minute crash.']}
    with gzip.open(OUT/'full-replay.json.gz','wt',encoding='utf8') as stream:json.dump(report,stream)
    for group in report['historical'].values():
        for value in group.values():value['curve']=value['curve'][::12]+value['curve'][-1:]
    for value in report['synthetic_volume_scenarios']:value['curve']=value['curve'][::12]+value['curve'][-1:]
    (OUT/'report.json').write_text(json.dumps(report,indent=2));Path('frontend/public/flash-summary.json').write_text(json.dumps(report))
    print(json.dumps({d:{m:round(r['return_pct'],3) for m,r in x.items()} for d,x in results.items()},indent=2))


if __name__=='__main__':main()
