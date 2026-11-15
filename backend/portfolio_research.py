"""Train a small transparent allocation policy on 2011-2019; freeze before test.

Separate USD ETF portfolio experiment, never a claim about the stock committee.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path('benchmark-runs/strategy');OUT=Path('benchmarks/strategy')


def targets(history, lookback, risk_target):
    """Only completed past bars; total returns for ranking, no future labels."""
    scores={};vols={}
    for symbol,f in history.items():
        if len(f)<lookback+1:continue
        ret=(f.Close+f.Dividends)/f.Close.shift(1)-1
        scores[symbol]=float((1+ret.tail(lookback)).prod()-1)
        vols[symbol]=float(ret.tail(60).std()*np.sqrt(252))
    chosen=sorted(scores,key=lambda s:(-scores[s],s))[:2]
    return {s:(.5*min(1.,risk_target/max(vols[s],.001)) if s in chosen and scores[s]>0 else 0.) for s in history}


def rebalance_portfolio(cash,shares,prices,weights,fee):
    old=shares*prices;before=float(cash+old.sum());after=before
    # Fixed point accounts for fees before allocating the remaining equity.
    for _ in range(50):
        new=before-fee*float(abs(weights*after-old).sum())
        if abs(new-after)<1e-10:break
        after=new
    wanted=weights*new;cost=fee*float(abs(wanted-old).sum())
    cash=before-cost-float(wanted.sum())
    assert cash>=-1e-7 and abs(cash+wanted.sum()+cost-before)<1e-7
    return max(0.,cash),wanted/prices,cost


def simulate(frames,start,end,lookback=252,risk_target=.25,passive=False,fee=.0015):
    symbols=list(frames);dates=frames[symbols[0]].loc[start:end].index
    assert all(f.loc[start:end].index.equals(dates) for f in frames.values())
    cash=10000.;shares=np.zeros(len(symbols));curve=[];ledger=[];previous_month=None
    for date in dates:
        bars=[frames[s].loc[date] for s in symbols]
        dividends=float(sum(n*b.Dividends for n,b in zip(shares,bars)));cash+=dividends
        month=(date.year,date.month)
        if previous_month!=month and (not passive or previous_month is None):
            past={s:f.loc[f.index<date] for s,f in frames.items()}
            weights=({'SPY':1.} if passive else targets(past,lookback,risk_target))
            w=np.array([weights.get(s,0.) for s in symbols]);prices=np.array([b.Open for b in bars])
            before=float(cash+sum(shares*prices));old=shares.copy()
            cash,shares,cost=rebalance_portfolio(cash,shares,prices,w,fee)
            ledger.append({'signal_date':str(past[symbols[0]].index[-1].date()),'execution_date':str(date.date()),'weights':weights,'prices':dict(zip(symbols,map(float,prices))),'shares_before':old.tolist(),'shares_after':shares.tolist(),'cash_after':cash,'fee':cost,'equity_before':before})
        previous_month=month
        value=float(cash+sum(n*b.Close for n,b in zip(shares,bars)))
        if date==dates[-1]:
            prices=np.array([b.Close for b in bars]);old=shares.copy()
            cash,shares,cost=rebalance_portfolio(cash,shares,prices,np.zeros(len(symbols)),fee)
            ledger.append({'execution_date':str(date.date()),'signal_date':None,'reason':'Final close liquidation','prices':dict(zip(symbols,map(float,prices))),'shares_before':old.tolist(),'shares_after':shares.tolist(),'cash_after':cash,'fee':cost,'equity_before':value,'weights':{s:0 for s in symbols}});value=cash
        curve.append({'date':str(date.date()),'balance':value,'dividends':dividends})
    values=np.r_[10000.,[x['balance'] for x in curve]];years=len(dates)/252
    annual={};base=10000.
    for year in sorted({d.year for d in dates}):
        last=next(x['balance'] for x in reversed(curve) if x['date'].startswith(str(year)))
        annual[str(year)]=100*(last/base-1);base=last
    return {'ending_balance':float(values[-1]),'pnl':float(values[-1]-10000),'total_return_pct':100*(values[-1]/10000-1),'cagr_pct':100*((values[-1]/10000)**(1/years)-1),'max_drawdown_pct':100*float((1-values/np.maximum.accumulate(values)).max()),'annual_returns':annual,'curve':curve,'ledger':ledger,'fees':sum(x['fee'] for x in ledger)}


def main():
    cfg=json.loads((OUT/'protocol.json').read_text());frames={s:pd.read_csv(ROOT/(s+'.csv'),index_col='Date',parse_dates=True) for s in cfg['universe']}
    trials=[]
    for lb in cfg['candidates']['lookback_sessions']:
        for risk in cfg['candidates']['annual_vol_target']:
            r=simulate(frames,cfg['training_start'],cfg['training_end'],lb,risk)
            trials.append({'lookback':lb,'risk_target':risk,'cagr_pct':r['cagr_pct'],'max_drawdown_pct':r['max_drawdown_pct'],'selection_score':r['cagr_pct']-.5*r['max_drawdown_pct']})
    chosen=max(trials,key=lambda x:x['selection_score'])
    frozen={'selected':chosen,'training_trials':trials,'protocol_sha256':hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()}
    model=OUT/'frozen-policy.json'
    if model.exists():assert json.loads(model.read_text())==frozen,'Frozen training selection changed'
    else:model.write_text(json.dumps(frozen,indent=2))
    # Test outcomes are calculated only after training policy is written.
    test=simulate(frames,cfg['test_start'],cfg['test_end'],chosen['lookback'],chosen['risk_target'])
    baseline=simulate(frames,cfg['test_start'],cfg['test_end'],passive=True)
    report={'protocol':cfg,'frozen_policy':frozen,'strategy':test,'buy_hold_spy':baseline,'cost_sensitivity':{str(b):{k:v for k,v in simulate(frames,cfg['test_start'],cfg['test_end'],chosen['lookback'],chosen['risk_target'],fee=b/10000).items() if k not in ('curve','ledger')} for b in (0,15,50)},'prices_sha256':{s:hashlib.sha256((ROOT/(s+'.csv')).read_bytes()).hexdigest() for s in frames},'production_enabled':False}
    (OUT/'report.json').write_text(json.dumps(report,indent=2));Path('frontend/public/strategy-report.json').write_text(json.dumps(report))
    print(json.dumps({'selected':chosen,'strategy':{k:v for k,v in test.items() if k not in ('curve','ledger')},'baseline':{k:v for k,v in baseline.items() if k not in ('curve','ledger')}},indent=2))


if __name__=='__main__':main()
