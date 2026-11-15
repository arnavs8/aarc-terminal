"""Independent cash-return reconciliation for the pinned upstream comparison."""
import gzip,json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    root=Path('benchmarks/upstream-comparison');report=json.loads((root/'report.json').read_text())
    with gzip.open(root/'cases.json.gz','rt',encoding='utf8') as f:rows=json.load(f)
    count=0
    for row in rows:
        bars=row['future_bars'];entry=float(bars[0]['Open']);entry_date=pd.Timestamp(bars[0]['Date'])
        assert entry_date>pd.Timestamp(row['date'])
        for variant,result in row['variants'].items():
            trade=result['trade'];v=result['events']['cro']
            if not trade['traded']:assert trade['weighted_return']==0;continue
            direction=1 if v['action']=='LONG' else -1;stop=entry*(1-direction*v['stop_loss_pct']/100)
            multiple=2 if variant=='fork_2r_target' else v.get('execution_plan',{}).get('take_profit_r_multiple')
            target=entry*(1+direction*v['stop_loss_pct']/100*multiple) if multiple else None
            div=0.
            for i,bar in enumerate(bars):
                date=pd.Timestamp(bar['Date'])
                if i:div+=float(bar.get('Dividends',0))
                hit_stop=(bar['Low']<=stop) if direction==1 else (bar['High']>=stop)
                hit_target=target is not None and ((bar['High']>=target) if direction==1 else (bar['Low']<=target))
                if not (hit_stop or hit_target or i==len(bars)-1):continue
                if hit_stop:price=min(bar['Open'],stop) if direction==1 else max(bar['Open'],stop)
                elif hit_target:price=max(bar['Open'],target) if direction==1 else min(bar['Open'],target)
                else:price=bar['Close']
                borrow=max(1,(date-entry_date).days+1)/365*.03 if direction==-1 else 0.
                net=direction*(price-entry+div)/entry-.003-borrow
                assert abs(price-trade['exit_price'])<1e-8
                assert str(date.date())==trade['exit_date']
                assert abs(net-trade['net_return'])<1e-8
                assert abs(net*v['recommended_allocation_pct']/100-trade['weighted_return'])<1e-8
                count+=1;break
        if 'fork_after' in row['variants']:
            a=row['variants']['fork_before']['events']['cro'];b=row['variants']['fork_after']['events']['cro']
            assert a['action']==b['action'] and a['recommended_allocation_pct']==b['recommended_allocation_pct']
    for key,group in report['groups'].items():
        episode,market=key.split('-');subset=[r for r in rows if r['episode']==episode and r['market']==market];tickers=sorted({r['ticker'] for r in subset})
        for variant,expected in group['variants'].items():
            end=0.
            for ticker in tickers:
                balance=10000/len(tickers)
                for row in sorted([r for r in subset if r['ticker']==ticker],key=lambda r:r['date']):balance*=1+row['variants'][variant]['trade']['weighted_return']
                end+=balance
            assert abs(end-expected['ending_balance'])<1e-6
    result={'passed':True,'cases':len(rows),'active_trades_checked':count,'portfolios':sum(len(x['variants']) for x in report['groups'].values()),'checks':['Independent exit prices, stop/target ordering, dividends, borrow and cost reconciliation','Next-open entry strictly after signal','Every final basket balance rebuilt from trade returns','Updated fork uses identical directions and allocation sizes to the previous fork on the frozen replay']}
    (root/'verification.json').write_text(json.dumps(result,indent=2));print(result)


if __name__=='__main__':main()
