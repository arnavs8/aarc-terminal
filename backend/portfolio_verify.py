"""Independent daily cash/position reconciliation against frozen source prices."""
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    r=json.loads(Path('benchmarks/strategy/report.json').read_text())
    frames={s:pd.read_csv('benchmark-runs/strategy/'+s+'.csv',index_col='Date',parse_dates=True) for s in r['protocol']['universe']}
    syms=list(frames);count=0
    for kind in ['strategy','buy_hold_spy']:
        out=r[kind];cash=10000.;shares=np.zeros(4)
        for point in out['curve']:
            date=pd.Timestamp(point['date']);cash+=sum(shares[i]*frames[s].loc[date,'Dividends'] for i,s in enumerate(syms))
            for e in [e for e in out['ledger'] if e['execution_date']==point['date']]:
                px=np.array([frames[s].loc[date,'Close' if e.get('reason') else 'Open'] for s in syms])
                assert np.allclose(px,list(e['prices'].values()))
                new=np.array(e['shares_after']);assert np.allclose(shares,e['shares_before'])
                fee=sum(abs(new-shares)*px)*.0015;cash-=sum((new-shares)*px)+fee;shares=new
                assert abs(cash-e['cash_after'])<1e-6 and abs(fee-e['fee'])<1e-6
                assert cash>=-1e-6 and (shares>=-1e-8).all()
                if e['signal_date']:assert pd.Timestamp(e['signal_date'])<date
                count+=1
            balance=cash+sum(shares[i]*frames[s].loc[date,'Close'] for i,s in enumerate(syms))
            assert abs(balance-point['balance'])<1e-6
        assert abs(cash-out['ending_balance'])<1e-6
    result={'passed':True,'transactions':count,'portfolios':2,'checks':['Each daily balance reconstructed from raw prices, dividend cash, shares and fees','Signals precede executions','No negative cash or short positions','No recomputed strategy targets used in reconciliation']}
    Path('benchmarks/strategy/verification.json').write_text(json.dumps(result,indent=2));print(result)


if __name__=='__main__':main()
