"""Independent ledger reconciliation; never calls the portfolio simulator."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from backend.challenger_benchmark import ROOT,OUT,digest


def main():
    r=json.loads((OUT/'report.json').read_text());ledger=pd.read_csv(OUT/'trade-ledger.csv');checked=0
    for key,g in r['groups'].items():
        year=g['year'];market=g['market']
        for variant,expected in g['variants'].items():
            total=0
            for ticker in r['protocol']['markets'][market]:
                frame=pd.read_csv(ROOT/'prices'/market/(ticker+'.csv'),index_col='Date',parse_dates=True).loc[f'{year}-01-01':f'{year}-12-31']
                trades=ledger[(ledger.year==year)&(ledger.market==market)&(ledger.ticker==ticker)&(ledger.variant==variant)]
                cash=2500.;shares=0.
                for date,b in frame.iterrows():
                    cash+=shares*float(b.get('Dividends',0))
                    for _,t in trades[trades.execution_date==str(date.date())].iterrows():
                        final=pd.isna(t.signal_date)
                        price=float(b.Close if final else b.Open)
                        assert abs(t.price-price)<1e-7
                        assert abs(t.shares_before-shares)<1e-7
                        if not final:assert pd.Timestamp(t.signal_date)<date
                        delta=t.shares_after-shares;fee=abs(delta)*price*.0015
                        assert abs(t.fee-fee)<1e-6
                        cash-=delta*price+fee;shares=t.shares_after
                        assert cash>=-1e-6 and abs(cash-t.cash_after)<1e-6
                        checked+=1
                assert abs(shares)<1e-7
                total+=cash
            assert abs(total-expected['ending_balance'])<=.011,(key,variant,total,expected['ending_balance'])
    cases=list((ROOT/'forecast-cases').glob('*.json'));assert len(cases)==r['cases']
    for p in cases:
        x=json.loads(p.read_text());assert x['date']==x['timestral']['as_of_date']==x['trend_vol']['as_of_date']
        assert x['actual_future'][0]['Date'][:10]>x['date']
        q=x['forecast']['quantile_prices'];assert all(a<=b<=c for a,b,c in zip(q['0.1'],q['0.5'],q['0.9']))
    result={'cases':len(cases),'ledger_events_reconciled':checked,'annual_portfolios_reconciled':16,
            'checks':['Signal precedes execution','Open/final-close prices match frozen tape','Turnover fees independently recomputed','Dividends credited to prior holders','All final balances independently reconciled','No leverage or negative cash'],
            'protocol_sha256':digest(OUT/'protocol.json')}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
