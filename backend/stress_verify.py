"""Independently reconcile saved crisis evidence with the frozen source tape."""
import json
import hashlib
import math
from pathlib import Path
import pandas as pd
from backend.stress_benchmark import ROOT, PUBLIC, sha


def main():
    report=json.loads((PUBLIC/'report.json').read_text())
    tasks=json.loads((ROOT/'plan.json').read_text())['tasks']
    sources={s['id']:s for s in json.loads((PUBLIC/'news-sources.json').read_text())}
    balances={};checked=0
    for task in tasks:
        row=json.loads((ROOT/'cases'/(task['id']+'.json')).read_text())
        date=pd.Timestamp(row['date'])
        frame=pd.read_csv(ROOT/'data'/row['market']/(row['ticker']+'.csv'),index_col='Date',parse_dates=True)
        future=frame.loc[frame.index>date].iloc[:5]
        assert [str(x.date()) for x in future.index]==[x['Date'][:10] for x in row['future_bars']]
        assert sha(ROOT/'data'/row['market']/(row['ticker']+'.csv'))==row['data_sha256']
        assert row['quant']['as_of_date'][:10]==row['date']
        assert pd.Timestamp(row['quant']['learned_forecast']['training_data_end'])<date
        for source in row['news']['source_ids']:
            assert pd.Timestamp(sources[source]['available_from'])<=date
        for recorded,(_,actual) in zip(row['future_bars'],future.iterrows()):
            for col in ['Open','High','Low','Close']:
                assert math.isclose(recorded[col],actual[col],rel_tol=1e-10)
        for version in ['current','original']:
            t=row[version]['trade'];key=(row['episode'],row['market'],row['ticker'],version)
            balances.setdefault(key,2500.)
            if not t['traded']:
                assert t['weighted_return']==0
                continue
            direction=1 if t['action']=='LONG' else -1
            entry=float(future.iloc[0].Open)
            # Independent scalar reconciliation; does not call trade_path or equity_series.
            exit_date=pd.Timestamp(t['exit_date']);end=future.loc[exit_date]
            stop=t['stop_price']
            hits=future.index[future.Low<=stop] if direction==1 else future.index[future.High>=stop]
            assert exit_date==(hits[0] if len(hits) else future.index[-1])
            hit=end.Low<=stop if direction==1 else end.High>=stop
            expected_exit=(min(stop,end.Open) if direction==1 else max(stop,end.Open)) if hit else end.Close
            assert math.isclose(expected_exit,t['exit_price'],rel_tol=1e-10)
            dividends=float(future.loc[(future.index>future.index[0])&(future.index<=exit_date),'Dividends'].sum())
            borrow=.03*((exit_date-future.index[0]).days+1)/365 if direction==-1 else 0
            net=direction*(expected_exit-entry+dividends)/entry-.003-borrow
            assert math.isclose(net,t['net_return'],abs_tol=1e-10)
            balances[key]*=1+net*t['allocation'];checked+=1
    for episode,markets in report['results'].items():
        for market,result in markets.items():
            for version in ['current','original']:
                ending=sum(v for (e,m,t,c),v in balances.items() if (e,m,c)==(episode,market,version))
                assert abs(ending-result['variants'][version]['ending_balance'])<=.011
    result={'cases_verified':len(tasks),'active_trades_reconciled':checked,
            'checks':['source tape matches all future bars','training ends before signal','macro sources available before signal','independent fee/dividend/borrow arithmetic','independent compounded ending balances'],
            'report_sha256':hashlib.sha256((PUBLIC/'report.json').read_text(encoding='utf8').encode('utf8')).hexdigest(),
            'report_hash_basis':'UTF-8 with LF-normalized line endings, so Git checkout settings do not change the hash'}
    (PUBLIC/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
