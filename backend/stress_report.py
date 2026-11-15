"""Independent accounting checks for crisis replays, without changing the model."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from backend.stress_benchmark import ROOT, PUBLIC, config, trade_path, sha

VARIANTS=['current','original','matched_long','no_short','price_only']

def selected_trade(row,variant):
    return row[variant]['trade'] if variant in ('current','original','price_only') else row[variant]


def equity_series(frame, rows, variant, start, end, initial=10000, cost_bps=30):
    dates=frame.loc[start:end].index
    equity=pd.Series(initial,index=dates,dtype=float)
    balance=initial
    for row in sorted(rows,key=lambda r:r['date']):
        trade=selected_trade(row,variant)
        if not trade['traded']: continue
        future=pd.DataFrame(row['future_bars']);future['Date']=pd.to_datetime(future.Date);future=future.set_index('Date')
        if variant in ('current','original','price_only'):
            verdict=row[variant]['events']['cro']
        else:
            verdict=dict(row['current']['events']['cro'])
            if variant=='matched_long': verdict['action']='LONG'
            if variant=='no_short' and verdict['action']=='SHORT':verdict['recommended_allocation_pct']=0
        trade=trade_path(verdict,future,cost_bps)
        for point in trade['path']:
            equity.loc[pd.Timestamp(point['date'])]=balance*(1+trade['allocation']*point['return'])
        balance*=1+trade['weighted_return']
        equity.loc[equity.index>pd.Timestamp(trade['path'][-1]['date'])]=balance
    return equity


def held_stock(frame,start,end,initial=10000,cost_bps=30):
    bars=frame.loc[start:end];entry=float(bars.iloc[0].Open)
    div=bars.get('Dividends',pd.Series(0.,index=bars.index)).copy();div.iloc[0]=0
    result=initial*(bars.Close/entry+div.cumsum()/entry-cost_bps/20000)
    result.iloc[-1]-=initial*cost_bps/20000
    return result


def drawdown(series,initial):
    values=np.r_[initial,series.to_numpy()]
    return float(np.max(1-values/np.maximum.accumulate(values))*100)


def basket(rows,episode,market,variant,cost_bps=30):
    cfg=config();symbols=cfg['symbols'][market];period=cfg['periods'][episode];parts=[]
    for ticker in symbols:
        subset=[r for r in rows if r['ticker']==ticker]
        frame=pd.read_csv(ROOT/'data'/market/(ticker+'.csv'),index_col='Date',parse_dates=True)
        first_signal=pd.Timestamp(subset[0]['date']);start=frame.index[frame.index>first_signal][0]
        if variant=='hold': curve=held_stock(frame,start,period['end'],cfg['starting_capital']/len(symbols),cost_bps)
        else: curve=equity_series(frame,subset,variant,start,period['end'],cfg['starting_capital']/len(symbols),cost_bps)
        parts.append(curve.rename(ticker))
    aligned=pd.concat(parts,axis=1).ffill().fillna(cfg['starting_capital']/len(symbols))
    curve=aligned.sum(axis=1)
    return {'ending_balance':round(float(curve.iloc[-1]),2),'return_pct':round((float(curve.iloc[-1])/cfg['starting_capital']-1)*100,4),
        'max_closing_drawdown_pct':round(drawdown(curve,cfg['starting_capital']),4),
        'curve':[{'date':str(d.date()),'balance':round(float(v),4)} for d,v in curve.items()]}


def summarize(rows):
    cfg=config();out={}
    for episode in cfg['periods']:
        out[episode]={}
        for market in cfg['symbols']:
            group=[r for r in rows if r['episode']==episode and r['market']==market]
            result={'cases':len(group),'news_cases':sum(r['news']['news_available'] for r in group),'variants':{}}
            for v in VARIANTS:
                trades=[selected_trade(r,v) for r in group];active=[t for t in trades if t['traded']]
                wins=sum(t['net_return']>0 for t in active)
                result['variants'][v]={**basket(group,episode,market,v),'trades':len(active),'wins':wins,
                    'win_rate':wins/len(active) if active else None,'stop_hits':sum(t['stop_hit'] for t in active),
                    'mean_allocation_pct':float(np.mean([t['allocation'] for t in trades])*100)}
                if v in ('original','current'):
                    result['variants'][v]['checks_passed']={k:sum(r[v]['checks'][k] for r in group) for k in group[0][v]['checks']}
            result['variants']['hold']=basket(group,episode,market,'hold')
            result['variants']['cash']={'ending_balance':10000,'return_pct':0,'max_closing_drawdown_pct':0}
            result['cost_sensitivity']={str(c):{k:v for k,v in basket(group,episode,market,'current',c).items() if k!='curve'} for c in cfg['sensitivity_costs_bps']}
            targets=[r['classification_target'] for r in group];cls=['DOWN','FLAT','UP']
            probs=np.array([[r['quant']['learned_forecast']['probabilities'][c] for c in cls] for r in group]);labels=np.array([cls.index(t) for t in targets])
            # Baseline uses only labels ending before the period's training cutoff.
            from backend.historical import assemble
            series={ticker:pd.read_csv(ROOT/'data'/market/(ticker+'.csv'),index_col='Date',parse_dates=True).loc[:cfg['periods'][episode]['train_through']] for ticker in cfg['symbols'][market]}
            table,_=assemble(series,5);counts=np.bincount(table.target,minlength=3);prior=counts/counts.sum()
            from backend.historical import metrics
            result['classifier']={'model':group[0]['quant']['learned_forecast']['model'],
                'trained_through':cfg['periods'][episode]['train_through'],'selected_model':metrics(labels,probs),
                'training_frequency_baseline':metrics(labels,np.tile(prior,(len(labels),1))),
                'abstentions':sum(r['quant']['learned_forecast']['decision']=='ABSTAIN' for r in group),
                'note':'Separate price-direction panel; these probabilities do not drive the committee verdict.'}
            result['one_day_risk']={'var_breaches':sum(float(r['future_bars'][0]['Close'])/float(pd.read_csv(ROOT/'data'/market/(r['ticker']+'.csv'),index_col='Date').loc[r['date'],'Close'])-1 < r['quant']['daily_VaR_95_pct']/100 for r in group),'observations':len(group),'expected_coverage_label':'95% one-day risk threshold; actual coverage is measured on the next unseen close.'}
            result['rating_bins']=[]
            for lo,hi in [(0,49),(50,69),(70,84),(85,100)]:
                selected=[r for r in group if r['current']['trade']['traded'] and lo<=r['current']['events']['cro']['confidence']<=hi]
                result['rating_bins'].append({'range':f'{lo}-{hi}','trades':len(selected),'wins':sum(r['current']['trade']['net_return']>0 for r in selected)})
            out[episode][market]=result
    return out


def compact(row):
    q=row['quant'];v=row['current']['events']['cro']
    result={k:row[k] for k in ('id','episode','market','ticker','date','future_bars','classification_target','classification_threshold','future_price_return','elapsed_seconds')}
    result['quant']={k:q.get(k) for k in ('as_of_date','current_price','rsi_14','sma20_trend','recent_5d_return_pct','annualized_volatility','daily_VaR_95_pct','expected_shortfall_95_pct','half_kelly_allocation_pct','recent_10d_anomalies','hmm_regime','temporal_anomaly','garch_metrics')}
    forecast=q['learned_forecast'];result['forecast']={k:forecast[k] for k in ('model','training_data_end','prediction','decision','probabilities')}
    result['news']=row['news']
    for variant in ('current','original'):
        entry=row[variant];cro=entry['events']['cro']
        result[variant]={'action':cro['action'],'rating':cro['confidence'],'allocation_pct':cro['recommended_allocation_pct'],
            'summary':cro['verdict_summary'],'bull':entry['events']['bull']['bull_thesis'],'bear':entry['events']['bear']['bear_thesis'],
            'checks':entry['checks'],'trade':entry['trade']}
    result['full_evidence_sha256']=sha(ROOT/'cases'/(row['id']+'.json'))
    result['input_price_sha256']=row['data_sha256'];result['model_sha256']=row['model_sha256']
    return result


def main():
    cfg=config();plan=json.loads((ROOT/'plan.json').read_text());paths=[ROOT/'cases'/(t['id']+'.json') for t in plan['tasks']]
    missing=[p.name for p in paths if not p.exists()]
    if missing:raise RuntimeError(f'Incomplete run: {len(missing)} cases missing')
    rows=[json.loads(p.read_text()) for p in paths]
    failures=[f for p in ROOT.glob('failures-*.json') for f in json.loads(p.read_text())]
    if failures:raise RuntimeError(failures)
    for r in rows:
        assert sha(ROOT/'data'/r['market']/(r['ticker']+'.csv'))==r['data_sha256']
        assert sha(ROOT/'models'/f"{r['episode']}-{r['market']}.joblib")==r['model_sha256']
    results=summarize(rows)
    report={'config':cfg,'case_count':len(rows),'failed_cases':len(failures),'results':results,
        'sources':json.loads((PUBLIC/'news-sources.json').read_text()),
        'data_manifests':{m:json.loads((ROOT/'data'/m/'manifest.json').read_text()) for m in cfg['symbols']},
        'exploratory_news_ablation':json.loads((ROOT/'news-ablation.json').read_text()),
        'execution_assumptions':{'borrow_rate_annual':.03,'borrow_basis':'Entry notional, inclusive calendar days; no borrow-availability check','dividends':'Provider-reported ex-date cashflows, excluding entry-day dividends; no reinvestment','drawdown':'Daily closing account balances including starting capital'},
        'proof':{'original_graph_commit':__import__('subprocess').check_output(['git','rev-parse',cfg['original_graph_commit']]).decode().strip(),'benchmark_code_sha256':{p.name:sha(p) for p in Path('backend').glob('stress_*.py')},'plan_sha256':sha(ROOT/'plan.json'),'config_sha256':sha(PUBLIC/'config.json'),'code_manifest_sha256':sha(ROOT/'code-manifest.json'),'news_file_sha256':sha(ROOT/'news.json')},
        'limitations':[
            'A retrospective stress test of the present rules, not proof the system existed or predicted the crisis at the time.',
            'Selected surviving instruments and two known crisis periods; no claim of market-wide or future predictive accuracy.',
            'Gemini is not configured. Its reasoning is not tested. Bull/bear narratives use production evidence templates.',
            'News is a small curated set of official macro-event summaries. Missing company news, India-specific policy news and wording choices limit conclusions.',
            'Historical Yahoo prices are split-adjusted and may be revised. They are not an immutable point-in-time exchange feed.',
            'Hypothetical short trades assume borrow and instrument availability, with 3% annual borrowing cost. India overnight cash shorts may require a separate lending/derivative arrangement; the no-short variant avoids that assumption.',
            'Fractional mathematical positions are assumed; whole-share or lot-size constraints and small-account executability are not tested.',
            'Fixed stops can fill at worse opening prices. Daily bars do not model intraday liquidity, spread spikes or actual order fills.',
            'Closing-balance drawdown misses worse intraday losses. Cash earns zero; margin funding, taxes and FX are omitted.',
            'Calibration stays at the clean starting prior; this is not a test of online outcome learning.',
            'A score is a rule-based rating, not a measured probability. Trade wins, price classification and output consistency are separate measurements.',
            'FinBERT weights are the current downloaded model; this is a retrospective component test, not a vintage-2020 deployment reconstruction.'
        ]}
    (PUBLIC/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    public=Path('frontend/public');public.mkdir(exist_ok=True)
    payload={'report':report,'cases':[compact(r) for r in rows]}
    (public/'stress-summary.json').write_text(json.dumps(report,separators=(',',':')),encoding='utf8')
    for ep in cfg['periods']:
        for market in cfg['symbols']:
            part=[c for c in payload['cases'] if c['episode']==ep and c['market']==market]
            (public/f'stress-cases-{ep}-{market}.json').write_text(json.dumps(part,separators=(',',':')),encoding='utf8')
    ledger=[]
    for row in rows:
        for version in ('current','original'):
            t=row[version]['trade'];ledger.append({'case_id':row['id'],'version':version,**{k:row[k] for k in ('episode','market','ticker','date')},**{k:v for k,v in t.items() if k!='path'}})
    pd.DataFrame(ledger).to_csv(PUBLIC/'trade-ledger.csv',index=False)
    lines=['# AARC through COVID and the 2022 rate hikes','',f"{len(rows)} scheduled decisions, eight instruments, two episodes. Every decision is preserved; none is silently discarded.",'',
        '## Read this first','',
        'This asks what the present committee would have said using only information available at each historical date. It does not claim the committee predicted COVID or interest-rate policy. Gemini is not active in this run.', '',
        'Start with 10,000 units of money in each market separately: dollars for the US, rupees for India. Divide it equally across four instruments. Each recommendation invests only its approved fraction of that instrument\'s share; the remainder stays in cash. Profits and losses carry into later decisions.', '',
        '| Episode / market | Current committee | Original committee | Buy and hold | Always buy with same sizing | No-short committee |',
        '|---|---:|---:|---:|---:|---:|']
    for ep,markets in results.items():
        for market,r in markets.items():
            vs=r['variants'];lines.append(f"| {cfg['periods'][ep]['title']} / {market} | "+' | '.join(f"{vs[v]['ending_balance']:,.2f}" for v in ('current','original','hold','matched_long','no_short'))+' |')
    lines+=['','Cash remains 10,000. Buy-and-hold takes much more market exposure: beating it during a crash can result from holding cash rather than forecasting well. The same-sizing long comparison helps test whether the chosen direction adds value.','','## Wins, losses and risk','',
        '| Episode / market | Recommendations | Executed trades | Profitable after costs | Worst closing-balance decline | Mean allocation per recommendation |',
        '|---|---:|---:|---:|---:|---:|']
    for ep,markets in results.items():
        for market,r in markets.items():
            v=r['variants']['current'];lines.append(f"| {ep} / {market} | {r['cases']} | {v['trades']} | {v['wins']}/{v['trades']} | {v['max_closing_drawdown_pct']:.2f}% | {v['mean_allocation_pct']:.2f}% |")
    lines+=['','A no-trade recommendation earns zero; it is not counted as a successful trade. A short is a bet that the price falls, with assumed borrowing. Stops limit losses imperfectly because prices can jump past them.','','## How the model was tested','',
        '1. Download and hash daily prices, dividends and split records from Yahoo Finance. Check dates, missing values and OHLC bounds.',
        '2. Train separate price classifiers through 2019-12-31 for COVID and 2021-12-31 for 2022. Select models only within those earlier samples.',
        '3. At each scheduled close, compute the actual production GARCH, HMM, Isolation Forest and LSTM outputs from at most 252 past bars.',
        '4. Feed only already-published curated macro summaries to the actual FinBERT model. If none exist within 30 days, disclose unavailable news.',
        '5. Run the original and revised full streamed committees on identical evidence and clean isolated calibration state. Record every argument and final decision.',
        '6. Only after that decision, reveal the next five sessions to the execution evaluator. Enter at the next open, apply the fixed stop with gap handling, dividends, 30-basis-point round-trip costs, and 3% annual short borrowing.',
        '7. Track daily closing balances, retain cash between decisions, and compare with the predefined alternatives. Repeat accounting at 0 and 100 basis points of trading costs.', '',
        'The separate price predictor is shown and scored too, but the production CRO does not use its probabilities to choose the trade. It would be misleading to credit that predictor for committee returns.', '',
        '## Inspect the proof','',
        '- `/stress-test` in the website: every dated case, original/current answers, sources, five future bars and trade arithmetic.',
        '- `trade-ledger.csv`: all original and current recommendations, including zero-exposure cases.',
        '- `report.json`: exact portfolio curves, costs, classification scores, evidence checks, source manifests and hashes.',
        '- Downloadable local evidence ZIP: original CSV inputs, trained artifacts, every complete case, frozen plan, news inputs and code hashes.', '',
        '## Limits','']+['- '+x for x in report['limitations']]
    lines+=['','## Evidence consistency, separate from profit','', '| Check | Original passed | Current passed |','|---|---:|---:|']
    for key in rows[0]['current']['checks']:
        old=sum(r['original']['checks'][key] for r in rows);new=sum(r['current']['checks'][key] for r in rows)
        lines.append(f"| {key.replace('_',' ')} | {old}/{len(rows)} | {new}/{len(rows)} |")
    lines+=['','These targeted checks detect known failure modes. They do not establish universal truth, profit probabilities, or the quality of untested Gemini responses.','','## Exploratory macro-news diagnostic','',
        'The no-macro-news comparison was added after inspecting sentiment labels. It is a diagnostic, not an independently validated improvement or a production change.','','| Episode / market | Current committee | Without macro summaries |','|---|---:|---:|']
    for ep,markets in results.items():
        for market,r in markets.items():
            lines.append(f"| {ep} / {market} | {r['variants']['current']['ending_balance']:,.2f} | {r['variants']['price_only']['ending_balance']:,.2f} |")
    lines+=['','## Primary sources','']+[f"- {s['date']}: [{s['publisher']}]({s['url']}) — {s['title']}" for s in report['sources']]
    (PUBLIC/'report.md').write_text('\n'.join(lines),encoding='utf8')
    import shutil
    shutil.copy2(PUBLIC/'report.md',public/'stress-report.md')
    shutil.copy2(PUBLIC/'trade-ledger.csv',public/'stress-ledger.csv')
    print(json.dumps({ep:{m:{v:r['variants'][v]['ending_balance'] for v in ('current','original','hold','matched_long','no_short')} for m,r in markets.items()} for ep,markets in results.items()},indent=2))

if __name__=='__main__':main()
