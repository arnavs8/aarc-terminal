"""Replay actual quant/bull/bear/CRO stages around May 2010 with no future tape."""
import contextlib,io,json,os,tempfile
from pathlib import Path
import numpy as np,pandas as pd,torch
from backend import graph
from backend.tools.quant_math import analyze_ticker_quant
from backend.committee_benchmark import checks
from backend.stress_benchmark import trade_path

def main():
    torch.set_num_threads(2)
    frame=pd.read_csv('benchmark-runs/strategy/SPY.csv',index_col='Date',parse_dates=True)
    output=[]
    with tempfile.TemporaryDirectory() as temp:
        os.environ['AARC_STATE_DIR']=temp
        graph._get_gemini_client=lambda:None
        news={'headlines':[],'aggregate_sentiment_score':None,'news_available':False,'news_error':'No verified pre-decision archived company-news tape. Groq is not credited with these historical decisions.'}
        for day in ['2010-04-30','2010-05-03','2010-05-04','2010-05-05','2010-05-06','2010-05-07']:
            date=pd.Timestamp(day);past=frame.loc[:date].tail(252).copy();future=frame.loc[frame.index>date].head(5)
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                np.random.seed(42);torch.manual_seed(42)
                q=analyze_ticker_quant('SPY',day,price_data=past)
                bull=graph.run_bull_node(q,news);bear=graph.run_bear_node(q,news,bull);v=graph.run_cro_node(q,news,bull,bear)
            assert q['time_machine_actual_return_pct'] is None
            events={'bull':{'bull_thesis':bull},'bear':{'bear_thesis':bear},'cro':v}
            output.append({'date':day,'quant':q,'news':news,'events':events,'checks':checks(q,events),'trade':trade_path(v,future),'future':json.loads(future.reset_index().to_json(orient='records',date_format='iso'))})
            print(day,v['action'],v['recommended_allocation_pct'],flush=True)
    Path('benchmarks/flash-crash/committee.json').write_text(json.dumps({'cases':output,'scope':'Six overlapping daily decision cases, not an additive or annual portfolio. Current code with only pre-signal prices; models trained after signal are unavailable. Daily OHLC stops cannot reproduce flash-crash fills. Intraday execution tests are separate.'},indent=2))

if __name__=='__main__':main()
