"""Run a trusted repository revision in an isolated process against frozen inputs."""
import argparse
import contextlib
from copy import deepcopy
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repository',required=True);parser.add_argument('--input-root',required=True);parser.add_argument('--output',required=True);parser.add_argument('--recompute-quant',action='store_true');parser.add_argument('--refresh',action='store_true');parser.add_argument('--shards',type=int,default=1);parser.add_argument('--shard',type=int,default=0);args=parser.parse_args()
    repo=Path(args.repository).resolve();root=Path(args.input_root).resolve();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(repo))
    import numpy as np
    import pandas as pd
    import torch
    from backend import graph
    from backend.tools import quant_math
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    assert Path(graph.__file__).resolve().is_relative_to(repo)
    torch.set_num_threads(1)
    graph._get_gemini_client=lambda:None
    graph.RollingAccuracyTracker.log_prediction=lambda *a,**kw:None
    source_hashes={p.relative_to(repo).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in (repo/'backend').rglob('*.py') if not any(x in p.parts for x in ('__pycache__','data_cache','.venv','tests'))}
    manifest={'repository':str(repo),'code_sha256':source_hashes,'recompute_quant':args.recompute_quant,'scope':'No external LLM calls; empty isolated calibration state; fixed dated macro-news interpretation shared by all variants. Full local quant computation for upstream, stored pre-signal quant for fork.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp);os.environ['AARC_STATE_DIR']=tmp
        for i,path in enumerate(sorted((root/'cases').glob('*.json'))):
            if i%args.shards!=args.shard:continue
            started=time.perf_counter();r=json.loads(path.read_text(encoding='utf8'));target=out/path.name
            if target.exists() and not args.refresh:continue
            q=deepcopy(r['quant']);news=deepcopy(r['news'])
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                if args.recompute_quant:
                    frame=pd.read_csv(root/'data'/r['market']/(r['ticker']+'.csv'),index_col='Date',parse_dates=True)
                    past=frame.loc[:r['date']].tail(252).copy();np.random.seed(42);torch.manual_seed(42)
                    if 'price_data' in inspect.signature(quant_math.analyze_ticker_quant).parameters:
                        q=quant_math.analyze_ticker_quant(r['ticker'],r['date'],price_data=past)
                    else:
                        quant_math.yf.download=lambda *a,**kw:past.copy()
                        q=quant_math.analyze_ticker_quant(r['ticker'],r['date'])
                # Metadata is attached only for isolation/tracking, never fabricated outcomes.
                q['ticker']=r['ticker'];q['as_of_date']=r['date'];q['is_historical_replay']=True
                bull=graph.run_bull_node(q,news);bear=graph.run_bear_node(q,news,bull);cro=graph.run_cro_node(q,news,bull,bear)
            row={'id':r['id'],'date':r['date'],'ticker':r['ticker'],'episode':r['episode'],'market':r['market'],'input_case_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'quant':q,'news':news,'events':{'bull':{'bull_thesis':bull},'bear':{'bear_thesis':bear},'cro':cro},'elapsed_seconds':time.perf_counter()-started}
            pending=target.with_suffix('.pending');pending.write_text(json.dumps(row),encoding='utf8');pending.replace(target)
            print(i+1,r['id'],cro['action'],round(row['elapsed_seconds'],2),flush=True)
        os.chdir(repo)


if __name__=='__main__':main()
