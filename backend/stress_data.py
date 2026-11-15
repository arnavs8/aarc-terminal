"""Acquire and freeze the stress-test inputs. No live app state is written."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import pandas as pd
import yfinance as yf

ROOT=Path('benchmark-runs/stress')
CONFIG=Path('benchmarks/stress/config.json')

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate_bars(frame):
    if frame.empty or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError('Empty, duplicate or unordered price dates')
    cols=['Open','High','Low','Close','Volume']
    if frame[cols].isna().any().any() or (frame[['Open','High','Low','Close']]<=0).any().any():
        raise ValueError('Missing or non-positive OHLC data')
    if (frame.High < frame[['Open','Close','Low']].max(axis=1)-1e-7).any() or (frame.Low > frame[['Open','Close','High']].min(axis=1)+1e-7).any():
        raise ValueError('Invalid OHLC bounds')
    if (frame.Volume < 0).any(): raise ValueError('Negative volume')


def download():
    cfg=json.loads(CONFIG.read_text());ROOT.mkdir(parents=True,exist_ok=True)
    for market,symbols in cfg['symbols'].items():
        folder=ROOT/'data'/market;folder.mkdir(parents=True,exist_ok=True);records=[]
        for ticker in symbols:
            p=folder/(ticker+'.csv')
            if not p.exists():
                raw=yf.download(ticker,start=cfg['price_start'],end=cfg['price_end_exclusive'],auto_adjust=False,actions=True,progress=False,threads=False)
                if isinstance(raw.columns,pd.MultiIndex):raw.columns=raw.columns.get_level_values(0)
                raw.index=pd.DatetimeIndex(raw.index).tz_localize(None);raw.index.name='Date'
                validate_bars(raw)
                raw.to_csv(p,float_format='%.10g')
            data=pd.read_csv(p,index_col='Date',parse_dates=True);validate_bars(data)
            record={'symbol':ticker,'daily_sha256':digest(p),'rows':len(data),'start':str(data.index.min().date()),'end':str(data.index.max().date()),
                'url':f'https://finance.yahoo.com/quote/{ticker}/history/','provider':'Yahoo Finance via yfinance',
                'requested_start':cfg['price_start'],'requested_end_exclusive':cfg['price_end_exclusive'],
                'price_basis':'Provider OHLC with auto_adjust=False: split-adjusted as delivered, not adjusted for cash dividends. Historical revisions remain possible.'}
            records.append(record);print('Downloaded',ticker,len(data),flush=True)
        manifest={'market':market,'source':'Yahoo Finance daily history (frozen download)','retrieved_utc':datetime.now(timezone.utc).isoformat(),'yfinance_version':yf.__version__,'records':records}
        (folder/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    proof={'config_sha256':digest(CONFIG),'news_sources_sha256':digest('benchmarks/stress/news-sources.json'),'python':sys.version,'platform':platform.platform(),'downloaded_utc':datetime.now(timezone.utc).isoformat()}
    (ROOT/'acquisition.json').write_text(json.dumps(proof,indent=2),encoding='utf8')


def train():
    from types import SimpleNamespace
    from backend.historical import train as fit
    cfg=json.loads(CONFIG.read_text());out=ROOT/'models';out.mkdir(parents=True,exist_ok=True)
    for episode,period in cfg['periods'].items():
        for market in cfg['symbols']:
            model=out/f'{episode}-{market}.joblib'
            if not model.exists():
                fit(SimpleNamespace(data=ROOT/'data'/market,through=period['train_through'],horizon=5,output=model))
            meta=json.loads(model.with_suffix('.json').read_text())
            assert pd.Timestamp(meta['training_data_end']) < pd.Timestamp(period['start'])
            print('Trained',episode,market,meta['model_name'],'through',meta['training_data_end'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['download','train']);args=p.parse_args()
    (download if args.command=='download' else train)()
