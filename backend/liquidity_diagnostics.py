"""Daily-volume diagnostics adapted from upstream 94bd421; not observed order depth."""
import numpy as np
import pandas as pd


def daily_liquidity(frame):
    f=frame.sort_index()
    if len(f)<21:return {'available':False,'reason':'Need current bar plus 20 prior completed sessions.'}
    if f.index.has_duplicates or not np.isfinite(f[['Open','Close','Volume']].to_numpy()).all() or (f.Volume<0).any() or (f[['Open','Close']]<=0).any().any():
        return {'available':False,'reason':'Invalid or ambiguous real OHLCV data.'}
    events=[];series=[]
    for i in range(max(20,len(f)-30),len(f)):
        prior=f.Volume.iloc[i-20:i];bar=f.iloc[i];mean=float(prior.mean());std=float(prior.std(ddof=0))
        rvol=float(bar.Volume/mean) if mean>0 else None
        z=float((bar.Volume-mean)/std) if std>0 else None
        intraday=float((bar.Close/bar.Open-1)*100);close_move=float((bar.Close/f.Close.iloc[i-1]-1)*100)
        move=max([intraday,close_move],key=abs)
        thin=(rvol is not None and rvol<.7) or (z is not None and z<-.75)
        drift=abs(move)>=1.5 and thin
        item={'date':str(f.index[i].date()),'volume':float(bar.Volume),'prior_20_mean':mean,'rvol':rvol,'volume_z_score':z,'price_move_pct':move,'abnormal_drift':drift}
        series.append(item)
        if drift:events.append(item)
    latest=series[-1];rvol=latest['rvol'];z=latest['volume_z_score']
    status='LOW_VOLUME_PRICE_MOVE' if latest['abnormal_drift'] else 'LOW_RELATIVE_VOLUME' if ((rvol is not None and rvol<.65) or (z is not None and z<-.7)) else 'NO_LOW_VOLUME_FLAG'
    return {'available':True,'as_of':latest['date'],'status':status,'current_volume':latest['volume'],'rvol':rvol,'volume_z_score':z,'prior_20_mean':latest['prior_20_mean'],'recent_drift_events':events[-5:],'series':series,'affects_allocation':False,'scope':'Daily volume proxy only; does not measure bid/ask depth, identify institutional flow, or establish fill quality. Advisory only; no invented liquidity probability.'}
