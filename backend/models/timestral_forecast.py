"""Pinned, opt-in Timestral challenger. Quantiles are not calibrated profit odds."""
from functools import lru_cache
from threading import Lock
import numpy as np

MODEL_ID = 'mistralai/Timestral-28M'
REVISION = '49b26409ac87b5935539a5d8ffe7f397b720db47'
_LOCK = Lock()


@lru_cache(maxsize=1)
def forecaster():
    from timestral import TimestralForecaster
    return TimestralForecaster.from_pretrained(MODEL_ID, revision=REVISION, device='cpu')


def forecast_batch(contexts, horizon=5):
    values=np.array(contexts,dtype=np.float32,copy=True)
    if values.ndim!=2 or values.shape[1]<200 or not np.isfinite(values).all() or (values<=0).any():
        raise ValueError('Require finite positive past closing prices, at least 200 sessions')
    if horizon!=5:raise ValueError('This reviewed challenger uses a five-session horizon')
    with _LOCK:
        result=forecaster().forecast(values,prediction_length=horizon)
    arrays={str(q):result.quantile(q).detach().cpu().numpy()[:,0,:] for q in (.1,.5,.9)}
    if any(not np.isfinite(a).all() or (a<=0).any() for a in arrays.values()):
        raise ValueError('Invalid forecast values')
    if (arrays['0.1']>arrays['0.5']).any() or (arrays['0.5']>arrays['0.9']).any():
        raise ValueError('Crossed forecast quantiles')
    return [{'available':True,'model':MODEL_ID,'revision':REVISION,'horizon_sessions':horizon,
             'last_close':float(values[i,-1]),'quantile_prices':{k:a[i].tolist() for k,a in arrays.items()},
             'median_return':float(arrays['0.5'][i,-1]/values[i,-1]-1),
             'scope':'Current pretrained model replay; pretraining overlap with historical markets is unknown. Quantiles are not profit probabilities.'}
            for i in range(len(values))]


def decide(history, forecast, use_forecast=True, cost_hurdle=.003, annual_risk_target=.20):
    """Frozen long-only policy: 200-session trend, 60-session vol sizing, forecast gate.

    Weight is a fraction of ONE instrument's sleeve, not of the entire portfolio.
    Four sleeves receive equal initial capital; no leverage or short borrowing.
    """
    close=history.Close.astype(float)
    if len(close)<252 or not np.isfinite(close).all() or (close<=0).any():
        raise ValueError('Insufficient or invalid causal price history')
    vol=float(close.pct_change().tail(60).std()*np.sqrt(252))
    trend=bool(close.iloc[-1]>close.tail(200).mean())
    expected=float(forecast['median_return']) if use_forecast else None
    if use_forecast and (not forecast.get('available') or not np.isfinite(expected)):
        raise ValueError('A forecast-gated decision requires a valid forecast')
    approved=trend and (not use_forecast or expected>cost_hurdle)
    weight=min(1.,annual_risk_target/max(vol,.01)) if approved else 0.
    return {'action':'LONG' if weight>0 else 'REJECT_TRADE','sleeve_weight':weight,
            'as_of_date':str(history.index[-1].date()),'current_price':float(close.iloc[-1]),
            'trend_above_200d':trend,'annualized_volatility':vol,'median_return':expected,
            'cost_hurdle':cost_hurdle,'risk_target':annual_risk_target,'leverage':False,
            'summary':('Approve a long position' if approved else 'Stay in cash')+
                f'; allocate {weight*100:.2f}% of this instrument sleeve. '+
                (f'Timestral median five-session price change: {expected*100:.2f}%; hurdle: {cost_hurdle*100:.2f}%. ' if use_forecast else 'Trend/volatility baseline; no forecast used. ')+
                'This is an experimental scenario, not a measured probability or a guaranteed return.'}
