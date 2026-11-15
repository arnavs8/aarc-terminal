"""Causal intraday position guard. Order-flow pressure is not a profit forecast."""
import math
import numpy as np


def assess(history, direction='LONG'):
    if direction not in ('LONG','SHORT'):raise ValueError('Invalid position direction')
    if len(history)<72:return {'available':False,'exposure_multiplier':1.,'reason':'Need six minutes of five-second observations.'}
    x=history.iloc[-72:];recent=x.iloc[-12:];reference=x.iloc[:-12]
    required=['midquote','signed_volume','trading_volume','spread','depth0_bid','depth0_ask']
    if not np.isfinite(x[required].to_numpy()).all() or (x.midquote<=0).any() or (x.trading_volume<0).any():
        return {'available':False,'exposure_multiplier':0.,'reason':'Invalid intraday evidence; new risk blocked.'}
    volume=float(recent.trading_volume.sum());base=float(reference.trading_volume.mean()*12)
    ratio=volume/max(base,1.)
    # Publisher extraction code: signed_volume is shares; trading_volume is USD.
    imbalance=float(np.clip((recent.signed_volume*recent.midquote).sum()/max(volume,1.),-1,1))
    change=float(recent.midquote.iloc[-1]/reference.midquote.iloc[-1]-1)
    spread=float(recent.spread.max());side='depth0_bid' if direction=='LONG' else 'depth0_ask'
    depth=float(recent[side].iloc[-1]/max(reference[side].median(),1.))
    adverse=imbalance if direction=='LONG' else -imbalance
    move=change if direction=='LONG' else -change
    reasons=[];scale=1.
    if spread>=10:reasons.append('Spread reached 10 basis points; liquidity is expensive.');scale=0.
    if move<=-.01:reasons.append('Price moved at least 1% against the position in one minute.');scale=0.
    if depth<=.25:reasons.append('Exit-side visible depth fell below 25% of the prior five-minute median.');scale=min(scale,.5)
    if ratio>=3 and adverse<=-.6:reasons.append('At least 3x volume with strong adverse signed order flow.');scale=min(scale,.5)
    return {'available':True,'exposure_multiplier':scale,'volume_ratio':ratio,'signed_imbalance':imbalance,'one_minute_return':change,'max_spread_bps':spread,'depth_ratio':depth,'reason':' '.join(reasons) or 'No risk reduction triggered; favorable flow alone never increases exposure.','as_of':str(history.index[-1]),'kind':'risk_rule_not_probability'}
