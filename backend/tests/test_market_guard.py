import numpy as np
import pandas as pd
from backend.market_guard import assess
from backend.flash_research import inject, replay, verify
from backend.portfolio_research import rebalance_portfolio


def tape():
    return pd.DataFrame({'midquote':100.,'signed_volume':0.,'trading_volume':100000.,'spread':1.,'depth0_bid':10000.,'depth0_ask':10000.},index=pd.date_range('2010-05-06 13:50:00',periods=300,freq='5s'))


def test_directional_volume_and_unsigned_control():
    f=tape();end=pd.Timestamp('2010-05-06 14:00:55')
    bearish=inject(f,'bearish',5);bullish=inject(f,'bullish',5);unsigned=inject(f,'unsigned',5)
    assert assess(bearish.loc[:end])['exposure_multiplier']==.5
    assert assess(bullish.loc[:end])['exposure_multiplier']==1.
    assert assess(unsigned.loc[:end])['exposure_multiplier']==1.
    assert assess(bullish.loc[:end],'SHORT')['exposure_multiplier']==.5
    pd.testing.assert_series_equal(bearish.midquote,f.midquote)
    pd.testing.assert_frame_equal(bearish.loc[:'2010-05-06 13:59:55'],f.loc[:'2010-05-06 13:59:55'])


def test_future_mutation_cannot_change_past_guard():
    f=tape();cut=f.index[100];before=assess(f.loc[:cut]);f.loc[f.index>cut,'midquote']=1
    assert before==assess(f.loc[:cut])


def test_latency_and_cash_conservation():
    f=inject(tape(),'bearish',5);r=replay(f,latency=30);verify(r)
    first=pd.Timestamp(r['decisions'][0]['time']);sell=next(x for x in r['ledger'] if x['side']=='SELL')
    assert pd.Timestamp(sell['time'])>=first+pd.Timedelta(seconds=30)
    assert r['ending_balance']<10000 # Flat price still incurs spread and fees.


def test_portfolio_turnover_fee_and_no_negative_cash():
    cash,shares,fee=rebalance_portfolio(10000.,np.zeros(2),np.array([100.,50.]),np.array([.5,.5]),.0015)
    assert cash>=0 and np.isclose(shares@np.array([100.,50.])+fee,10000.)
    cash2,shares2,fee2=rebalance_portfolio(cash,shares,np.array([100.,50.]),np.array([.5,.5]),.0015)
    assert np.isclose(fee2,0) and np.allclose(shares,shares2)
