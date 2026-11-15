"""Accounting and scheduling regressions for the stress benchmark."""
import pandas as pd
import pytest
from backend.stress_benchmark import trade_path,schedule
from backend.stress_report import equity_series,drawdown

def bars():
    return pd.DataFrame({'Open':[100,90,92,95,100],'High':[102,93,96,100,104],
        'Low':[99,88,91,94,99],'Close':[101,92,95,99,102],'Dividends':[3,1,0,0,0]},index=pd.bdate_range('2020-03-02',periods=5))

def verdict(action='LONG',allocation=10):
    return {'action':action,'recommended_allocation_pct':allocation,'stop_loss_pct':5}

def test_gap_fill_dividend_and_cost_arithmetic():
    result=trade_path(verdict(),bars())
    assert result['exit_price']==90
    assert result['dividends_per_share']==1  # Entry-day dividend belongs to prior holders.
    assert result['net_return']==pytest.approx(-.093)
    assert result['weighted_return']==pytest.approx(-.0093)
    assert result['path'][-1]['return']==result['path'][1]['return']

def test_short_dividend_and_borrow_cost():
    result=trade_path(verdict('SHORT'),bars())
    assert result['net_return']==pytest.approx(-.02-.01-.003-.03*5/365)

def test_zero_allocation_is_not_a_win_or_loss():
    result=trade_path(verdict(allocation=0),bars())
    assert not result['traded'] and result['weighted_return']==0

def test_session_schedule_has_complete_nonoverlapping_outcomes():
    frame=pd.DataFrame(index=pd.bdate_range('2022-01-03',periods=28))
    dates=schedule(frame,{'start':'2022-01-01','end':'2022-02-28','stride_sessions':5})
    positions=[frame.index.get_loc(d) for d in dates]
    assert positions==[0,5,10,15,20]
    assert all(p+5<len(frame) for p in positions)

def test_equity_uses_allocation_and_stays_cash_after_exit():
    frame=bars();future=frame.reset_index(names='Date')
    row={'date':'2020-02-28','future_bars':future.to_dict('records'),'current':{'events':{'cro':verdict()},'trade':trade_path(verdict(),frame)}}
    curve=equity_series(frame,[row],'current','2020-03-02','2020-03-06')
    assert curve.iloc[-1]==pytest.approx(9907)
    assert curve.iloc[1]==curve.iloc[-1]
    assert drawdown(curve,10000)>0
