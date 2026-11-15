"""Regression cases adapted from upstream's CRO alignment fix, with our risk caps."""
import json
from types import SimpleNamespace
import pandas as pd
import pytest
from backend import graph
from backend.tools.quant_math import chronological_bars


@pytest.mark.parametrize('overrides,news,model_action,expected',[
    ({'rsi_14':80},-.2,'REJECT_TRADE','SHORT'),
    ({'close_gt_sma20':False},-.2,'REJECT_TRADE','SHORT'),
    ({},0.,'SHORT','LONG'),
    ({'close_gt_sma20':False},None,'LONG','REJECT_TRADE'),
    ({'recent_10d_anomalies':3,'daily_VaR_95_pct':-3.1},.2,'LONG','REJECT_TRADE'),
])
def test_conflicting_model_proposal_cannot_override_rules(monkeypatch,overrides,news,model_action,expected):
    class Tracker:
        def calibrate_confidence(self,confidence,ticker):return confidence,{'allocation_multiplier':1.}
    class Cal:
        def calibrate_score(self,raw_ai_confidence,**kw):return raw_ai_confidence,{'bayesian_allocation_multiplier':1.}
    monkeypatch.setattr(graph,'RollingAccuracyTracker',Tracker);monkeypatch.setattr(graph,'BetaBinomialConfidenceCalibrator',Cal)
    monkeypatch.setattr(graph,'_get_gemini_client',lambda:object())
    payload={'action':model_action,'confidence':80,'recommended_allocation_pct':8,'stop_loss_pct':3,'kill_switch_reason':'Generated reason','verdict_summary':'Conflicting narrative'}
    monkeypatch.setattr(graph,'_call_gemini',lambda *a,**kw:SimpleNamespace(text=json.dumps(payload)))
    q={'ticker':'TEST','current_price':100,'rsi_14':55,'half_kelly_allocation_pct':10,'daily_VaR_95_pct':-2,'close_gt_sma20':True,'suggested_stop_loss_atr_pct':3,'is_historical_replay':True,**overrides}
    v=graph.run_cro_node(q,{'aggregate_sentiment_score':news},'bull','bear')
    assert v['action']==expected and v['recommended_allocation_pct']<=10
    assert 'Conflicting narrative' not in v['verdict_summary']


def test_history_order_and_duplicates():
    f=pd.DataFrame({'Close':[102,100,101]},index=pd.to_datetime(['2020-01-03','2020-01-01','2020-01-02']))
    sorted_bars=chronological_bars(f)
    assert sorted_bars.Close.tolist()==[100,101,102]
    assert sorted_bars.loc[:'2020-01-02'].Close.tolist()==[100,101]
    with pytest.raises(ValueError,match='unique'):chronological_bars(pd.concat([f,f.iloc[:1]]))


def test_upstream_target_candidate_records_actual_early_exit():
    from backend.upstream_comparison import target_trade
    future=pd.DataFrame({'Open':[100,102,110],'High':[102,107,112],'Low':[99,101,109],'Close':[101,104,111],'Dividends':[2,1,5]},index=pd.date_range('2020-01-02',periods=3))
    verdict={'action':'LONG','recommended_allocation_pct':10,'stop_loss_pct':3}
    result=target_trade(verdict,future)
    assert result['exit_date']=='2020-01-03' and result['exit_price']==106
    assert result['dividends_per_share']==1
    assert result['gross_return']==pytest.approx(.07)
    assert result['net_return']==pytest.approx(.067)
    assert result['weighted_return']==pytest.approx(.0067)
    assert result['path'][-1]['return']==result['path'][-2]['return']
    from backend.stress_benchmark import trade_path
    executable=trade_path({**verdict,'execution_plan':{'take_profit_r_multiple':2}},future)
    assert executable['net_return']==pytest.approx(result['net_return'])
    assert executable['target_hit'] and executable['exit_price']==106


def test_ambiguous_target_stop_bar_favors_stop():
    from backend.stress_benchmark import trade_path
    f=pd.DataFrame({'Open':[100],'High':[108],'Low':[95],'Close':[104]},index=pd.to_datetime(['2020-01-02']))
    v={'action':'LONG','recommended_allocation_pct':10,'stop_loss_pct':3,'execution_plan':{'take_profit_r_multiple':2}}
    r=trade_path(v,f)
    assert r['exit_price']==97 and r['stop_hit'] and not r['target_hit']


def test_volume_diagnostic_uses_prior_window_preserves_zero_and_makes_no_depth_claim():
    from backend.liquidity_diagnostics import daily_liquidity
    f=pd.DataFrame({'Open':[100.]*21,'Close':[100.]*20+[103.],'Volume':[100.]*20+[0.]},index=pd.date_range('2020-01-01',periods=21))
    r=daily_liquidity(f)
    assert r['prior_20_mean']==100 and r['rvol']==0 and r['current_volume']==0
    assert r['volume_z_score'] is None # Zero baseline variance is undefined, not fabricated.
    assert r['status']=='LOW_VOLUME_PRICE_MOVE' and not r['affects_allocation']
    assert not daily_liquidity(f.iloc[:20])['available']
    f.iloc[-1,f.columns.get_loc('Volume')]=-1
    assert not daily_liquidity(f)['available']
