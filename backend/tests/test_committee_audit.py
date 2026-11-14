"""Specific output-integrity regressions, separate from historical win-rate evidence."""
import asyncio
from copy import deepcopy
import pandas as pd
import pytest
from backend.committee_audit import finalize_verdict, grounded_arguments
from backend.committee_benchmark import execution
from backend import graph

Q = dict(ticker='TEST', current_price=100, rsi_14=55, daily_VaR_95_pct=-2,
         half_kelly_allocation_pct=10, close_gt_sma20=True, sma20_trend='BULLISH',
         as_of_date='2020-03-02', is_historical_replay=True,
         hmm_regime={'kelly_scale_factor': .2, 'active_state': 1, 'regime_name': 'HIGH_VOL_PANIC', 'probabilities': {'P_Panic': .9}})
P = dict(action='LONG', recommended_allocation_pct=4.5, stop_loss_pct=3, confidence=88)
N = dict(aggregate_sentiment_score=None, headlines=[])


def test_floor_cannot_exceed_scaled_cap():
    v = finalize_verdict(P, Q, N)
    assert v['recommended_allocation_pct'] == 2
    assert 'final allocation 2.0%' in v['verdict_summary']
    assert v['calibrated_probabilities'] is None


def test_zero_exposure_does_not_approve_direction():
    v = finalize_verdict(dict(P, regime_gating={'is_suppressed': True}), Q, N)
    assert v['action'] == 'REJECT_TRADE'
    assert v['recommended_allocation_pct'] == 0
    assert 'No position approved' in v['verdict_summary']


def test_below_trend_without_news_cannot_default_long():
    v = finalize_verdict(P, dict(Q, close_gt_sma20=False), N)
    assert v['action'] == 'REJECT_TRADE'


@pytest.mark.parametrize('value', [None, float('nan'), float('inf')])
def test_invalid_risk_evidence_blocks_exposure(value):
    assert finalize_verdict(P, dict(Q, daily_VaR_95_pct=value), N)['recommended_allocation_pct'] == 0


def test_arguments_distinguish_observation_and_inference():
    bull, bear = grounded_arguments(Q, N)
    assert 'News evidence unavailable' in bull
    assert 'do not identify institutional' in bear
    assert 'hypothesis' in bull


def test_gap_through_stop_is_not_filled_at_stop_price():
    bars = pd.DataFrame([{'Open':100, 'High':101, 'Low':99, 'Close':100},
                         {'Open':90, 'High':92, 'Low':88, 'Close':91}])
    v = dict(P, recommended_allocation_pct=10, stop_loss_pct=3)
    outcome = execution(v, bars)
    assert outcome['net_trade_return'] == pytest.approx(-.103)
    assert outcome['weighted_return'] == pytest.approx(-.0103)
    assert execution(dict(v, recommended_allocation_pct=0), bars)['weighted_return'] == 0


def test_replay_does_not_write_live_prediction_log_and_stream_matches_direct(monkeypatch, tmp_path):
    from backend.calibration.rolling_tracker import RollingAccuracyTracker
    from backend.calibration.bayesian_calibrator import BetaBinomialConfidenceCalibrator
    class Tracker(RollingAccuracyTracker):
        def __init__(self): super().__init__(str(tmp_path/'rolling.json'))
        def log_prediction(self, **kwargs): pytest.fail('Historical replay wrote a live prediction')
    monkeypatch.setattr(graph, 'RollingAccuracyTracker', Tracker)
    monkeypatch.setattr(graph, 'BetaBinomialConfidenceCalibrator', lambda: BetaBinomialConfidenceCalibrator(storage_path=str(tmp_path/'bayes.json')))
    monkeypatch.setattr(graph, '_get_gemini_client', lambda: None)
    monkeypatch.setattr(graph, 'analyze_ticker_quant', lambda *args: deepcopy(Q))
    monkeypatch.setattr(graph, 'analyze_ticker_sentiment', lambda *args: N)
    bull = graph.run_bull_node(Q,N); bear = graph.run_bear_node(Q,N,bull)
    direct = graph.run_cro_node(Q,N,bull,bear)
    async def collect(): return [e async for e in graph.stream_committee('TEST', '2020-03-02')]
    events = asyncio.run(collect())
    assert events[0]['data']['regime']['label'] == 'HIGH_VOL_PANIC'
    assert events[0]['data']['regime']['prob_high_vol'] == .9
    streamed = events[-1]['data']
    for key in ('action','recommended_allocation_pct','verdict_summary','evidence_audit','calibrated_probabilities'):
        assert streamed[key] == direct[key]

def test_live_log_uses_only_final_approved_allocation(monkeypatch):
    calls = []
    class Tracker:
        def log_prediction(self, **kwargs): calls.append(kwargs)
    monkeypatch.setattr(graph, 'RollingAccuracyTracker', Tracker)
    monkeypatch.setattr(graph, '_run_cro_proposal', lambda *args: dict(P))
    q = dict(Q, is_historical_replay=False, close_gt_sma20=False)
    graph.run_cro_node(q, N, '', '')
    assert calls == []
    q['close_gt_sma20'] = True
    v = graph.run_cro_node(q, N, '', '')
    assert len(calls) == 1
    assert calls[0]['action'] == v['action']
    assert calls[0]['date_str'] == q['as_of_date']
