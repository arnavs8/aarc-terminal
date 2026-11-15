"""Evidence rendering and final risk checks shared by both committee API routes.

An audit validates consistency with supplied evidence, not external factual truth.
Committee scores remain heuristics until evaluated on independent resolved outcomes.
"""
import math


def finite(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (ValueError, TypeError):
        return default


def grounded_arguments(q, news):
    trend = q.get('sma20_trend', 'unavailable')
    momentum = q.get('recent_5d_return_pct', 'unavailable')
    rsi = q.get('rsi_14', 'unavailable')
    var = q.get('daily_VaR_95_pct', 'unavailable')
    vol = q.get('annualized_volatility', 'unavailable')
    anomalies = q.get('recent_10d_anomalies', 'unavailable')
    score = news.get('aggregate_sentiment_score')
    news_text = (('Headlines reviewed, but company-specific direction remains uncertain.' if news.get('interpretation_available')
                  else 'News evidence unavailable; sentiment cannot support a trade.') if score is None
                 else f'Headline sentiment score is {score}; this measures wording, not event truth or future returns.')
    bull = (f'• Observed SMA20 trend: {trend}; five-session return: {momentum}%. A bullish continuation is a hypothesis.\n'
            f'• RSI-14: {rsi}. RSI alone does not establish upside or a reversal.\n'
            f'• {news_text}')
    bear = (f'• Estimated daily 95% VaR: {var}%; annualized volatility: {vol}%. Losses can exceed this estimate.\n'
            f'• Detected anomalies over ten sessions: {anomalies}. These do not identify institutional buying or selling.\n'
            '• Trend reversal, gaps and execution costs can invalidate a directional trade; a stop is not a guaranteed exit price.')
    return bull, bear


def finalize_verdict(proposal, q, news):
    result = dict(proposal)
    issues = []
    action = result.get('action', 'REJECT_TRADE')
    cap = max(0.0, min(100.0, finite(q.get('half_kelly_allocation_pct')) *
                      max(0.0, min(1.0, finite(q.get('hmm_regime', {}).get('kelly_scale_factor', 1), 1)))))
    allocation = finite(result.get('recommended_allocation_pct'))
    if allocation < 0 or allocation > cap:
        issues.append('Allocation clipped to the regime-adjusted risk cap.')
    allocation = max(0.0, min(cap, allocation))
    guard = q.get('market_guard')
    if guard is not None:
        scale = max(0., min(1., finite(guard.get('exposure_multiplier'), 0.)))
        allocation *= scale
        result['market_guard'] = guard
        if scale < 1:
            issues.append('Intraday risk guard reduced exposure: ' + str(guard.get('reason', 'invalid evidence')))
    # A default LONG without a supporting trend or observed positive news is not evidence.
    sentiment = finite(news.get('aggregate_sentiment_score'))
    unsupported_long = action == 'LONG' and not q.get('close_gt_sma20', False) and sentiment <= 0
    missing = [k for k in ('current_price', 'rsi_14', 'daily_VaR_95_pct', 'half_kelly_allocation_pct')
               if q.get(k) is None or not math.isfinite(finite(q.get(k), float('nan')))]
    if unsupported_long:
        issues.append('No supporting long signal: below SMA20 and no positive news evidence.')
    if missing:
        issues.append('Missing or invalid risk inputs: ' + ', '.join(missing))
    if result.get('regime_gating', {}).get('is_suppressed'):
        issues.append('Regime risk gate suppresses exposure.')
        allocation = 0
    if unsupported_long or missing or action not in ('LONG', 'SHORT'):
        allocation = 0
    stop = max(0.0, finite(result.get('stop_loss_pct')))
    if stop <= 0:
        allocation = 0
        issues.append('No valid stop distance supplied.')
    result['proposed_action'] = action
    allocation = round(allocation, 2)
    if allocation == 0:
        action = 'REJECT_TRADE'
    result.update(action=action, verdict=action, recommended_allocation_pct=allocation,
                  stop_loss_pct=stop, confidence_kind='heuristic_score_not_probability',
                  calibrated_probabilities=None, conformal_prediction_set=None)
    result['verdict_summary'] = (
        f'No position approved (allocation 0%). Proposed direction: {result["proposed_action"]}. '
        if allocation == 0 else
        f'{action} scenario: final allocation {allocation}%, stop distance {stop}%. '
    ) + (f'As of {q.get("as_of_date", "supplied snapshot")}: close {q.get("current_price")}, '
         f'RSI-14 {q.get("rsi_14")}, SMA20 trend {q.get("sma20_trend")}. '
         'The committee score is a heuristic, not a measured probability of success. ')
    if news.get('aggregate_sentiment_score') is None:
        result['verdict_summary'] += ('Headlines were reviewed; no directional company-specific evidence was established. '
                                     if news.get('interpretation_available') else 'Historical/news evidence is unavailable. ')
    if issues:
        result['verdict_summary'] += ' '.join(issues)
    result['evidence_audit'] = {
        'scope': 'Consistency with supplied metrics; external news truth is not verified.',
        'corrections': issues, 'allocation_cap_pct': round(cap, 2),
        'news_available': news.get('aggregate_sentiment_score') is not None,
        'narrative_mode': 'evidence_templates',
        'future_performance_verified': False,
        'evidence': [{'field': k, 'value': q.get(k), 'source': 'quant_snapshot'} for k in
                     ('as_of_date', 'current_price', 'rsi_14', 'sma20_trend', 'daily_VaR_95_pct')],
    }
    # No invented attribution percentages: these are observed inputs, not causal impacts.
    result['top_drivers'] = [{'feature': k, 'value': q.get(k), 'impact': None} for k in
                             ('close_gt_sma20', 'rsi_14', 'daily_VaR_95_pct', 'recent_10d_anomalies')]
    return result
