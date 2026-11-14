import os
import json
import asyncio
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Ensure .env is loaded before anything else
ROOT_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ROOT_ENV_PATH, override=True)

from backend.tools.quant_math import analyze_ticker_quant
from backend.tools.sentiment_nlp import analyze_ticker_sentiment
from backend.calibration.rolling_tracker import RollingAccuracyTracker
from backend.calibration.bayesian_calibrator import BetaBinomialConfidenceCalibrator

MODEL_CANDIDATES = ["gemini-3.8-flash", "gemini-2.5-flash"]


class CROVerdict(BaseModel):
    action: Literal["LONG", "SHORT", "REJECT_TRADE"] = Field(
        description="Trade action: LONG, SHORT, or REJECT_TRADE"
    )
    verdict: Optional[Literal["LONG", "SHORT", "REJECT_TRADE"]] = Field(
        default=None,
        description="Alias for action: LONG, SHORT, or REJECT_TRADE"
    )
    confidence: int = Field(
        description="Confidence score between 0 and 100",
        ge=0,
        le=100
    )
    recommended_allocation_pct: float = Field(
        description="Recommended portfolio allocation percentage, capped at half_kelly_allocation_pct"
    )
    stop_loss_pct: float = Field(
        description="Suggested stop loss percentage"
    )
    kill_switch_reason: str = Field(
        description="Primary risk or reason why the trade could be rejected or killed"
    )
    verdict_summary: str = Field(
        description="Summary of the CRO final decision synthesizing bull and bear cases"
    )


def _get_gemini_client():
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE":
        print("[GEMINI] Warning: GOOGLE_API_KEY is unset or default placeholder.")
        return None
    try:
        from google import genai
        # Set 35s timeout to allow for network transmission without hanging indefinitely
        return genai.Client(api_key=api_key, http_options={"timeout": 35.0})
    except Exception as e:
        print(f"[GEMINI INIT ERROR]: {e}")
        traceback.print_exc()
        return None


def _call_gemini(client, prompt: str, schema=None):
    """
    Tries gemini-3.8-flash first (since gemini-2.5-flash is retired/404 for new users),
    then falls back to gemini-2.5-flash. Prints all exceptions to logs.
    """
    if not client:
        return None

    for model_name in MODEL_CANDIDATES:
        try:
            if schema:
                from google.genai import types
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=schema,
                        temperature=0.1,
                    )
                )
            else:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
            if response and response.text:
                return response
        except Exception as e:
            print(f"[GEMINI CALL ERROR with {model_name}]: {e}")
            traceback.print_exc()
    return None


def _sentiment_inputs(news_data: Dict[str, Any]) -> tuple:
    """Return (numeric sentiment for rules, display label for generated text)."""
    value = news_data.get("aggregate_sentiment_score")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value), f"{value}"
    return 0.0, "UNAVAILABLE"


def run_bull_node(quant_data, news_data):
    from backend.committee_audit import grounded_arguments
    return grounded_arguments(quant_data, news_data)[0]


def run_bear_node(quant_data, news_data, bull_thesis):
    from backend.committee_audit import grounded_arguments
    return grounded_arguments(quant_data, news_data)[1]


def run_cro_node(quant_data, news_data, bull_thesis, bear_thesis):
    from backend.committee_audit import finalize_verdict
    proposal = _run_cro_proposal(quant_data, news_data, bull_thesis, bear_thesis)
    result = finalize_verdict(proposal, quant_data, news_data)
    # Track only the final approved decision, not a proposal later rejected by audit.
    if not quant_data.get("is_historical_replay", False) and result["recommended_allocation_pct"] > 0:
        import datetime
        RollingAccuracyTracker().log_prediction(
            ticker=quant_data.get("ticker", "TICKER"), action=result["action"],
            raw_confidence=result.get("raw_confidence", result["confidence"]),
            current_price=quant_data.get("current_price", 0),
            date_str=quant_data.get("as_of_date", datetime.date.today().isoformat()),
            garch_var_pct=quant_data.get("daily_VaR_95_pct", 0),
            stop_loss_pct=result["stop_loss_pct"],
            expected_shortfall_pct=quant_data.get("expected_shortfall_95_pct"))
    return result


def _run_cro_proposal(
    quant_data: Dict[str, Any],
    news_data: Dict[str, Any],
    bull_thesis: str,
    bear_thesis: str
) -> Dict[str, Any]:
    """
    CRO node: Renders definitive arbitration verdict:
    - Output LONG (with Half-Kelly allocation) when Close > SMA20 or (sentiment > 0.15 and rsi < 75)
    - Output SHORT when rsi_14 >= 75 (parabolic overbought exhaustion) or when Close < SMA20 with negative sentiment
    - Output REJECT_TRADE ONLY when recent_10d_anomalies >= 3 AND daily_VaR_95_pct < -3.0
    """
    client = _get_gemini_client()
    ticker = quant_data.get("ticker", "TICKER")
    price = quant_data.get("current_price", 0.0)
    vol = quant_data.get("annualized_volatility", 0.0)
    var = quant_data.get("daily_VaR_95_pct", 0.0)
    kelly = quant_data.get("half_kelly_allocation_pct", 10.0)
    suggested_stop = quant_data.get("suggested_stop_loss_atr_pct", 2.5)
    anomalies = quant_data.get("recent_10d_anomalies", 0)
    rsi = quant_data.get("rsi_14", 50.0)
    ret_5d = quant_data.get("recent_5d_return_pct", 0.0)
    close_gt_sma20 = quant_data.get("close_gt_sma20", True)
    sma_trend = quant_data.get("sma20_trend", "BULLISH")
    sentiment, sentiment_label = _sentiment_inputs(news_data)

    # Dynamic conditioning from HMM Regime & LSTM Temporal Anomaly
    hmm_data = quant_data.get("hmm_regime", {})
    temp_anomaly = quant_data.get("temporal_anomaly", {})
    stop_mult = hmm_data.get("stop_loss_multiplier", 1.0)
    suggested_stop = round(suggested_stop * stop_mult, 2)
    kelly_scale = hmm_data.get("kelly_scale_factor", 1.0)
    kelly = round(kelly * kelly_scale, 2)

    # 1. REJECT_TRADE ONLY when recent_10d_anomalies >= 3 AND daily_VaR_95_pct < -3.0
    is_reject_regime = (anomalies >= 3 and var < -3.0)

    # 2. SHORT when rsi_14 >= 75 (parabolic overbought exhaustion) OR (Close < SMA20 with negative sentiment)
    is_short_regime = (rsi >= 75.0) or (not close_gt_sma20 and sentiment < 0.0)

    # 3. LONG when Close > SMA20 or (sentiment > 0.15 and rsi < 75)
    is_long_regime = (close_gt_sma20 or (sentiment > 0.15 and rsi < 75.0)) and not is_short_regime and not is_reject_regime

    prompt = f"""You are the Chief Risk Officer (CRO) of an Autonomous Adversarial Risk Committee (AARC).
Analyze the Bull and Bear cases alongside quantitative signals and render a definitive verdict.

Asset: {ticker}
Current Price: {price}
14-day RSI: {rsi}
20-day SMA Trend: {sma_trend} (Close > SMA20: {close_gt_sma20}, 5-day Return: {ret_5d}%)
Annualized Volatility: {vol}%
95% Daily VaR: {var}%
Suggested Stop Loss (2x ATR %): {suggested_stop}%
Half-Kelly Allocation Cap: {kelly}%
Recent 10d Anomalies: {anomalies}
News Sentiment Score: {sentiment_label}

Bull Argument:
{bull_thesis}

Bear Argument:
{bear_thesis}

MANDATORY DIRECTIONAL CRO RULES:
1. REJECT_TRADE: Output action="REJECT_TRADE" and recommended_allocation_pct=0.0 ONLY when recent_10d_anomalies >= 3 AND daily_VaR_95_pct < -3.0.
2. SHORT: Output action="SHORT" when rsi_14 >= 75 (parabolic overbought exhaustion) OR when Close < SMA20 with negative sentiment.
3. LONG: Output action="LONG" with Half-Kelly allocation ({kelly}%) when Close > SMA20 or (sentiment_score > 0.15 and rsi_14 < 75).

Instructions:
- Return a valid JSON matching CROVerdict schema.
- Action must be one of: "LONG", "SHORT", "REJECT_TRADE".
- recommended_allocation_pct should be up to {kelly}%. If action is REJECT_TRADE, set to 0.0.
"""
    if client:
        res = _call_gemini(client, prompt, schema=CROVerdict)
        if res and res.text:
            try:
                raw_data = json.loads(res.text)
                verdict = CROVerdict.model_validate(raw_data)

                # Ensure strict alignment with the institutional rules
                if is_reject_regime:
                    verdict.action = "REJECT_TRADE"
                    verdict.recommended_allocation_pct = 0.0
                elif is_short_regime and verdict.action == "LONG":
                    verdict.action = "SHORT"
                    verdict.recommended_allocation_pct = min(kelly, 12.0)
                elif is_long_regime and verdict.action != "LONG":
                    verdict.action = "LONG"
                    verdict.recommended_allocation_pct = min(kelly, 15.0)

                # Set allocation floors
                if verdict.action in ("LONG", "SHORT"):
                    verdict.recommended_allocation_pct = max(4.5, min(kelly, verdict.recommended_allocation_pct or kelly))
                else:
                    verdict.recommended_allocation_pct = 0.0

                verdict.verdict = verdict.action
                result = verdict.model_dump()
                
                # Apply Rolling Accuracy & Bayesian Beta-Binomial Calibration
                import datetime
                tracker = RollingAccuracyTracker()
                cal_conf, tracker_telemetry = tracker.calibrate_confidence(result["confidence"], ticker)

                bayesian_calibrator = BetaBinomialConfidenceCalibrator()
                bayes_conf, bayes_telemetry = bayesian_calibrator.calibrate_score(
                    raw_ai_confidence=result["confidence"],
                    ticker=ticker,
                    hmm_regime=quant_data.get("hmm_regime"),
                    action=result["action"],
                    volatility_shock_ratio=quant_data.get("volatility_shock_ratio", 1.0),
                    expected_shortfall_pct=quant_data.get("expected_shortfall_95_pct"),
                    empirical_coverage_pct=quant_data.get("empirical_coverage_pct"),
                    brier_metrics=tracker_telemetry.get("brier_metrics"),
                    walk_forward_metrics=tracker_telemetry.get("walk_forward_metrics"),
                )

                result["raw_confidence"] = result["confidence"]
                result["confidence"] = bayes_conf
                result["rolling_accuracy_telemetry"] = tracker_telemetry
                result["bayesian_calibration"] = bayes_telemetry
                result["temperature_scaling"] = bayes_telemetry.get("temperature_scaling", {})
                result["regime_gating"] = bayes_telemetry.get("regime_gating", {})
                result["brier_metrics"] = tracker_telemetry.get("brier_metrics", {})
                result["walk_forward_validation"] = tracker_telemetry.get("walk_forward_metrics", {})
                result["tail_risk_grounding"] = bayes_telemetry.get("tail_risk_grounding", {})

                if result["action"] in ("LONG", "SHORT"):
                    alloc_mult = bayes_telemetry.get("bayesian_allocation_multiplier", 1.0)
                    if bayes_telemetry.get("regime_gating", {}).get("is_suppressed", False):
                        alloc_mult = 0.0
                    if temp_anomaly.get("is_temporal_anomaly", False):
                        alloc_mult *= 0.85
                    result["recommended_allocation_pct"] = round(result["recommended_allocation_pct"] * alloc_mult, 2)
                    if alloc_mult > 0:
                        result["recommended_allocation_pct"] = max(4.5, min(kelly, result["recommended_allocation_pct"]))
                    else:
                        result["recommended_allocation_pct"] = 0.0

                result["generation_mode"] = "gemini_proposal_with_rule_constraints"
                return result
            except Exception as e:
                print(f"[CRO JSON PARSE ERROR]: {e}")
                traceback.print_exc()

    # Deterministic fallback following exact user rules:
    if is_reject_regime:
        action = "REJECT_TRADE"
        conf = 92
        alloc = 0.0
        kill = f"Isolation Forest flagged critical anomaly clustering ({anomalies} anomalies >= 3) and severe tail loss (VaR {var}% < -3.0%)."
        summary = f"Trade rejected: Extreme microstructural anomaly clustering ({anomalies} anomalies) combined with severe tail loss risk (Daily VaR: {var}%). Capital preservation mandate triggered."
    elif rsi >= 75.0:
        action = "SHORT"
        conf = 88
        alloc = min(kelly, 12.0)
        kill = f"Continuation of parabolic squeeze above 2x ATR trailing stop ({suggested_stop}%)."
        summary = f"Parabolic overbought exhaustion detected (14-day RSI: {rsi} >= 75, 5-day Return: {ret_5d}%). Mean reversion to historical trend lines expected; tactical SHORT sizing approved at {alloc}% with ATR stop at {suggested_stop}%."
    elif (not close_gt_sma20) and sentiment < 0.0:
        action = "SHORT"
        conf = 80
        alloc = min(kelly, 10.0)
        kill = f"Breakout back above 20-day SMA or ATR stop ({suggested_stop}%)."
        summary = f"Bearish trend breakdown confirmed: Spot price trading below 20-day SMA with adverse headline sentiment ({sentiment_label}). SHORT position sizing approved at {alloc}% with stop at {suggested_stop}%."
    elif is_long_regime or close_gt_sma20 or sentiment > 0.0:
        action = "LONG"
        conf = 84
        alloc = min(kelly, 15.0)
        kill = f"Breakdown below 20-day SMA or ATR trailing stop ({suggested_stop}%)."
        sentiment_summary = (
            f"positive news sentiment ({sentiment_label})"
            if sentiment > 0.0
            else f"news sentiment {sentiment_label}"
        )
        summary = f"Institutional LONG execution approved: Trend structure is constructive (SMA20: {sma_trend}, Close: {price}, RSI: {rsi}) with {sentiment_summary}. Half-Kelly capacity allocated at {alloc}% with stop loss at {suggested_stop}%."
    else:
        action = "LONG"
        conf = 72
        alloc = min(kelly, 8.5)
        kill = f"Breach of 2x ATR trailing stop ({suggested_stop}%)."
        summary = f"Controlled LONG positioning permitted at {alloc}%. Volatility ({vol}%) warrants trailing stop at {suggested_stop}%."

    # Bayesian & Rolling Accuracy Calibration Loop
    import datetime
    tracker = RollingAccuracyTracker()
    calibrated_conf, tracker_telemetry = tracker.calibrate_confidence(conf, ticker)

    bayesian_calibrator = BetaBinomialConfidenceCalibrator()
    bayes_conf, bayes_telemetry = bayesian_calibrator.calibrate_score(
        raw_ai_confidence=conf,
        ticker=ticker,
        hmm_regime=quant_data.get("hmm_regime"),
        action=action,
        volatility_shock_ratio=quant_data.get("volatility_shock_ratio", 1.0),
        expected_shortfall_pct=quant_data.get("expected_shortfall_95_pct"),
        empirical_coverage_pct=quant_data.get("empirical_coverage_pct"),
        brier_metrics=tracker_telemetry.get("brier_metrics"),
        walk_forward_metrics=tracker_telemetry.get("walk_forward_metrics"),
    )

    if action in ("LONG", "SHORT"):
        alloc_mult = bayes_telemetry.get("bayesian_allocation_multiplier", 1.0) * tracker_telemetry.get("allocation_multiplier", 1.0)
        if bayes_telemetry.get("regime_gating", {}).get("is_suppressed", False):
            alloc_mult = 0.0
        if temp_anomaly.get("is_temporal_anomaly", False):
            alloc_mult *= 0.85
        alloc = round(alloc * alloc_mult, 2)
        if alloc_mult > 0:
            alloc = max(4.5, min(kelly, alloc))
        else:
            alloc = 0.0
    else:
        alloc = 0.0

    if temp_anomaly.get("is_temporal_anomaly", False) and action != "REJECT_TRADE":
        recon_err = temp_anomaly.get("reconstruction_error", 0.0)
        kill = f"{kill} [LSTM Autoencoder Alert: Sequential anomaly flagged with reconstruction error {recon_err:.3f}]"

    if bayes_telemetry.get("regime_gating", {}).get("is_suppressed", False):
        supp_reason = bayes_telemetry["regime_gating"].get("suppression_reason", "Regime uncertainty")
        kill = f"{kill} [Gated Alert: {supp_reason}]"

    return {
        "generation_mode": "deterministic_rules",
        "action": action,
        "verdict": action,
        "raw_confidence": conf,
        "confidence": bayes_conf,
        "recommended_allocation_pct": alloc,
        "stop_loss_pct": suggested_stop,
        "kill_switch_reason": kill,
        "verdict_summary": summary,
        "rolling_accuracy_telemetry": tracker_telemetry,
        "bayesian_calibration": bayes_telemetry,
        "temperature_scaling": bayes_telemetry.get("temperature_scaling", {}),
        "regime_gating": bayes_telemetry.get("regime_gating", {}),
        "brier_metrics": tracker_telemetry.get("brier_metrics", {}),
        "walk_forward_validation": tracker_telemetry.get("walk_forward_metrics", {}),
        "tail_risk_grounding": bayes_telemetry.get("tail_risk_grounding", {}),
    }


async def stream_committee(ticker: str, cutoff_date: Optional[str] = None):
    """
    Sequentially runs the 5 nodes:
    quant -> news -> bull -> bear -> cro
    and yields each node's dictionary state update.
    """
    clean_ticker = ticker.strip().upper()

    # 1. Quant Node
    try:
        quant_data = await asyncio.to_thread(analyze_ticker_quant, clean_ticker, cutoff_date)
        vol = quant_data.get("annualized_volatility", 20.0)
        hmm_state = quant_data.get("hmm_regime", {})
        quant_data["regime"] = {
            "state": hmm_state.get("active_state"),
            "label": hmm_state.get("regime_name", "UNAVAILABLE"),
            "prob_high_vol": hmm_state.get("probabilities", {}).get("P_Panic"),
            "state_probabilities": hmm_state.get("probabilities", {}),
            "kelly_scale_factor": hmm_state.get("kelly_scale_factor", 1.0),
        }
        quant_data["garch_volatility_forecast_pct"] = quant_data.get("garch_vol_forecast_pct")
        yield {
            "node": "quant",
            "data": quant_data
        }
    except Exception as e:
        print(f"[STREAM QUANT ERROR]: {e}")
        traceback.print_exc()
        yield {
            "node": "error",
            "data": {"stage": "quant", "error": str(e)}
        }
        return

    # 2. News Node
    try:
        news_data = await asyncio.to_thread(
            analyze_ticker_sentiment, clean_ticker, cutoff_date
        )
        yield {
            "node": "news",
            "data": news_data
        }
    except Exception as e:
        print(f"[STREAM NEWS ERROR]: {e}")
        traceback.print_exc()
        # Report sentiment as unavailable; never substitute a fake neutral score.
        news_data = {
            "ticker": clean_ticker,
            "aggregate_sentiment_score": None,
            "headlines": [],
            "news_available": False,
            "news_error": str(e),
        }
        yield {
            "node": "news",
            "data": news_data
        }

    # Indices are benchmarks, not directly tradable instruments. Keep their
    # market analysis, but never generate trade theses or allocation advice.
    if clean_ticker.startswith("^"):
        yield {"node": "index_complete", "data": {"ticker": clean_ticker}}
        return

    # 3. Bull Node
    try:
        bull_thesis = await asyncio.to_thread(run_bull_node, quant_data, news_data)
        yield {
            "node": "bull",
            "data": {
                "bull_thesis": bull_thesis
            }
        }
    except Exception as e:
        print(f"[STREAM BULL ERROR]: {e}")
        traceback.print_exc()
        bull_thesis = "• Bull case unavailable due to generation error."
        yield {
            "node": "bull",
            "data": {"bull_thesis": bull_thesis}
        }

    # 4. Bear Node
    try:
        bear_thesis = await asyncio.to_thread(run_bear_node, quant_data, news_data, bull_thesis)
        yield {
            "node": "bear",
            "data": {
                "bear_thesis": bear_thesis
            }
        }
    except Exception as e:
        print(f"[STREAM BEAR ERROR]: {e}")
        traceback.print_exc()
        bear_thesis = "• Bear case unavailable due to generation error."
        yield {
            "node": "bear",
            "data": {"bear_thesis": bear_thesis}
        }

    # 5. CRO Node
    try:
        cro_verdict = await asyncio.to_thread(run_cro_node, quant_data, news_data, bull_thesis, bear_thesis)

        yield {
            "node": "cro",
            "data": cro_verdict
        }
    except Exception as e:
        print(f"[STREAM CRO ERROR]: {e}")
        traceback.print_exc()
        yield {
            "node": "error",
            "data": {"stage": "cro", "error": str(e)}
        }
