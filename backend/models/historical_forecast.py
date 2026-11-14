"""Opt-in integration of locally trained OHLCV models into the terminal."""

import os
from pathlib import Path


def historical_forecast(ticker, bars):
    market = "india" if ticker.endswith((".NS", ".BO")) else "us"
    path = os.getenv(f"AARC_MODEL_{market.upper()}")
    if ticker.startswith("^") or not path:
        return {"available": False, "reason": "No historical model configured for this instrument"}
    try:
        from backend.historical import predict_bundle
        result = predict_bundle(Path(path), bars, market)
        return {"available": True, **result}
    except Exception as exc:
        return {"available": False, "reason": str(exc)}
