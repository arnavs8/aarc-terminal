"""Regression tests for chronological labels, ingestion and executable returns."""

import numpy as np
import pandas as pd
import pytest

from backend.data.historical import normalize_bars
from backend.historical import build_samples, chronological_split, strategy_results, predict_bundle, save_bundle, metrics
from backend.models.historical_forecast import historical_forecast


def bars(n=500):
    rng = np.random.default_rng(32)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    return pd.DataFrame({
        "Open": close * 0.999, "High": close * 1.01, "Low": close * 0.99,
        "Close": close, "Volume": rng.integers(1000, 10000, n),
    }, index=pd.bdate_range("2015-01-01", periods=n))


def test_features_ignore_future_and_labels_enter_next_open():
    prices = bars()
    table, features = build_samples(prices, "TEST")
    changed = prices.copy()
    changed.iloc[301:, :4] *= 3
    changed_table, _ = build_samples(changed, "TEST")
    pd.testing.assert_frame_equal(table.loc[:250, features], changed_table.loc[:250, features])
    first = table.iloc[0]
    i = prices.index.get_loc(first.signal_date)
    assert first["return"] == pytest.approx(prices.Close.iloc[i + 5] / prices.Open.iloc[i + 1] - 1)


def test_split_purges_label_endpoints():
    table, _ = build_samples(bars(), "TEST")
    train, validation, test = chronological_split(table, "2015-12-31", "2016-05-31")
    assert train.label_end.max() < validation.signal_date.min()
    assert validation.label_end.max() < test.signal_date.min()


def test_intraday_aggregation_and_missing_bar_audit():
    frame = pd.DataFrame({
        "timestamp": ["2020-01-02 09:15:00+05:30", "2020-01-02 09:16:00+05:30", "2020-01-02 15:29:00+05:30"],
        "open": [100, np.nan, 101], "high": [102, np.nan, 104],
        "low": [99, np.nan, 100], "close": [101, np.nan, 103], "volume": [10, np.nan, 20],
    })
    daily = normalize_bars(frame, "india")
    assert daily.iloc[0].tolist() == [100, 104, 99, 103, 30]
    assert daily.attrs["missing_source_bars"] == 1
    assert daily.index[0] == pd.Timestamp("2020-01-02")
    with pytest.raises(ValueError, match="Duplicate"):
        normalize_bars(pd.concat([frame, frame]), "india")


def test_strategy_has_no_overlapping_positions_and_cash_has_no_profit():
    table, _ = build_samples(bars(), "TEST")
    probabilities = np.tile([0.1, 0.8, 0.1], (len(table), 1))
    summary, trades = strategy_results(table, probabilities)
    assert summary["TEST"]["strategy_return"] == 0
    assert (trades.entry_date.iloc[1:].reset_index(drop=True) >
            trades.exit_date.iloc[:-1].reset_index(drop=True)).all()
    long = np.tile([0.1, 0.1, 0.8], (len(table), 1))
    _, positions = strategy_results(table, long, round_trip_bps=30)
    assert positions.iloc[0].net_return == pytest.approx(positions.iloc[0].asset_return - 0.003)


def test_inference_refuses_training_period_and_wrong_market(tmp_path):
    path = tmp_path / "model.joblib"
    save_bundle(path, None, [], {
        "market": "us", "training_data_end": "2030-01-01",
        "temperature": 1, "horizon": 5, "model_name": "test",
    })
    with pytest.raises(ValueError, match="later"):
        predict_bundle(path, bars(), "us")
    with pytest.raises(ValueError, match="market mismatch"):
        predict_bundle(path, bars(), "india")


def test_absent_model_preserves_committee(monkeypatch):
    monkeypatch.delenv("AARC_MODEL_US", raising=False)
    assert historical_forecast("AAPL", bars())["available"] is False


def test_ece_includes_exact_bin_boundaries():
    result = metrics([0, 0], np.array([[0.6, 0.2, 0.2], [0.6, 0.2, 0.2]]))
    assert result["ece_10_bins"] == pytest.approx(0.4)


def test_temporal_folds_purge_future_labels_for_every_symbol():
    from backend.models.historical_selection import purged_folds
    first, _ = build_samples(bars(), "FIRST")
    second, _ = build_samples(bars(), "SECOND")
    for fit, validation in purged_folds(pd.concat([first, second], ignore_index=True)):
        assert fit.label_end.max() < validation.signal_date.min()
        assert set(fit.symbol) == {"FIRST", "SECOND"}


def test_calibration_rejects_wrong_confident_model():
    from backend.models.historical_selection import choose_calibration, PriorBlend
    validation = pd.DataFrame({"symbol": ["A"] * 30, "target": [0, 1, 2] * 10})
    prior = np.array([1/3, 1/3, 1/3])
    learned = np.tile([0.01, 0.01, 0.98], (30, 1))
    weight, temperature, _ = choose_calibration(validation, learned, prior)
    assert weight == 0
    assert temperature == 1
    np.testing.assert_allclose(PriorBlend(None, prior, weight).predict_proba(validation), np.tile(prior, (30, 1)))


def test_baseline_fallback_never_enters_positions():
    table, _ = build_samples(bars(), "TEST")
    probabilities = np.tile([0.05, 0.05, 0.9], (len(table), 1))
    result, trades = strategy_results(table, probabilities, allow_positions=False)
    assert result["TEST"]["active_trades"] == 0
    assert trades.position.sum() == 0
