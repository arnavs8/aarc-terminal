"""Regression: synthetic API outcomes must never alter live scoring state."""
import json
from pathlib import Path
from fastapi.testclient import TestClient
from backend.main import app
from backend.calibration.bayesian_calibrator import BetaBinomialConfidenceCalibrator
from backend.calibration.rolling_tracker import RollingAccuracyTracker
from backend.calibration.storage import calibration_path


def test_api_updates_are_isolated_from_production(isolated_runtime_state):
    live = Path(__file__).resolve().parents[1] / 'data_cache' / 'bayesian_calibration.json'
    before = live.read_bytes() if live.exists() else None
    client = TestClient(app)
    for hit in (True, False):
        response = client.post('/api/confidence/update', json={
            'ticker': 'TCS.NS', 'is_hit': hit, 'is_high_conf_divergence': not hit})
        assert response.status_code == 200
    state = json.loads((isolated_runtime_state/'bayesian_calibration.json').read_text())
    assert state['global']['n_updates'] == 2
    assert (live.read_bytes() if live.exists() else None) == before
    assert RollingAccuracyTracker().storage_path.parent == isolated_runtime_state


def test_production_path_is_independent_of_cwd(monkeypatch, tmp_path):
    monkeypatch.delenv('AARC_STATE_DIR')
    monkeypatch.delenv('PYTEST_CURRENT_TEST')
    monkeypatch.chdir(tmp_path)
    assert calibration_path('example.json') == Path(__file__).resolve().parents[1]/'data_cache'/'example.json'


def test_clean_state_does_not_cap_score_at_forty():
    cal = BetaBinomialConfidenceCalibrator()
    score, telemetry = cal.calibrate_score(88, ticker='RELIANCE.NS',
        hmm_regime={'regime_name':'LOW_VOL_TRENDING', 'probabilities':{'P_Trending':.9,'P_Panic':.05,'P_Choppy':.05}})
    assert score > 80
    assert telemetry['sample_updates'] == 0
