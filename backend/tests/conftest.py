"""All tests, including API requests, must use disposable runtime state."""
import pytest


@pytest.fixture(autouse=True)
def isolated_runtime_state(tmp_path, monkeypatch):
    state = tmp_path / 'runtime-state'
    monkeypatch.setenv('AARC_STATE_DIR', str(state))
    # Successful SSE tests must not overwrite the app's real demo recordings either.
    import backend.main as main
    monkeypatch.setenv('AARC_NEWS_PROVIDER', 'local')
    monkeypatch.delenv('GROQ_API_KEY', raising=False)
    monkeypatch.setattr(main, 'CACHE_FILE', state / 'demo_cache.json')
    state.mkdir(parents=True, exist_ok=True)
    return state
