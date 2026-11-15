"""Resolve runtime calibration storage independently of the process working directory."""
import os
from pathlib import Path


def calibration_path(filename: str) -> Path:
    configured = os.environ.get('AARC_STATE_DIR')
    # Fail closed if tests bypass the suite's isolation fixture.
    if os.environ.get('PYTEST_CURRENT_TEST') and not configured:
        raise RuntimeError('Tests must set AARC_STATE_DIR to isolated temporary storage')
    root = Path(configured).expanduser().resolve() if configured else Path(__file__).resolve().parent.parent / 'data_cache'
    return root / filename
