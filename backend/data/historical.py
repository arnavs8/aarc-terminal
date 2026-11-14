"""Auditable daily OHLCV ingestion for the two supported Kaggle archives."""

import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


SOURCES = {
    "india": "hk7797/stock-market-india",
    "us": "borismarjanovic/price-volume-data-for-all-us-stocks-etfs",
}


def normalize_bars(frame: pd.DataFrame, market: str) -> pd.DataFrame:
    """Aggregate minute observations by local session; retain supplied price basis."""
    frame = frame.copy()
    frame.columns = [str(c).strip().lower().replace("<", "").replace(">", "") for c in frame.columns]
    date_column = next((c for c in ("date", "datetime", "timestamp", "time") if c in frame), None)
    if date_column is None:
        raise ValueError("Missing date/datetime/timestamp column")
    required = ["open", "high", "low", "close", "volume"]
    if not set(required).issubset(frame.columns):
        raise ValueError(f"Missing OHLCV columns: {set(required) - set(frame.columns)}")
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    if dates.isna().any():
        raise ValueError("Invalid timestamps")
    if dates.dt.tz is not None:
        zone = "Asia/Kolkata" if market == "india" else "America/New_York"
        dates = dates.dt.tz_convert(zone).dt.tz_localize(None)
    values = frame[required].apply(pd.to_numeric, errors="coerce")
    values.index = pd.DatetimeIndex(dates)
    values = values.sort_index()
    if values.index.duplicated().any():
        raise ValueError("Duplicate timestamps require source reconciliation")
    empty = values.isna().all(axis=1)
    missing_rows = int(empty.sum())
    incomplete_sessions = int(values.index[empty].normalize().nunique())
    values = values.loc[~empty]
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Nonfinite OHLCV values")
    if (values[required[:4]] <= 0).any().any() or (values.volume < 0).any():
        raise ValueError("Nonpositive prices or negative volume")
    if ((values.high < values[["open", "close", "low"]].max(axis=1)) |
            (values.low > values[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("Inconsistent high/low bounds")
    daily = values.groupby(values.index.normalize()).agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    })
    daily.columns = [c.capitalize() for c in daily.columns]
    daily.index.name = "Date"
    daily.attrs["missing_source_bars"] = missing_rows
    daily.attrs["sessions_with_missing_bars"] = incomplete_sessions
    return daily


def ingest_archive(archive: Path, market: str, output: Path, symbols=None, limit=None):
    """Read CSV/text members directly, never extract archive paths."""
    if market not in SOURCES:
        raise ValueError("market must be india or us")
    output.mkdir(parents=True, exist_ok=True)
    selected = {s.upper() for s in symbols} if symbols else None
    records, failures = [], []
    with zipfile.ZipFile(archive) as zipped:
        members = sorted(n for n in zipped.namelist() if n.lower().endswith((".csv", ".txt")))
        seen = set()
        for member in members:
            name = Path(member).name
            symbol = name.split("__")[0].upper() if market == "india" else name.lower().removesuffix(".txt").removesuffix(".csv").removesuffix(".us").upper()
            if selected and symbol not in selected:
                continue
            if symbol in seen:
                continue
            seen.add(symbol)
            if limit is not None and len(records) >= limit:
                break
            try:
                raw = zipped.read(member)
                daily = normalize_bars(pd.read_csv(io.BytesIO(raw)), market)
                if len(daily) < 260:
                    raise ValueError("Fewer than 260 daily sessions")
                target = output / f"{symbol}.csv"
                daily.to_csv(target)
                records.append({
                    "symbol": symbol, "member": member, "source_sha256": hashlib.sha256(raw).hexdigest(),
                    "daily_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                    "rows": len(daily), "start": str(daily.index.min().date()),
                    "end": str(daily.index.max().date()),
                    "quality": daily.attrs,
                })
            except (ValueError, KeyError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
                failures.append({"member": member, "error": str(exc)})
    manifest = {
        "schema_version": 1, "market": market, "source": SOURCES[market],
        "price_basis": "as supplied by source; adjustment status unverified",
        "aggregation": "daily local-session OHLC and summed volume",
        "missing_bar_policy": "All-empty source bars omitted; partial/nonfinite bars rejected. Daily aggregates use available observations; inspect quality counts.",
        "requested_symbols": sorted(selected) if selected else None,
        "missing_symbols": sorted(selected - seen) if selected else [],
        "records": records, "rejected": failures,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not records:
        raise ValueError(f"No usable series; inspect {output / 'manifest.json'}")
    return manifest
