"""Offline historical benchmark, retraining and inference.

Run python -m backend.historical --help. Uses only OHLCV available at signal time.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from backend.data.historical import ingest_archive
from backend.features.technical import compute_technical_features
from backend.models.historical_selection import select_temporal_model, temperature_scale


SCHEMA_VERSION = 1
CLASSES = ["DOWN", "FLAT", "UP"]


def build_samples(bars, symbol, horizon=5):
    if horizon < 1:
        raise ValueError("Horizon must be positive")
    features = compute_technical_features(bars).replace([np.inf, -np.inf], np.nan)
    # Signal is known after the close; executable entry is the next session open.
    returns = bars.Close.shift(-horizon) / bars.Open.shift(-1) - 1
    threshold = (0.5 * bars.Close.pct_change().rolling(20).std() * np.sqrt(horizon)).clip(0.01, 0.15)
    labels = pd.Series(np.select([returns < -threshold, returns > threshold], [0, 2], default=1), index=bars.index)
    dates = pd.Series(bars.index, index=bars.index)
    table = features.copy()
    table["target"] = labels
    table["return"] = returns
    table["signal_date"] = bars.index
    table["entry_date"] = dates.shift(-1)
    table["label_end"] = dates.shift(-horizon)
    table["symbol"] = symbol
    table = table.loc[features.notna().all(axis=1) & returns.notna() & threshold.notna()]
    return table.reset_index(drop=True), list(features.columns)


def read_dataset(directory):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    series = {}
    for record in manifest["records"]:
        path = directory / (record["symbol"] + ".csv")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["daily_sha256"]:
            raise ValueError(f"Data changed since ingestion: {path}")
        bars = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
        series[record["symbol"]] = bars
    return manifest, series


def assemble(series, horizon):
    tables = []
    for symbol, bars in series.items():
        table, features = build_samples(bars, symbol, horizon)
        tables.append(table)
    result = pd.concat(tables, ignore_index=True).sort_values(["signal_date", "symbol"])
    if result.empty:
        raise ValueError("No usable labeled samples")
    return result, features


def chronological_split(table, train_end, validation_end):
    train_end, validation_end = pd.Timestamp(train_end), pd.Timestamp(validation_end)
    if train_end >= validation_end:
        raise ValueError("train-end must precede validation-end")
    # Label endpoints, not just signal dates, determine training eligibility.
    train = table.loc[table.label_end <= train_end]
    validation = table.loc[(table.signal_date > train_end) & (table.label_end <= validation_end)]
    test = table.loc[table.signal_date > validation_end]
    for name, sample in [("train", train), ("validation", validation), ("test", test)]:
        if len(sample) < 30:
            raise ValueError(f"{name} has only {len(sample)} samples; choose different boundaries")
    if set(train.target.unique()) != {0, 1, 2}:
        raise ValueError("Training data must contain all three classes")
    return train, validation, test


def scale_probabilities(probabilities, temperature):
    return temperature_scale(probabilities, temperature)


def metrics(y, probabilities):
    y = np.asarray(y, dtype=int)
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    ece = 0.0
    bins = np.minimum((confidence * 10).astype(int), 9)
    for bin_index in range(10):
        selected = bins == bin_index
        if selected.any():
            ece += selected.mean() * abs((predicted[selected] == y[selected]).mean() - confidence[selected].mean())
    return {
        "n": len(y), "accuracy": float(accuracy_score(y, predicted)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1, 2])),
        "brier": float(np.mean(np.sum((probabilities - np.eye(3)[y]) ** 2, axis=1))),
        "ece_10_bins": float(ece),
    }


def select_model(train, validation, features):
    return select_temporal_model(train, validation, features)


def strategy_results(test, probabilities, round_trip_bps=30, allow_positions=True):
    """Independent equal-budget sleeves, sequential five-session trades per ticker.

    Report each sleeve separately. No compounding overlapping forward returns.
    Direction is long/cash, allowing comparison across the two markets.
    """
    trades, summaries = [], {}
    for symbol, rows in test.assign(p_up=probabilities[:, 2], p_down=probabilities[:, 0]).groupby("symbol"):
        next_signal = pd.Timestamp.min
        equity, benchmark = 1.0, 1.0
        peaks, max_drawdown = 1.0, 0.0
        count = 0
        for row in rows.sort_values("signal_date").itertuples():
            if row.signal_date < next_signal:
                continue
            position = int(allow_positions and row.p_up >= 0.5 and row.p_up > row.p_down)
            # "return" is a Python keyword; use the column value through the index.
            realized = float(rows.loc[row.Index, "return"])
            net = position * (realized - round_trip_bps / 10000)
            equity *= 1 + net
            benchmark *= 1 + realized - round_trip_bps / 10000
            peaks = max(peaks, equity)
            max_drawdown = min(max_drawdown, equity / peaks - 1)
            count += position
            trades.append({
                "symbol": symbol, "signal_date": row.signal_date, "entry_date": row.entry_date,
                "exit_date": row.label_end, "position": position, "asset_return": realized,
                "net_return": net, "equity": equity,
            })
            next_signal = row.label_end
        summaries[symbol] = {
            "strategy_return": equity - 1, "always_long_same_schedule_return": benchmark - 1,
            "max_drawdown_at_exit": -max_drawdown, "active_trades": count,
        }
    return summaries, pd.DataFrame(trades)


def save_bundle(path, model, features, metadata):
    metadata = {
        **metadata,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": 42,
        "versions": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "lightgbm", "joblib")},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    joblib.dump({"schema_version": SCHEMA_VERSION, "model": model, "features": features, **metadata}, temporary)
    temporary.replace(path)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def predict_bundle(path, bars, market, as_of=None):
    # Load only a locally trained artifact that the operator trusts.
    bundle = joblib.load(path)
    if bundle.get("schema_version") != SCHEMA_VERSION or bundle["market"] != market:
        raise ValueError("Model schema or market mismatch")
    history = bars.loc[:as_of] if as_of else bars
    if history.empty:
        raise ValueError("No data at requested cutoff")
    date = history.index[-1]
    if date <= pd.Timestamp(bundle["training_data_end"]):
        raise ValueError("Inference date must be later than all training/calibration observations")
    feature = compute_technical_features(history).iloc[[-1]][bundle["features"]]
    if not np.isfinite(feature.to_numpy()).all():
        raise ValueError("Insufficient valid feature history")
    probability = scale_probabilities(bundle["model"].predict_proba(feature), bundle["temperature"])[0]
    return {
        "as_of": date.date().isoformat(), "market": market,
        "horizon_sessions": bundle["horizon"], "model": bundle["model_name"],
        "training_data_end": bundle["training_data_end"],
        "age_calendar_days": (date - pd.Timestamp(bundle["training_data_end"])).days,
        "probabilities": dict(zip(CLASSES, probability.tolist())),
        "prediction": CLASSES[int(probability.argmax())],
        "decision": "ABSTAIN" if bundle["model_name"] == "class_prior" or probability.max() < 0.5 else CLASSES[int(probability.argmax())],
        "selection": bundle.get("selection"),
        "scope": "OHLCV classifier; experimental, separate from committee confidence",
    }


def benchmark(args):
    manifest, series = read_dataset(args.data)
    table, features = assemble(series, args.horizon)
    train, validation, test = chronological_split(table, args.train_end, args.validation_end)
    model, name, temperature, selection = select_model(train, validation, features)
    raw = model.predict_proba(test[features])
    calibrated = scale_probabilities(raw, temperature)
    prior = np.bincount(train.target, minlength=3) / len(train)
    prior_predictions = np.tile(prior, (len(test), 1))
    momentum = np.tile([0.2, 0.6, 0.2], (len(test), 1))
    momentum[test.return_5d > 0.01] = [0.2, 0.2, 0.6]
    momentum[test.return_5d < -0.01] = [0.6, 0.2, 0.2]
    per_symbol = {}
    for symbol in test.symbol.unique():
        mask = (test.symbol == symbol).to_numpy()
        per_symbol[symbol] = metrics(test.loc[mask, "target"], calibrated[mask])
    strategy, trades = strategy_results(test, calibrated, args.round_trip_bps, allow_positions=name != "class_prior")
    metadata = {
        "market": manifest["market"], "source": manifest["source"], "model_name": name,
        "horizon": args.horizon, "temperature": temperature,
        "training_data_end": args.validation_end, "purpose": "frozen benchmark model",
        "features": features, "symbols": sorted(series),
        "selection": selection,
        "data_hashes": {record["symbol"]: record["daily_sha256"] for record in manifest["records"]},
    }
    report = {
        **metadata, "train_end": args.train_end, "validation_end": args.validation_end,
        "split_rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "test_start": str(test.signal_date.min().date()), "test_end": str(test.label_end.max().date()),
        "selection_audit": selection,
        "evaluation_status": "Previously inspected test periods are development comparisons; reserve fresh dates/universe for confirmation.",
        "test_metrics": {
            "class_prior": metrics(test.target, prior_predictions),
            "momentum": metrics(test.target, momentum),
            "selected_raw": metrics(test.target, raw),
            "selected_calibrated": metrics(test.target, calibrated),
        },
        "macro_symbol_log_loss": float(np.mean([v["log_loss"] for v in per_symbol.values()])),
        "per_symbol": per_symbol, "strategy": strategy, "round_trip_bps": args.round_trip_bps,
        "limitations": [
            "Selected historical universe can have survivorship and selection bias.",
            "Source corporate-action adjustment basis is unverified.",
            "Price/volume only; historical news and point-in-time fundamentals excluded.",
            "Test labels overlap for classification; effective sample size is lower than row count.",
            "Strategy uses non-overlapping long/cash trades and exit-only drawdown, excludes dividends and overnight gaps between trades.",
            "Historic test performance does not establish current-market performance.",
        ],
    }
    candidate = report["test_metrics"]["selected_calibrated"]
    baseline = report["test_metrics"]["class_prior"]
    report["beats_class_prior"] = candidate["log_loss"] < baseline["log_loss"] and candidate["brier"] < baseline["brier"]
    report["promotion"] = "Research candidate only; requires recent, untouched data and verified price basis."
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    predictions = test[["symbol", "signal_date", "entry_date", "label_end", "target", "return"]].copy()
    predictions[["p_down", "p_flat", "p_up"]] = calibrated
    predictions["decision"] = np.where(
        (calibrated.max(axis=1) < 0.5) | (name == "class_prior"),
        "ABSTAIN", np.array(CLASSES)[calibrated.argmax(axis=1)],
    )
    predictions.to_csv(args.output / "predictions.csv", index=False)
    trades.to_csv(args.output / "trades.csv", index=False)
    (args.output / "data-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    save_bundle(args.output / "model.joblib", model, features, metadata)
    lines = ["# Historical OHLCV benchmark", "", f"Market {manifest['market']}, model {name}, horizon {args.horizon} sessions.",
             f"Train through {args.train_end}, validate through {args.validation_end}, test {report['test_start']} to {report['test_end']}.",
             "", "| Model | Accuracy | Log loss | Brier | ECE |", "|---|---:|---:|---:|---:|"]
    for label, values in report["test_metrics"].items():
        lines.append(f"| {label} | {values['accuracy']:.3f} | {values['log_loss']:.4f} | {values['brier']:.4f} | {values['ece_10_bins']:.4f} |")
    lines += ["", f"Improves on class-prior log loss and Brier score: {report['beats_class_prior']}.",
              report["promotion"], report["evaluation_status"],
              f"Validation-selected model weight {selection['model_weight']}, temperature {temperature}.",
              "", "## Scope and assumptions", ""] + [f"- {s}" for s in report["limitations"]]
    (args.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"report": str(args.output / "report.md"), "metrics": report["test_metrics"]}


def train(args):
    manifest, series = read_dataset(args.data)
    cutoff = pd.Timestamp(args.through)
    latest = max(bars.index.max() for bars in series.values())
    if cutoff > latest:
        raise ValueError(f"Requested training cutoff exceeds available data ending {latest.date()}")
    series = {symbol: bars.loc[:cutoff] for symbol, bars in series.items()}
    table, features = assemble(series, args.horizon)
    dates = sorted(table.signal_date.unique())
    boundary = pd.Timestamp(dates[int(len(dates) * 0.8)])
    fit = table.loc[table.label_end < boundary]
    calibration = table.loc[table.signal_date >= boundary]
    if len(fit) < 100 or len(calibration) < 30 or set(fit.target.unique()) != {0, 1, 2}:
        raise ValueError("Insufficient training/calibration data")
    model, name, temperature, selection = select_model(fit, calibration, features)
    metadata = {
        "market": manifest["market"], "source": manifest["source"], "model_name": name,
        "horizon": args.horizon, "temperature": temperature, "training_data_end": args.through,
        "fit_end": str(fit.label_end.max().date()), "purpose": "deployment candidate; validate before promotion",
        "selection": selection, "symbols": sorted(series),
        "data_hashes": {record["symbol"]: record["daily_sha256"] for record in manifest["records"]},
    }
    save_bundle(args.output, model, features, metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest")
    ingest.add_argument("--archive", type=Path, required=True)
    ingest.add_argument("--market", choices=["india", "us"], required=True)
    ingest.add_argument("--output", type=Path, required=True)
    ingest.add_argument("--symbols", nargs="+")
    ingest.add_argument("--limit", type=int)
    bench = commands.add_parser("benchmark")
    fit = commands.add_parser("train")
    for command in (bench, fit):
        command.add_argument("--data", type=Path, required=True)
        command.add_argument("--horizon", type=int, default=5)
        command.add_argument("--output", type=Path, required=True)
    bench.add_argument("--train-end", required=True)
    bench.add_argument("--validation-end", required=True)
    bench.add_argument("--round-trip-bps", type=float, default=30)
    fit.add_argument("--through", required=True)
    predict = commands.add_parser("predict")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--csv", type=Path, required=True)
    predict.add_argument("--market", choices=["india", "us"], required=True)
    predict.add_argument("--as-of")
    args = parser.parse_args()
    if args.command == "ingest":
        result = ingest_archive(args.archive, args.market, args.output, args.symbols, args.limit)
    elif args.command == "benchmark":
        if args.round_trip_bps < 0:
            parser.error("round-trip-bps must be nonnegative")
        result = benchmark(args)
    elif args.command == "train":
        result = train(args)
    else:
        bars = pd.read_csv(args.csv, parse_dates=["Date"]).set_index("Date").sort_index()
        result = predict_bundle(args.model, bars, args.market, args.as_of)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
