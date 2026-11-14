"""Purged temporal model selection and validation-controlled baseline blending."""

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from backend.models.baseline import BaselineLogisticModel
from backend.models.tree import LightGBMClassifierModel


class PriorBlend:
    """Keep the empirical baseline explicit and serializable for inference."""

    def __init__(self, model, prior, model_weight):
        self.model = model
        self.prior = np.asarray(prior, dtype=float)
        self.model_weight = float(model_weight)

    def predict_proba(self, features):
        baseline = np.tile(self.prior, (len(features), 1))
        if self.model_weight == 0:
            return baseline
        return self.model_weight * self.model.predict_proba(features) + (1 - self.model_weight) * baseline


def purged_folds(table, n_splits=3):
    """All tickers share dates. No training label extends into a validation fold."""
    dates = np.sort(table.signal_date.unique())
    if len(dates) < 120:
        raise ValueError("At least 120 distinct signal dates are required for temporal selection")
    blocks = np.array_split(dates[len(dates) // 2:], n_splits)
    folds = []
    for block in blocks:
        start = pd.Timestamp(block[0])
        stop = pd.Timestamp(block[-1])
        fit = table.loc[table.label_end < start]
        validation = table.loc[(table.signal_date >= start) & (table.label_end <= stop)]
        if len(validation) < 10 or set(fit.target.unique()) != {0, 1, 2}:
            raise ValueError("A temporal fold lacks adequate observations or training classes")
        folds.append((fit, validation))
    return folds


def macro_log_loss(table, probabilities):
    scores = []
    for symbol in table.symbol.unique():
        mask = (table.symbol == symbol).to_numpy()
        scores.append(log_loss(table.loc[mask, "target"], probabilities[mask], labels=[0, 1, 2]))
    return float(np.mean(scores))


def recent_window(table, sessions):
    if sessions is None:
        return table
    dates = np.sort(table.signal_date.unique())
    return table.loc[table.signal_date >= dates[max(0, len(dates) - sessions)]]


def make_model(kind):
    if kind == "small_tree":
        return LightGBMClassifierModel(num_leaves=7, min_child_samples=100, reg_lambda=1.0)
    return BaselineLogisticModel(c_penalty=0.1 if kind == "regularized_logistic" else 1.0)


def temperature_scale(probabilities, temperature):
    logits = np.log(np.clip(probabilities, 1e-9, 1)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=1, keepdims=True)


def choose_calibration(validation, learned, prior):
    """Includes the unmodified prior; ties favor less model influence."""
    baseline = np.tile(prior, (len(validation), 1))
    best_score = macro_log_loss(validation, baseline)
    best_weight, best_temperature = 0.0, 1.0
    for weight in (0.25, 0.5, 0.75, 1.0):
        mixed = weight * learned + (1 - weight) * baseline
        for temperature in (1.0, 0.75, 1.25, 1.5, 2.0, 3.0):
            score = macro_log_loss(validation, temperature_scale(mixed, temperature))
            if score < best_score - 1e-8:
                best_score, best_weight, best_temperature = score, weight, temperature
    return best_weight, best_temperature, best_score


def select_temporal_model(train, validation, features):
    candidates = {
        "logistic_all": ("logistic", None),
        "regularized_logistic_all": ("regularized_logistic", None),
        "regularized_logistic_252": ("regularized_logistic", 252),
        "regularized_logistic_756": ("regularized_logistic", 756),
        "small_tree_756": ("small_tree", 756),
    }
    folds = purged_folds(train)
    scores = {}
    for name, (kind, sessions) in candidates.items():
        fold_scores = []
        for fit, held_out in folds:
            window = recent_window(fit, sessions)
            model = make_model(kind).fit(window[features], window.target)
            fold_scores.append(macro_log_loss(held_out, model.predict_proba(held_out[features])))
        scores[name] = {"fold_macro_log_loss": fold_scores, "mean": float(np.mean(fold_scores))}
    winner = min(scores, key=lambda name: scores[name]["mean"])
    kind, sessions = candidates[winner]
    fit = recent_window(train, sessions)
    model = make_model(kind).fit(fit[features], fit.target)
    prior = np.bincount(train.target, minlength=3).astype(float)
    prior /= prior.sum()
    learned = model.predict_proba(validation[features])
    weight, temperature, calibration_score = choose_calibration(validation, learned, prior)
    selection = {
        "method": "three purged expanding-window folds inside training; separate validation for shrinkage and temperature",
        "candidate_scores": scores, "selected_candidate": winner,
        "training_window_sessions": sessions, "model_weight": weight,
        "validation_macro_log_loss": calibration_score,
        "baseline_validation_macro_log_loss": macro_log_loss(validation, np.tile(prior, (len(validation), 1))),
        "folds": [{
            "fit_label_end": str(fit_fold.label_end.max().date()),
            "validation_start": str(val_fold.signal_date.min().date()),
            "validation_label_end": str(val_fold.label_end.max().date()),
        } for fit_fold, val_fold in folds],
    }
    name = "class_prior" if weight == 0 else f"{winner}_blend_{weight:g}"
    return PriorBlend(model, prior, weight), name, temperature, selection
