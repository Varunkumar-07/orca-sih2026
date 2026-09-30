"""
Validates backend/models/*.pkl — the 7 per-horizon forecast models
train_forecast_models.py produces — audit backlog item #12 ("gitignored
trained artifacts; no CI job to regenerate/validate them").

Checks:
  - all 7 forecast_day{1..7}.pkl files exist and load via joblib
  - each has the {"model", "features", "targets"} shape
    forecast_service.py's own _load_horizon_model expects, and that
    features/targets match train_forecast_models.py's OWN current
    FEATURE_COLS/TARGET_COLS — catches serving-time feature-building
    silently drifting out of lockstep with what a stale model was
    actually trained on
  - the loaded "model" object is actually usable (has .predict()), and the
    artifact carries a valid "target_transform" (what forecast_service.py
    decodes predictions with — see backend/services/forecast_targets.py)
  - forecast_metrics.json / forecast_feature_importance.json exist, parse,
    and cover the same 7 horizons with finite, non-negative MAE values
    within a loose sane range (catches a training run that silently
    produced garbage — e.g. an upstream API schema change or a
    degenerate/empty training set) — not an accuracy bar, just a
    "didn't go badly wrong" check
  - forecast_metrics.json has a "_meta" block (training time, data window,
    anchors, split, model settings) and, for every horizon, both baselines
    (persistence, climatology) scored per target — an error if missing
  - prints a model-vs-baseline MAE table; a horizon/target where the model
    does worse than a baseline is a WARNING, not a failure (it's reported,
    so it can't go unnoticed, but it doesn't block the run)

Usage (from the project root):
    python -m backend.scripts.validate_forecast_models
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib

from backend.scripts.train_forecast_models import BASELINES, FEATURE_COLS, HORIZONS, TARGET_COLS
from backend.services.forecast_targets import KINDS

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Loose sanity bounds, not accuracy targets — only meant to catch a
# training run that went badly wrong, not to enforce a specific accuracy bar.
_MAX_SANE_WAVE_MAE_M = 5.0
_MAX_SANE_WIND_MAE_KMH = 50.0
_REQUIRED_META_KEYS = ("trained_at", "data_window", "anchors", "split", "model")


def _validate_model_file(models_dir: Path, horizon: int) -> list[str]:
    errors: list[str] = []
    path = models_dir / f"forecast_day{horizon}.pkl"
    if not path.exists():
        return [f"missing {path.name}"]

    try:
        payload = joblib.load(path)
    except Exception as exc:
        return [f"{path.name} failed to load: {exc}"]

    if not isinstance(payload, dict) or set(payload) != {"model", "features", "targets", "target_transform"}:
        return [f"{path.name} has unexpected shape (expected keys model/features/targets/target_transform): {payload!r:.200}"]

    if payload["features"] != FEATURE_COLS:
        errors.append(f"{path.name}: features {payload['features']} != current FEATURE_COLS {FEATURE_COLS}")
    if payload["targets"] != TARGET_COLS:
        errors.append(f"{path.name}: targets {payload['targets']} != current TARGET_COLS {TARGET_COLS}")
    if not hasattr(payload["model"], "predict"):
        errors.append(f"{path.name}: 'model' object has no predict() method")
    transform = payload["target_transform"]
    if not isinstance(transform, dict) or transform.get("kind") not in KINDS:
        errors.append(f"{path.name}: target_transform {transform!r:.200} is not one of {KINDS}")
    elif transform["kind"] != "raw" and not (
        len(transform.get("mean", [])) == len(TARGET_COLS) and len(transform.get("std", [])) == len(TARGET_COLS)
    ):
        errors.append(f"{path.name}: target_transform needs a mean and std per target")

    return errors


def _validate_metrics(models_dir: Path) -> list[str]:
    errors: list[str] = []
    metrics_path = models_dir / "forecast_metrics.json"
    try:
        metrics = json.loads(metrics_path.read_text())
    except Exception as exc:
        return [f"forecast_metrics.json missing/unreadable: {exc}"]

    meta = metrics.get("_meta")
    if not isinstance(meta, dict):
        errors.append("forecast_metrics.json: missing _meta block (training time, data window, split, ...)")
    else:
        for field in _REQUIRED_META_KEYS:
            if field not in meta:
                errors.append(f"forecast_metrics.json: _meta is missing '{field}'")

    for horizon in HORIZONS:
        key = f"day{horizon}"
        day_metrics = metrics.get(key)
        if day_metrics is None:
            errors.append(f"forecast_metrics.json: missing entry for {key}")
            continue

        wave_mae = day_metrics.get("wave_height_m", {}).get("MAE")
        if not isinstance(wave_mae, (int, float)) or not (0 <= wave_mae <= _MAX_SANE_WAVE_MAE_M):
            errors.append(f"{key}: wave_height_m MAE {wave_mae!r} out of sane range [0, {_MAX_SANE_WAVE_MAE_M}]")

        wind_mae = day_metrics.get("wind_kmh", {}).get("MAE")
        if not isinstance(wind_mae, (int, float)) or not (0 <= wind_mae <= _MAX_SANE_WIND_MAE_KMH):
            errors.append(f"{key}: wind_kmh MAE {wind_mae!r} out of sane range [0, {_MAX_SANE_WIND_MAE_KMH}]")

        baselines = day_metrics.get("baselines")
        if not isinstance(baselines, dict):
            errors.append(f"{key}: missing baselines")
            continue
        for name in BASELINES:
            for target in TARGET_COLS:
                mae = (baselines.get(name) or {}).get(target, {}).get("MAE")
                if not isinstance(mae, (int, float)) or mae < 0:
                    errors.append(f"{key}: {name} baseline MAE for {target} missing or invalid ({mae!r})")

    return errors


def baseline_comparison(models_dir: Path = MODELS_DIR) -> tuple[list[str], list[str]]:
    """(table_lines, warnings): model vs baseline MAE per horizon and
    target, and one warning per case where the model's MAE is worse than a
    baseline's. Empty when the metrics can't be read or lack baselines —
    validate_models() reports those as errors."""
    try:
        metrics = json.loads((models_dir / "forecast_metrics.json").read_text())
    except Exception:
        return [], []

    lines = [f"{'day':<5}{'target':<15}{'model':>9}{'persist':>9}{'clim':>9}{'skill_p':>9}{'skill_c':>9}"]
    warnings: list[str] = []
    for horizon in HORIZONS:
        key = f"day{horizon}"
        day = metrics.get(key) or {}
        baselines = day.get("baselines")
        if not isinstance(baselines, dict):
            continue
        for target in TARGET_COLS:
            model_mae = day.get(target, {}).get("MAE")
            base = {name: (baselines.get(name) or {}).get(target, {}).get("MAE") for name in BASELINES}
            if not isinstance(model_mae, (int, float)) or not all(isinstance(v, (int, float)) for v in base.values()):
                continue
            skill = {name: (1 - model_mae / v if v > 0 else float("nan")) for name, v in base.items()}
            lines.append(
                f"{key:<5}{target:<15}{model_mae:>9.3f}{base['persistence']:>9.3f}{base['climatology']:>9.3f}"
                f"{skill['persistence']:>9.3f}{skill['climatology']:>9.3f}"
            )
            for name, v in base.items():
                if model_mae > v:
                    warnings.append(f"{key} {target}: model MAE {model_mae} is worse than {name} baseline MAE {v}")
    return lines, warnings


def _validate_feature_importance(models_dir: Path) -> list[str]:
    path = models_dir / "forecast_feature_importance.json"
    if not path.exists():
        return ["forecast_feature_importance.json is missing"]
    try:
        json.loads(path.read_text())
    except Exception as exc:
        return [f"forecast_feature_importance.json unreadable: {exc}"]
    return []


def validate_models(models_dir: Path = MODELS_DIR) -> list[str]:
    """Returns a list of human-readable error strings — empty means every
    check passed. Pure/side-effect-free beyond reading files, so this is
    unit-testable against a scratch directory without needing real
    (network-trained) models."""
    errors: list[str] = []
    for horizon in HORIZONS:
        errors.extend(_validate_model_file(models_dir, horizon))
    errors.extend(_validate_metrics(models_dir))
    errors.extend(_validate_feature_importance(models_dir))
    return errors


def main(models_dir: Path = MODELS_DIR) -> None:
    lines, warnings = baseline_comparison(models_dir)
    if lines:
        print("MAE — model vs baselines (skill = 1 - model/baseline; above 0 = model better):")
        for line in lines:
            print(f"  {line}")
    for w in warnings:
        print(f"WARNING: {w}")

    errors = validate_models(models_dir)
    if errors:
        for e in errors:
            print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
    note = f" ({len(warnings)} baseline warning(s) above)" if warnings else ""
    print(f"OK — all {len(list(HORIZONS))} forecast models + metrics + feature importance validated{note}.")


if __name__ == "__main__":
    main()
