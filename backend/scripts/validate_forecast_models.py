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
  - the loaded "model" object is actually usable (has .predict())
  - forecast_metrics.json / forecast_feature_importance.json exist, parse,
    and cover the same 7 horizons with finite, non-negative MAE values
    within a loose sane range (catches a training run that silently
    produced garbage — e.g. an upstream API schema change or a
    degenerate/empty training set) — not an accuracy bar, just a
    "didn't go badly wrong" check

Usage (from the project root):
    python -m backend.scripts.validate_forecast_models
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib

from backend.scripts.train_forecast_models import FEATURE_COLS, HORIZONS, TARGET_COLS

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Loose sanity bounds, not accuracy targets — only meant to catch a
# training run that went badly wrong, not to enforce a specific accuracy bar.
_MAX_SANE_WAVE_MAE_M = 5.0
_MAX_SANE_WIND_MAE_KMH = 50.0


def _validate_model_file(models_dir: Path, horizon: int) -> list[str]:
    errors: list[str] = []
    path = models_dir / f"forecast_day{horizon}.pkl"
    if not path.exists():
        return [f"missing {path.name}"]

    try:
        payload = joblib.load(path)
    except Exception as exc:
        return [f"{path.name} failed to load: {exc}"]

    if not isinstance(payload, dict) or set(payload) != {"model", "features", "targets"}:
        return [f"{path.name} has unexpected shape (expected keys model/features/targets): {payload!r:.200}"]

    if payload["features"] != FEATURE_COLS:
        errors.append(f"{path.name}: features {payload['features']} != current FEATURE_COLS {FEATURE_COLS}")
    if payload["targets"] != TARGET_COLS:
        errors.append(f"{path.name}: targets {payload['targets']} != current TARGET_COLS {TARGET_COLS}")
    if not hasattr(payload["model"], "predict"):
        errors.append(f"{path.name}: 'model' object has no predict() method")

    return errors


def _validate_metrics(models_dir: Path) -> list[str]:
    errors: list[str] = []
    metrics_path = models_dir / "forecast_metrics.json"
    try:
        metrics = json.loads(metrics_path.read_text())
    except Exception as exc:
        return [f"forecast_metrics.json missing/unreadable: {exc}"]

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

    return errors


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


def main() -> None:
    errors = validate_models()
    if errors:
        for e in errors:
            print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"OK — all {len(list(HORIZONS))} forecast models + metrics + feature importance validated.")


if __name__ == "__main__":
    main()
