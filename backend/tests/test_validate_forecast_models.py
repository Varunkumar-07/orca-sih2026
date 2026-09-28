"""Coverage for backend/scripts/validate_forecast_models.py (audit backlog
item #12 — "no CI job to regenerate/validate" backend/models/*.pkl).

Runs entirely against a scratch tmp_path directory — never touches (or
needs) the real, gitignored, network-trained backend/models/*.pkl files,
so this stays fast and deterministic in the normal per-PR test suite. The
actual regenerate-and-validate-for-real job (live Open-Meteo fetch, ~4
years x 11 anchors) is a separate, non-blocking scheduled/manual CI
workflow — see .github/workflows/models.yml — not this test's concern.

Run: PYTHONPATH=. pytest backend/tests/test_validate_forecast_models.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pytest

from backend.scripts.train_forecast_models import FEATURE_COLS, HORIZONS, TARGET_COLS
from backend.scripts.validate_forecast_models import validate_models


class _FakeModel:
    def predict(self, X):
        return X


def _write_valid_models_dir(models_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics = {}
    importance = {}
    for horizon in HORIZONS:
        joblib.dump(
            {"model": _FakeModel(), "features": FEATURE_COLS, "targets": TARGET_COLS},
            models_dir / f"forecast_day{horizon}.pkl",
        )
        key = f"day{horizon}"
        metrics[key] = {
            "wave_height_m": {"MAE": 0.3, "RMSE": 0.4},
            "wind_kmh": {"MAE": 4.5, "RMSE": 6.0},
            "train_rows": 1000,
            "test_rows": 250,
        }
        importance[key] = {f: round(1.0 / len(FEATURE_COLS), 4) for f in FEATURE_COLS}
    (models_dir / "forecast_metrics.json").write_text(json.dumps(metrics))
    (models_dir / "forecast_feature_importance.json").write_text(json.dumps(importance))


@pytest.fixture
def valid_models_dir(tmp_path) -> Path:
    d = tmp_path / "models"
    _write_valid_models_dir(d)
    return d


def test_a_well_formed_models_directory_has_no_errors(valid_models_dir):
    assert validate_models(valid_models_dir) == []


def test_a_missing_horizon_file_is_reported(valid_models_dir):
    (valid_models_dir / "forecast_day3.pkl").unlink()

    errors = validate_models(valid_models_dir)
    assert any("missing" in e and "forecast_day3.pkl" in e for e in errors)


def test_an_unreadable_pkl_file_is_reported(valid_models_dir):
    (valid_models_dir / "forecast_day1.pkl").write_bytes(b"not a real pickle")

    errors = validate_models(valid_models_dir)
    assert any("forecast_day1.pkl" in e and "failed to load" in e for e in errors)


def test_features_drifted_from_the_training_script_is_reported(valid_models_dir):
    joblib.dump(
        {"model": _FakeModel(), "features": ["only_one_feature"], "targets": TARGET_COLS},
        valid_models_dir / "forecast_day2.pkl",
    )

    errors = validate_models(valid_models_dir)
    assert any("forecast_day2.pkl" in e and "features" in e for e in errors)


def test_a_model_object_without_predict_is_reported(valid_models_dir):
    joblib.dump(
        {"model": object(), "features": FEATURE_COLS, "targets": TARGET_COLS},
        valid_models_dir / "forecast_day4.pkl",
    )

    errors = validate_models(valid_models_dir)
    assert any("forecast_day4.pkl" in e and "predict" in e for e in errors)


def test_missing_metrics_file_is_reported(valid_models_dir):
    (valid_models_dir / "forecast_metrics.json").unlink()

    errors = validate_models(valid_models_dir)
    assert any("forecast_metrics.json" in e for e in errors)


def test_an_absurd_mae_value_is_reported(valid_models_dir):
    metrics = json.loads((valid_models_dir / "forecast_metrics.json").read_text())
    metrics["day5"]["wave_height_m"]["MAE"] = 999.0  # a training run gone badly wrong
    (valid_models_dir / "forecast_metrics.json").write_text(json.dumps(metrics))

    errors = validate_models(valid_models_dir)
    assert any("day5" in e and "wave_height_m" in e and "out of sane range" in e for e in errors)


def test_a_negative_mae_value_is_reported(valid_models_dir):
    metrics = json.loads((valid_models_dir / "forecast_metrics.json").read_text())
    metrics["day1"]["wind_kmh"]["MAE"] = -1.0
    (valid_models_dir / "forecast_metrics.json").write_text(json.dumps(metrics))

    errors = validate_models(valid_models_dir)
    assert any("day1" in e and "wind_kmh" in e for e in errors)


def test_missing_feature_importance_file_is_reported(valid_models_dir):
    (valid_models_dir / "forecast_feature_importance.json").unlink()

    errors = validate_models(valid_models_dir)
    assert any("forecast_feature_importance.json" in e for e in errors)


def test_an_entirely_empty_directory_reports_one_error_per_missing_model_plus_metrics(tmp_path):
    empty_dir = tmp_path / "empty_models"
    empty_dir.mkdir()

    errors = validate_models(empty_dir)

    assert len(errors) >= len(list(HORIZONS)) + 1  # 7 missing .pkl + missing metrics (+ importance)
    assert all("missing" in e or "unreadable" in e for e in errors)
