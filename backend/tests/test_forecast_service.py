"""Pytest coverage for backend/services/forecast_service.py (previously
untested — added as part of the codebase audit's P1 coverage gaps).

Mocking strategy: _fetch_marine_daily/_fetch_archive_daily (imported from
analytics_service.py) are patched directly, same "patch our own function,
not the network" principle as everywhere else in this suite.
_load_horizon_model/_load_metrics are patched directly too, rather than
requiring real trained .pkl artifacts on disk — a fake model object with a
.predict() method stands in for the real RandomForestRegressor, matching
how test_pfz_service.py stands in real xarray.DataArrays for Copernicus
responses (structurally real, not a hand-rolled shortcut around the actual
row-building/feature logic under test).

Covers:
  - get_forecast: fewer than 2 marine/wind values -> status="error" with
    the "insufficient recent data" reason, no model ever loaded; any
    required input (wave/wind/SST/air-temp day-0) missing -> status=
    "error" with the "required input variables...unavailable" reason; a
    missing model artifact for any horizon -> status="error" naming that
    horizon, without silently skipping it; the full success path returns
    exactly 7 rounded (wave_height_m, wind_kmh) predictions with each
    horizon's real validation MAE attached, and never leaks the model's
    own numeric noise (rounding is real, not accidental); an unexpected
    exception anywhere in the pipeline degrades to status="error" rather
    than propagating

Run from project root:  PYTHONPATH=. pytest   or   pytest
"""
from __future__ import annotations

import asyncio

from backend.services import forecast_service as svc

_LAT, _LON = 13.08, 80.27
_FEATURES = [
    "lat", "lon", "month_sin", "month_cos",
    "wave_height_m", "wave_height_lag1",
    "wind_kmh", "wind_kmh_lag1",
    "sst_celsius", "air_temp_celsius",
]


class _FakeModel:
    def __init__(self, wave_pred: float, wind_pred: float):
        self._wave_pred = wave_pred
        self._wind_pred = wind_pred

    def predict(self, row):
        return [[self._wave_pred, self._wind_pred]]


def _async_return(value):
    async def _inner(*_a, **_k):
        return value

    return _inner


def _install_sufficient_fetches(monkeypatch, *, wave=(1.0, 1.2), sst=(27.5, 28.0), wind=(10.0, 12.0), temp=(29.0, 29.5)):
    async def fake_marine(lat, lon, start_date, end_date):
        return {"wave_height_max": list(wave), "sea_surface_temperature_max": list(sst)}

    async def fake_archive(lat, lon, start_date, end_date):
        return {"wind_speed_10m_max": list(wind), "temperature_2m_mean": list(temp)}

    monkeypatch.setattr(svc, "_fetch_marine_daily", fake_marine)
    monkeypatch.setattr(svc, "_fetch_archive_daily", fake_archive)


def _install_models(monkeypatch, *, missing_horizon: int | None = None):
    def fake_load(horizon):
        if horizon == missing_horizon:
            return None
        return {"model": _FakeModel(wave_pred=1.234, wind_pred=15.678), "features": _FEATURES}

    monkeypatch.setattr(svc, "_load_horizon_model", fake_load)
    monkeypatch.setattr(
        svc,
        "_load_metrics",
        lambda: {f"day{h}": {"wave_height_m": {"MAE": 0.1 * h}, "wind_kmh": {"MAE": 0.5 * h}} for h in svc.HORIZONS},
    )


def test_insufficient_data_fewer_than_two_values_is_error(monkeypatch):
    _install_sufficient_fetches(monkeypatch, wave=(1.0,), wind=(10.0,))  # only 1 value each

    async def fail_if_called(*_a, **_k):
        raise AssertionError("must not load a model when there isn't enough input data")

    monkeypatch.setattr(svc, "_load_horizon_model", fail_if_called)

    result = asyncio.run(svc.get_forecast(_LAT, _LON))

    assert result["status"] == "error"
    assert "insufficient recent" in result["reason"]
    assert result["forecast"] == []


def test_missing_required_variable_is_error(monkeypatch):
    """SST present but with a None in it — a real coastal grid gap, not
    just a short series."""
    _install_sufficient_fetches(monkeypatch, sst=(27.5, None))

    async def fail_if_called(*_a, **_k):
        raise AssertionError("must not load a model when a required input is missing")

    monkeypatch.setattr(svc, "_load_horizon_model", fail_if_called)

    result = asyncio.run(svc.get_forecast(_LAT, _LON))

    assert result["status"] == "error"
    assert "unavailable for this location" in result["reason"]


def test_missing_model_artifact_for_a_horizon_is_error(monkeypatch):
    _install_sufficient_fetches(monkeypatch)
    _install_models(monkeypatch, missing_horizon=3)

    result = asyncio.run(svc.get_forecast(_LAT, _LON))

    assert result["status"] == "error"
    assert "day+3" in result["reason"]
    assert result["forecast"] == []


def test_successful_forecast_returns_seven_rounded_predictions_with_mae(monkeypatch):
    _install_sufficient_fetches(monkeypatch)
    _install_models(monkeypatch)

    result = asyncio.run(svc.get_forecast(_LAT, _LON))

    assert result["status"] == "ok"
    assert len(result["forecast"]) == 7
    assert [f["horizon"] for f in result["forecast"]] == list(range(1, 8))
    first = result["forecast"][0]
    assert first["wave_height_m"] == 1.23  # rounded to 2 decimals
    assert first["wind_kmh"] == 15.7  # rounded to 1 decimal
    assert first["wave_height_mae"] == 0.1 * 1
    assert first["wind_kmh_mae"] == 0.5 * 1
    assert "model" in result and "RandomForestRegressor" in result["model"]
    assert "based_on_date" in result


def test_meta_block_and_baselines_in_metrics_do_not_change_per_day_mae(monkeypatch):
    """forecast_metrics.json now carries a top-level "_meta" block and
    per-day "baselines"; the MAE shown next to each prediction must still
    be the model's own per-day MAE."""
    _install_sufficient_fetches(monkeypatch)
    _install_models(monkeypatch)
    metrics = {
        "_meta": {"trained_at": "2026-09-30T00:00:00Z"},
        **{
            f"day{h}": {
                "wave_height_m": {"MAE": 0.1 * h},
                "wind_kmh": {"MAE": 0.5 * h},
                "baselines": {"persistence": {"wave_height_m": {"MAE": 9.9}, "wind_kmh": {"MAE": 99.0}}},
            }
            for h in svc.HORIZONS
        },
    }
    monkeypatch.setattr(svc, "_load_metrics", lambda: metrics)

    result = asyncio.run(svc.get_forecast(_LAT, _LON))

    assert result["status"] == "ok"
    assert [f["wave_height_mae"] for f in result["forecast"]] == [0.1 * h for h in svc.HORIZONS]
    assert [f["wind_kmh_mae"] for f in result["forecast"]] == [0.5 * h for h in svc.HORIZONS]


def test_predictions_are_decoded_with_the_artifacts_residual_transform(monkeypatch):
    """A residual model outputs a standardized change from day 0: the
    served value must be mean + std * output, plus that point's own day-0
    value (the last of the fetched pair: wave 1.2 m, wind 12.0 km/h)."""
    _install_sufficient_fetches(monkeypatch)
    transform = {"kind": "residual", "targets": ["wave_height_m", "wind_kmh"], "mean": [0.1, 1.0], "std": [0.5, 4.0]}

    def fake_load(horizon):
        return {"model": _FakeModel(wave_pred=1.0, wind_pred=-0.5), "features": _FEATURES, "target_transform": transform}

    monkeypatch.setattr(svc, "_load_horizon_model", fake_load)
    monkeypatch.setattr(svc, "_load_metrics", lambda: {})

    result = asyncio.run(svc.get_forecast(_LAT, _LON))

    assert result["status"] == "ok"
    first = result["forecast"][0]
    assert first["wave_height_m"] == round(1.2 + 0.1 + 0.5 * 1.0, 2)  # 1.8
    assert first["wind_kmh"] == round(12.0 + 1.0 + 4.0 * -0.5, 1)  # 11.0


def test_predictions_are_decoded_with_a_standardized_transform(monkeypatch):
    _install_sufficient_fetches(monkeypatch)
    transform = {"kind": "standardized", "targets": ["wave_height_m", "wind_kmh"], "mean": [1.5, 20.0], "std": [0.4, 5.0]}

    def fake_load(horizon):
        return {"model": _FakeModel(wave_pred=0.5, wind_pred=1.0), "features": _FEATURES, "target_transform": transform}

    monkeypatch.setattr(svc, "_load_horizon_model", fake_load)
    monkeypatch.setattr(svc, "_load_metrics", lambda: {})

    first = asyncio.run(svc.get_forecast(_LAT, _LON))["forecast"][0]
    assert first["wave_height_m"] == 1.7  # no day-0 add-back
    assert first["wind_kmh"] == 25.0


def test_forecast_dates_are_sequential_days_after_day0(monkeypatch):
    _install_sufficient_fetches(monkeypatch)
    _install_models(monkeypatch)

    result = asyncio.run(svc.get_forecast(_LAT, _LON))

    from datetime import date as _date
    from datetime import timedelta as _timedelta

    day0 = _date.fromisoformat(result["based_on_date"])
    expected_dates = [(day0 + _timedelta(days=h)).isoformat() for h in range(1, 8)]
    assert [f["date"] for f in result["forecast"]] == expected_dates


def test_unexpected_exception_degrades_to_error_not_raised(monkeypatch):
    async def broken(*_a, **_k):
        raise RuntimeError("simulated upstream failure")

    monkeypatch.setattr(svc, "_fetch_marine_daily", broken)
    monkeypatch.setattr(svc, "_fetch_archive_daily", _async_return({}))

    result = asyncio.run(svc.get_forecast(_LAT, _LON))

    assert result["status"] == "error"
    assert "forecast failed" in result["reason"]


def test_models_are_loaded_per_request_not_kept_resident(monkeypatch):
    """All 7 models resident measured ~80MB on Render's 512MB free tier —
    each request loads them one at a time and drops them again."""
    _install_sufficient_fetches(monkeypatch)
    _install_models(monkeypatch)
    loaded: list[int] = []
    fake_load = svc._load_horizon_model
    monkeypatch.setattr(svc, "_load_horizon_model", lambda h: loaded.append(h) or fake_load(h))

    asyncio.run(svc.get_forecast(_LAT, _LON))
    asyncio.run(svc.get_forecast(_LAT, _LON))

    assert loaded == list(range(1, 8)) * 2


def test_loaded_model_predicts_single_threaded(monkeypatch, tmp_path):
    class _TrainedModel:
        n_jobs = -1

    (tmp_path / "forecast_day1.pkl").write_bytes(b"")
    monkeypatch.setattr(svc, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(svc.joblib, "load", lambda _path: {"model": _TrainedModel(), "features": _FEATURES})

    assert svc._load_horizon_model(1)["model"].n_jobs == 1
