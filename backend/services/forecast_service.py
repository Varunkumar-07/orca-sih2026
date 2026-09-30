"""
Forecast serving — backing GET /weather/forecast.

Loads the 7 per-horizon models backend/scripts/train_forecast_models.py
produces (one at a time, per request — see _predict_horizons for why they
aren't kept resident) and predicts
wave_height_m / wind_kmh for day+1..day+7 at a given point, using that
point's OWN real day-0/day-1 conditions — never another location's
values, and never a previous horizon's predicted output fed back in
(every horizon model was trained on real values only; see the training
script's own docstring for why, matching ATMOS's direct multi-horizon
convention).

This is genuine model output — ORCA's own trained models (see the phase
report for why ATMOS's Bangalore-only models were not reused) — always
clearly labeled as a prediction, never presented as live data. Completely
separate from GET /weather (weather_service.py), which this module does
not call, touch, or share any caching with.

Feature-building reuses analytics_service's own Open-Meteo daily-fetch
functions (read-only import) rather than duplicating that HTTP logic, so
serving-time features are built from the exact same variable definitions
the models were trained on.
"""

from __future__ import annotations

import asyncio
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from backend.services import heavy_work
from backend.services.forecast_targets import TARGETS, decode
from backend.services.analytics_service import _fetch_archive_daily, _fetch_marine_daily
from backend.time_utils import now_iso as _now_iso

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
HORIZONS = range(1, 8)
# "Today" for the day0/lag date math must be IST's today, not the host
# server's — see backend/services/weather_service.py's own _IST for why.
_IST = timezone(timedelta(hours=5, minutes=30))

MODEL_DESCRIPTION = (
    "RandomForestRegressor (multi-output: wave height + wind speed), one model per "
    "forecast day, trained on ORCA's own ~4 years of Open-Meteo marine + weather "
    "archive history across 11 coastal anchor points — not reused from any other "
    "project's model, which would not generalize to these coordinates."
)

_metrics_cache: dict[str, dict] | None = None


def _load_metrics() -> dict[str, dict]:
    """The same validation-set MAE/RMSE train_forecast_models.py reported
    per horizon — read once and cached, so the API can report each
    prediction's real accuracy alongside it."""
    global _metrics_cache
    if _metrics_cache is None:
        path = MODELS_DIR / "forecast_metrics.json"
        try:
            _metrics_cache = json.loads(path.read_text())
        except Exception:
            _metrics_cache = {}
    return _metrics_cache



def _load_horizon_model(horizon: int) -> dict | None:
    path = MODELS_DIR / f"forecast_day{horizon}.pkl"
    if not path.exists():
        return None
    try:
        artifact = joblib.load(path)
    except Exception:
        return None
    # Trained with n_jobs=-1, which at predict time spins up a joblib
    # thread per core just to score one row. Same trees, same prediction.
    model = artifact.get("model")
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1
    return artifact


def _predict_horizons(row: pd.DataFrame) -> tuple[list[tuple[int, float, float]], int | None]:
    """Blocking — run via asyncio.to_thread. (horizon, wave, wind) for each
    horizon in order; stops at the first horizon whose model is missing
    and returns it as the second element (None when all were predicted).

    Memory, not speed, is why models are loaded per call and dropped
    before the next one is loaded, rather than all 7 cached for the
    process lifetime: the resident set measured ~80MB on top of the rest
    of the app, on Render free tier's 512MB. One ~8MB artifact loads in
    ~10ms (after the first load's one-time sklearn import), so the extra
    latency is small next to the Open-Meteo fetches this endpoint already
    makes."""
    preds: list[tuple[int, float, float]] = []
    try:
        for horizon in HORIZONS:
            artifact = _load_horizon_model(horizon)
            if artifact is None:
                return preds, horizon
            # Models are fit on encoded targets (standardized, or residuals
            # from day 0 — see forecast_targets.py); decode with the
            # artifact's own transform, using this row's day-0 values.
            encoded = artifact["model"].predict(row[artifact["features"]])
            pred = decode(encoded, row[list(TARGETS)].to_numpy(dtype=float), artifact.get("target_transform"))[0]
            preds.append((horizon, float(pred[0]), float(pred[1])))
            del artifact
        return preds, None
    finally:
        heavy_work.release_memory()


async def get_forecast(lat: float, lon: float) -> dict[str, Any]:
    """7-day wave height / wind speed forecast for one point. Never
    raises — degrades to status="error" with a clear reason (missing
    model artifacts, or insufficient recent real data for this location)
    rather than fabricating a prediction."""
    try:
        # "Yesterday" as day-0: the most recent day reliably settled in
        # both archive products (today's own data can still be partial).
        # The day before that supplies day-0's own lag1 feature.
        day0_date = datetime.now(_IST).date() - timedelta(days=1)
        lag_date = day0_date - timedelta(days=1)
        start_str, end_str = lag_date.isoformat(), day0_date.isoformat()

        marine, archive = await asyncio.gather(
            _fetch_marine_daily(lat, lon, start_str, end_str),
            _fetch_archive_daily(lat, lon, start_str, end_str),
        )

        wave_vals = marine.get("wave_height_max", [])
        sst_vals = marine.get("sea_surface_temperature_max", [])
        wind_vals = archive.get("wind_speed_10m_max", [])
        temp_vals = archive.get("temperature_2m_mean", [])

        if len(wave_vals) < 2 or len(wind_vals) < 2:
            return {
                "lat": lat, "lon": lon, "status": "error",
                "reason": "insufficient recent marine/weather data for this location to build a forecast input",
                "forecast": [], "generated_at": _now_iso(),
            }

        wave_lag1, wave_day0 = wave_vals[0], wave_vals[1]
        sst_day0 = sst_vals[1] if len(sst_vals) > 1 else None
        wind_lag1, wind_day0 = wind_vals[0], wind_vals[1]
        temp_day0 = temp_vals[1] if len(temp_vals) > 1 else None

        if any(v is None for v in (wave_day0, wave_lag1, wind_day0, wind_lag1, sst_day0, temp_day0)):
            return {
                "lat": lat, "lon": lon, "status": "error",
                "reason": "one or more required input variables (wave height, wind, SST, air temp) were unavailable for this location's most recent days",
                "forecast": [], "generated_at": _now_iso(),
            }

        month_sin = math.sin(2 * math.pi * day0_date.month / 12)
        month_cos = math.cos(2 * math.pi * day0_date.month / 12)
        row = pd.DataFrame([{
            "lat": lat, "lon": lon, "month_sin": month_sin, "month_cos": month_cos,
            "wave_height_m": wave_day0, "wave_height_lag1": wave_lag1,
            "wind_kmh": wind_day0, "wind_kmh_lag1": wind_lag1,
            "sst_celsius": sst_day0, "air_temp_celsius": temp_day0,
        }])

        preds, missing_horizon = await asyncio.to_thread(_predict_horizons, row)
        if missing_horizon is not None:
            return {
                "lat": lat, "lon": lon, "status": "error",
                "reason": f"forecast model for day+{missing_horizon} is not available (run backend/scripts/train_forecast_models.py)",
                "forecast": [], "generated_at": _now_iso(),
            }

        forecast: list[dict] = []
        for horizon, wave_pred, wind_pred in preds:
            metrics = _load_metrics().get(f"day{horizon}", {})
            forecast.append({
                "horizon": horizon,
                "date": (day0_date + timedelta(days=horizon)).isoformat(),
                "wave_height_m": round(wave_pred, 2),
                "wind_kmh": round(wind_pred, 1),
                # Same validation-set MAE train_forecast_models.py reported for
                # this horizon — a real accuracy figure, not a guess, so the
                # UI can show "typically ±X" alongside the prediction.
                "wave_height_mae": metrics.get("wave_height_m", {}).get("MAE"),
                "wind_kmh_mae": metrics.get("wind_kmh", {}).get("MAE"),
            })

        return {
            "lat": lat, "lon": lon, "status": "ok",
            "based_on_date": day0_date.isoformat(),
            "model": MODEL_DESCRIPTION,
            "forecast": forecast,
            "generated_at": _now_iso(),
        }
    except Exception as exc:  # noqa: BLE001 - must never raise across the boundary
        return {
            "lat": lat, "lon": lon, "status": "error",
            "reason": f"forecast failed: {exc}",
            "forecast": [], "generated_at": _now_iso(),
        }
