"""
One-time (periodically re-runnable) training script for ORCA's own
multi-horizon wave-height / wind-speed forecast models — backing
GET /weather/forecast.

Why this script exists instead of reusing A.T.M.O.S.'s already-trained
models: investigated directly (see the phase report) — ATMOS's 7
RandomForestRegressor models (weather_model_day1..7.pkl) are fit on a
SINGLE fixed inland location (Bangalore, 12.9716/77.5946), with no
location feature anywhere in their input, predicting only air temperature.
Applying those fitted trees to a different location's feature values (as
ATMOS's own /predict endpoint already does for every non-Bangalore city)
just replays Bangalore's specific climate patterns — not a genuine
generalization, and worse for ORCA specifically since these zones are
ocean/coastal points with no marine variable in ATMOS's models at all.
Direct reuse is not honestly viable; see the phase report for the full
investigation.

This script instead follows ATMOS's own architecture — direct (not
recursive) per-horizon RandomForestRegressor models, trained on real day-0
feature values only — but:
  - predicts wave_height_m and wind_kmh (the two most safety-relevant
    "should I go out fishing" variables), both sourced from Open-Meteo
    with genuinely deep history (unlike chlorophyll's confirmed ~17-day
    rolling window from the Analytics phase, or salinity/dissolved oxygen,
    which would need one slow Copernicus SDK call per training row);
  - trains across MULTIPLE zones at once, with lat/lon as an explicit
    input feature, so this ONE model family generalizes across every
    zone ORCA tracks, instead of needing (or silently mis-applying) one
    fixed-location model the way ATMOS does.

Data source: the exact same Open-Meteo marine + weather archive endpoints
analytics_service.py already calls for GET /analytics/historical — same
variable names, same units, so the serving-time feature-building code can
stay consistent with what these models were actually trained on.

Confirmed directly against the live API before writing this: the marine
archive's real (non-null) data only starts around October 2021 — nowhere
near ATMOS's 8-year window — so training covers ~4 years across 11
coastal anchor points instead.

Usage (from the project root):
    python -m backend.scripts.train_forecast_models
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from backend.agents.reasoning.planning_agent import _KNOWN_LOCATIONS

_MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"
_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
_HTTP_TIMEOUT = 30.0

# Same 11 anchors GET /zones already uses — a representative spread across
# the whole Indian coastline, not just one city.
_ANCHOR_NAMES = [
    "kandla", "mumbai", "goa", "mangaluru", "kochi", "chennai",
    "gulf of mannar", "visakhapatnam", "paradip", "kolkata", "port blair",
]

# A safe buffer past the confirmed ~Oct 2021 real-data start of the marine
# archive. End date stays 2 days before "today" to avoid very recent,
# not-yet-stable archive rows. "Today" is IST's today, not the CI runner's
# host timezone — see backend/services/weather_service.py's own _IST.
_IST = timezone(timedelta(hours=5, minutes=30))
_TRAIN_START = "2021-11-01"
_TRAIN_END = (datetime.now(_IST).date() - timedelta(days=2)).isoformat()

HORIZONS = range(1, 8)

FEATURE_COLS = [
    "lat", "lon", "month_sin", "month_cos",
    "wave_height_m", "wave_height_lag1",
    "wind_kmh", "wind_kmh_lag1",
    "sst_celsius", "air_temp_celsius",
]
TARGET_COLS = ["wave_height_m", "wind_kmh"]

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def _fetch_marine_daily(client: httpx.Client, lat: float, lon: float) -> dict:
    resp = client.get(
        _MARINE_URL,
        params={
            "latitude": lat, "longitude": lon,
            "daily": "wave_height_max,sea_surface_temperature_max",
            "start_date": _TRAIN_START, "end_date": _TRAIN_END, "timezone": "UTC",
        },
    )
    resp.raise_for_status()
    return resp.json()["daily"]


def _fetch_archive_daily(client: httpx.Client, lat: float, lon: float) -> dict:
    resp = client.get(
        _ARCHIVE_URL,
        params={
            "latitude": lat, "longitude": lon,
            "daily": "wind_speed_10m_max,temperature_2m_mean",
            "start_date": _TRAIN_START, "end_date": _TRAIN_END, "timezone": "UTC",
        },
    )
    resp.raise_for_status()
    return resp.json()["daily"]


def _build_location_frame(name: str, lat: float, lon: float, marine: dict, archive: dict) -> pd.DataFrame:
    marine_df = pd.DataFrame({
        "date": pd.to_datetime(marine["time"]),
        "wave_height_m": marine["wave_height_max"],
        "sst_celsius": marine["sea_surface_temperature_max"],
    })
    archive_df = pd.DataFrame({
        "date": pd.to_datetime(archive["time"]),
        "wind_kmh": archive["wind_speed_10m_max"],
        "air_temp_celsius": archive["temperature_2m_mean"],
    })
    df = marine_df.merge(archive_df, on="date", how="inner")
    df["lat"] = lat
    df["lon"] = lon
    df["anchor"] = name
    df = df.sort_values("date").reset_index(drop=True)
    # Real values only — a gap here would poison the lag/target shift below
    # with silently-wrong neighbors.
    df = df.dropna(subset=["wave_height_m", "wind_kmh"]).reset_index(drop=True)
    df["wave_height_lag1"] = df["wave_height_m"].shift(1)
    df["wind_kmh_lag1"] = df["wind_kmh"].shift(1)
    df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12)
    return df


def _build_horizon_dataset(frames: list[pd.DataFrame], horizon: int) -> pd.DataFrame:
    """Per-location target shift (never crossing an anchor's own
    timeline), concatenated after — direct (non-recursive) prediction,
    same convention as ATMOS's own per-horizon models."""
    parts = []
    for df in frames:
        d = df.copy()
        d["target_wave_height_m"] = d["wave_height_m"].shift(-horizon)
        d["target_wind_kmh"] = d["wind_kmh"].shift(-horizon)
        parts.append(d)
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.dropna(subset=FEATURE_COLS + ["target_wave_height_m", "target_wind_kmh"])
    # Global chronological sort (not per-anchor) so the later train/test
    # split is a genuine date cutoff across every zone at once, not just
    # within one anchor's own timeline.
    return combined.sort_values("date").reset_index(drop=True)


def main() -> None:
    print(f"Fetching {_TRAIN_START} .. {_TRAIN_END} for {len(_ANCHOR_NAMES)} anchors…")
    frames: list[pd.DataFrame] = []
    with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
        for name in _ANCHOR_NAMES:
            loc = _KNOWN_LOCATIONS[name]
            marine = _fetch_marine_daily(client, loc.lat, loc.lon)
            archive = _fetch_archive_daily(client, loc.lat, loc.lon)
            df = _build_location_frame(name, loc.lat, loc.lon, marine, archive)
            print(f"  {name:20s} {len(df):4d} usable days")
            frames.append(df)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics: dict[str, dict] = {}
    all_importance: dict[str, dict] = {}

    for horizon in HORIZONS:
        data = _build_horizon_dataset(frames, horizon)
        X = data[FEATURE_COLS]
        y = data[["target_wave_height_m", "target_wind_kmh"]]

        split_idx = int(len(data) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # n_estimators matches ATMOS's own models; max_depth/min_samples_leaf
        # are NOT in ATMOS's version but are necessary here — an unbounded
        # RandomForest on this row count produced ~200MB *per horizon*
        # (~1.4GB total, and over GitHub's 100MB single-file limit) for a
        # negligible accuracy gain over a depth-capped forest. Capping tree
        # depth/leaf size keeps the model honest-sized without materially
        # changing what it learned.
        model = RandomForestRegressor(
            n_estimators=100, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae_wave = mean_absolute_error(y_test["target_wave_height_m"], y_pred[:, 0])
        rmse_wave = np.sqrt(mean_squared_error(y_test["target_wave_height_m"], y_pred[:, 0]))
        mae_wind = mean_absolute_error(y_test["target_wind_kmh"], y_pred[:, 1])
        rmse_wind = np.sqrt(mean_squared_error(y_test["target_wind_kmh"], y_pred[:, 1]))

        key = f"day{horizon}"
        all_metrics[key] = {
            "wave_height_m": {"MAE": round(float(mae_wave), 3), "RMSE": round(float(rmse_wave), 3)},
            "wind_kmh": {"MAE": round(float(mae_wind), 2), "RMSE": round(float(rmse_wind), 2)},
            "train_rows": len(X_train),
            "test_rows": len(X_test),
        }
        all_importance[key] = {f: round(float(imp), 4) for f, imp in zip(FEATURE_COLS, model.feature_importances_)}

        joblib.dump(
            {"model": model, "features": FEATURE_COLS, "targets": TARGET_COLS},
            MODELS_DIR / f"forecast_day{horizon}.pkl",
        )
        print(
            f"day+{horizon}: wave MAE={mae_wave:.3f}m RMSE={rmse_wave:.3f}m | "
            f"wind MAE={mae_wind:.2f}km/h RMSE={rmse_wind:.2f}km/h "
            f"(train={len(X_train)}, test={len(X_test)})"
        )

    (MODELS_DIR / "forecast_metrics.json").write_text(json.dumps(all_metrics, indent=2))
    (MODELS_DIR / "forecast_feature_importance.json").write_text(json.dumps(all_importance, indent=2))
    print(f"\nSaved 7 models + metrics + feature importance to {MODELS_DIR}")


if __name__ == "__main__":
    main()
