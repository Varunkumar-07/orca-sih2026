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

Data actually used (checked on the 2026-09-30 run, see _meta in
forecast_metrics.json for each run's own numbers): Goa's and Kolkata's
gazetteer points return no marine-archive data (Kolkata is inland), so 9
of the 11 anchors contribute rows; and daily SST is missing for ~400 days
before early 2025, so complete rows start around late 2022.

Train/validation/test: one cutoff DATE for all anchors (never a row
index), and a training row is only kept if its target date is also before
the cutoff, so no date's data is on both sides. The last ~15% of the
training period (by date) is a validation set used ONLY to choose the
target encoding (see backend/services/forecast_targets.py — raw,
standardized, or residual); the chosen encoding is then refit on the whole
training period and scored on the test set once.

Every run also scores two reference forecasts on the exact same test rows
as the model, so its error has something to be compared against:
  - persistence: day+h = day 0's own value ("tomorrow = today")
  - climatology: the mean of the target for that anchor and the target
    date's calendar month, computed from TRAINING rows only (falling back to
    the anchor's overall training mean for a month it has no rows for)
and writes a skill score per day and target (1 - model MAE / baseline MAE;
above 0 means the model beats that baseline). forecast_metrics.json's
top-level "_meta" block records when the run happened, the data window
actually used, row counts per anchor, the split and the model settings.

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
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from backend.agents.reasoning.planning_agent import _KNOWN_LOCATIONS
from backend.services.forecast_targets import decode, fit_encode

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

# Date-based split: dates before the TRAIN_FRACTION point of all distinct
# dates train, the rest test. Inside the training period, the last
# (1 - FIT_FRACTION) of dates is the validation set for choosing the
# target encoding.
TRAIN_FRACTION = 0.8
FIT_FRACTION = 0.85
# Encodings compared on validation. "raw" (the original models) is scored
# for reference only; the choice is between the two encoded variants.
CANDIDATE_TRANSFORMS = ("standardized", "residual")
REFERENCE_TRANSFORMS = ("raw",)
MODEL_PARAMS = {"n_estimators": 100, "max_depth": 12, "min_samples_leaf": 5, "random_state": 42}
BASELINES = ("persistence", "climatology")

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


def _shifted_by_days(df: pd.DataFrame, cols: list[str], days: int, suffix: str) -> pd.DataFrame:
    """`cols` from the row dated `days` calendar days after each row's own
    date (negative = before), joined on date — so a missing day yields NaN
    rather than silently borrowing a neighbouring row's value."""
    other = df[["date", *cols]].copy()
    other["date"] = other["date"] - pd.Timedelta(days=days)
    other = other.rename(columns={c: f"{c}{suffix}" for c in cols})
    return df.merge(other, on="date", how="left")


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
    # Real values only — a missing day becomes a NaN lag/target below (the
    # joins are by calendar date), never a neighbouring day's value.
    df = df.dropna(subset=["wave_height_m", "wind_kmh"]).reset_index(drop=True)
    df = _shifted_by_days(df, ["wave_height_m", "wind_kmh"], -1, "_lag1")
    df = df.rename(columns={"wave_height_m_lag1": "wave_height_lag1"})
    df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12)
    return df


def _build_horizon_dataset(frames: list[pd.DataFrame], horizon: int) -> pd.DataFrame:
    """Per-anchor target = the value exactly `horizon` calendar days after
    day 0 (joined on date, never crossing an anchor's own timeline) —
    direct (non-recursive) prediction, same convention as ATMOS's own
    per-horizon models. Rows with any missing feature or target are
    dropped."""
    parts = []
    for df in frames:
        if not len(df):
            continue
        d = _shifted_by_days(df, TARGET_COLS, horizon, "_target")
        d = d.rename(columns={f"{t}_target": f"target_{t}" for t in TARGET_COLS})
        d["target_date"] = d["date"] + pd.Timedelta(days=horizon)
        parts.append(d)
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.dropna(subset=FEATURE_COLS + [f"target_{t}" for t in TARGET_COLS])
    for col in FEATURE_COLS + [f"target_{t}" for t in TARGET_COLS]:
        combined[col] = combined[col].astype(float)
    return combined.sort_values(["date", "anchor"]).reset_index(drop=True)


def date_cutoff(dates: pd.Series, fraction: float) -> pd.Timestamp:
    """The first date of the last (1 - fraction) of `dates`' distinct dates."""
    unique = np.sort(pd.to_datetime(dates).unique())
    return pd.Timestamp(unique[int(len(unique) * fraction)])


def split_by_date(data: pd.DataFrame, cutoff: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, test) at one calendar-date cutoff for every anchor. A train
    row needs both its own date and its target date before the cutoff (so
    no label comes from the test period); a test row's date is on or after
    it. No date contributes to both sides."""
    train = data[data["target_date"] < cutoff]
    test = data[data["date"] >= cutoff]
    return train, test


def _y(df: pd.DataFrame) -> np.ndarray:
    return df[[f"target_{t}" for t in TARGET_COLS]].to_numpy(dtype=float)


def _day0(df: pd.DataFrame) -> np.ndarray:
    return df[TARGET_COLS].to_numpy(dtype=float)


def fit_model(train: pd.DataFrame, kind: str) -> tuple[RandomForestRegressor, dict]:
    """One multi-output forest on `kind`-encoded targets (see
    backend/services/forecast_targets.py). Same tree settings for every
    kind, so file size and memory don't depend on it.

    n_estimators matches ATMOS's own models; max_depth/min_samples_leaf are
    NOT in ATMOS's version but are necessary here — an unbounded forest on
    this row count produced ~200MB *per horizon* (~1.4GB total, and over
    GitHub's 100MB single-file limit) for a negligible accuracy gain."""
    y_enc, transform = fit_encode(_y(train), _day0(train), kind)
    model = RandomForestRegressor(**MODEL_PARAMS, n_jobs=-1)
    model.fit(train[FEATURE_COLS], y_enc)
    return model, transform


def predict_model(model, transform: dict, df: pd.DataFrame) -> np.ndarray:
    """Real-unit predictions (n x 2, TARGET_COLS order) — the same decode
    forecast_service.py applies at serving time."""
    return decode(model.predict(df[FEATURE_COLS]), _day0(df), transform)


def validation_score(val_mae: dict[str, dict[str, dict[str, float]]], kind: str) -> float:
    """One number to choose an encoding by: the mean, over every horizon and
    target, of validation MAE / validation persistence MAE (scale-free, so
    wind's km/h can't outweigh wave's metres). Lower is better."""
    ratios = [
        day[kind][t] / day["persistence"][t]
        for day in val_mae.values()
        for t in TARGET_COLS
        if day["persistence"][t] > 0
    ]
    return float(np.mean(ratios))


def choose_transform(val_mae: dict) -> tuple[str, dict[str, float]]:
    """(chosen kind, every scored kind's validation score). Only
    CANDIDATE_TRANSFORMS can be chosen; validation data only."""
    scores = {k: round(validation_score(val_mae, k), 4) for k in (*REFERENCE_TRANSFORMS, *CANDIDATE_TRANSFORMS)}
    chosen = min(CANDIDATE_TRANSFORMS, key=lambda k: scores[k])
    return chosen, scores


def _target_month(df: pd.DataFrame, horizon: int) -> pd.Series:
    return (df["date"] + pd.Timedelta(days=horizon)).dt.month


def persistence_predictions(test: pd.DataFrame) -> pd.DataFrame:
    """day+h predicted as day 0's own value, for each target."""
    return pd.DataFrame({t: test[t].to_numpy() for t in TARGET_COLS}, index=test.index)


def climatology_predictions(train: pd.DataFrame, test: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Per-anchor, per-calendar-month (of the target date) mean of each
    target, from `train` only — test rows never inform their own baseline.
    An (anchor, month) pair absent from training falls back to that
    anchor's overall training mean."""
    train_month = _target_month(train, horizon)
    test_month = _target_month(test, horizon)
    preds = {}
    for t in TARGET_COLS:
        # float cast: an anchor with no usable rows contributes an empty
        # frame, and concatenating that can leave the column dtype "object".
        target = train[f"target_{t}"].astype(float)
        by_month = target.groupby([train["anchor"], train_month]).mean()
        by_anchor = target.groupby(train["anchor"]).mean()
        keys = pd.MultiIndex.from_arrays([test["anchor"], test_month])
        values = by_month.reindex(keys).to_numpy(dtype=float)
        fallback = test["anchor"].map(by_anchor).to_numpy(dtype=float)
        preds[t] = np.where(np.isnan(values), fallback, values)
    return pd.DataFrame(preds, index=test.index)


def score(y_true, y_pred) -> tuple[float, float]:
    """(MAE, RMSE)."""
    return float(mean_absolute_error(y_true, y_pred)), float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _rounded_scores(target: str, mae: float, rmse: float) -> dict:
    digits = 3 if target == "wave_height_m" else 2
    return {"MAE": round(mae, digits), "RMSE": round(rmse, digits)}


def _date_range(df: pd.DataFrame) -> list[str]:
    return [df["date"].min().date().isoformat(), df["date"].max().date().isoformat()]


def evaluate_horizon(train: pd.DataFrame, test: pd.DataFrame, model_pred: np.ndarray, horizon: int) -> dict:
    """forecast_metrics.json's entry for one horizon: the model's MAE/RMSE
    per target, both baselines scored on the same test rows, skill vs each
    baseline, and the split's row counts/dates."""
    entry: dict = {}
    model_mae: dict[str, float] = {}
    for i, t in enumerate(TARGET_COLS):
        mae, rmse = score(test[f"target_{t}"], model_pred[:, i])
        model_mae[t] = mae
        entry[t] = _rounded_scores(t, mae, rmse)
    entry["train_rows"] = len(train)
    entry["test_rows"] = len(test)
    entry["train_dates"] = _date_range(train)
    entry["test_dates"] = _date_range(test)

    baseline_preds = {
        "persistence": persistence_predictions(test),
        "climatology": climatology_predictions(train, test, horizon),
    }
    entry["baselines"] = {}
    entry["skill"] = {}
    for name in BASELINES:
        entry["baselines"][name] = {}
        entry["skill"][f"vs_{name}"] = {}
        for t in TARGET_COLS:
            mae, rmse = score(test[f"target_{t}"], baseline_preds[name][t])
            entry["baselines"][name][t] = _rounded_scores(t, mae, rmse)
            entry["skill"][f"vs_{name}"][t] = round(1 - model_mae[t] / mae, 3) if mae > 0 else None
    return entry


def validation_maes(fit: pd.DataFrame, val: pd.DataFrame, horizon: int) -> dict[str, dict[str, float]]:
    """Validation MAE per target for every scored encoding plus the
    persistence baseline (the normalizer in validation_score)."""
    out: dict[str, dict[str, float]] = {}
    for kind in (*REFERENCE_TRANSFORMS, *CANDIDATE_TRANSFORMS):
        model, transform = fit_model(fit, kind)
        pred = predict_model(model, transform, val)
        out[kind] = {t: round(score(val[f"target_{t}"], pred[:, i])[0], 4) for i, t in enumerate(TARGET_COLS)}
    persist = persistence_predictions(val)
    out["persistence"] = {t: round(score(val[f"target_{t}"], persist[t])[0], 4) for t in TARGET_COLS}
    return out


def build_meta(frames: dict[str, pd.DataFrame], anchor_points: dict[str, tuple[float, float]], trained_at: str) -> dict:
    """forecast_metrics.json's "_meta" block. `frames` and `anchor_points`
    are keyed by anchor name; an anchor whose archive returned no usable
    days is still listed, with usable_days 0, so its absence from training
    is recorded rather than silent."""
    all_dates = pd.concat([df["date"] for df in frames.values() if len(df)])
    return {
        "trained_at": trained_at,
        "data_window": {
            "requested_start": _TRAIN_START,
            "requested_end": _TRAIN_END,
            "first_date": all_dates.min().date().isoformat(),
            "last_date": all_dates.max().date().isoformat(),
        },
        "anchors": {
            name: {"lat": lat, "lon": lon, "usable_days": len(frames.get(name, ()))}
            for name, (lat, lon) in anchor_points.items()
        },
        "anchors_used": sum(1 for df in frames.values() if len(df)),
        "total_usable_days": int(sum(len(df) for df in frames.values())),
        "split": {
            "method": (
                "one calendar-date cutoff for all anchors; a training row's own date and target date "
                "are both before the cutoff, a test row's date is on or after it"
            ),
            "train_fraction_of_dates": TRAIN_FRACTION,
            "validation": f"last {1 - FIT_FRACTION:.0%} of the training period's dates, used only to choose the target encoding",
        },
        "model": {"type": "RandomForestRegressor (multi-output)", **MODEL_PARAMS},
        "features": FEATURE_COLS,
        "targets": TARGET_COLS,
        "baselines": {
            "persistence": "day+h = day 0's own value",
            "climatology": "per-anchor, per-target-month mean from training rows only (anchor mean if the month is missing)",
        },
        "skill_score": "1 - model MAE / baseline MAE (above 0: model beats the baseline)",
        "sklearn_version": sklearn.__version__,
        "data_source": "Open-Meteo marine archive + weather archive (daily)",
    }


def main() -> None:
    trained_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
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

    # Pass 1 — choose the target encoding on validation data only.
    datasets: dict[int, tuple[pd.DataFrame, pd.DataFrame, dict]] = {}
    val_mae: dict[str, dict] = {}
    for horizon in HORIZONS:
        data = _build_horizon_dataset(frames, horizon)
        test_cutoff = date_cutoff(data["date"], TRAIN_FRACTION)
        train, test = split_by_date(data, test_cutoff)
        val_cutoff = date_cutoff(train["date"], FIT_FRACTION)
        fit, val = split_by_date(train, val_cutoff)
        key = f"day{horizon}"
        val_mae[key] = validation_maes(fit, val, horizon)
        datasets[horizon] = (
            train,
            test,
            {
                "test_cutoff": test_cutoff.date().isoformat(),
                "validation_cutoff": val_cutoff.date().isoformat(),
                "fit_rows": len(fit),
                "validation_rows": len(val),
            },
        )
        print(f"{key} validation MAE: " + ", ".join(f"{k}={v}" for k, v in val_mae[key].items()))

    chosen, scores = choose_transform(val_mae)
    print(f"\nValidation scores (mean MAE / persistence MAE, lower is better): {scores} -> chosen: {chosen}\n")

    # Pass 2 — refit the chosen encoding on the whole training period and
    # score it on the test set, once.
    all_metrics: dict[str, dict] = {}
    all_importance: dict[str, dict] = {}
    for horizon in HORIZONS:
        train, test, split_info = datasets[horizon]
        model, transform = fit_model(train, chosen)
        key = f"day{horizon}"
        entry = evaluate_horizon(train, test, predict_model(model, transform, test), horizon)
        entry["split"] = split_info
        entry["validation_mae"] = val_mae[key]
        all_metrics[key] = entry
        all_importance[key] = {f: round(float(imp), 4) for f, imp in zip(FEATURE_COLS, model.feature_importances_)}

        joblib.dump(
            {"model": model, "features": FEATURE_COLS, "targets": TARGET_COLS, "target_transform": transform},
            MODELS_DIR / f"forecast_day{horizon}.pkl",
        )
        b = entry["baselines"]
        print(
            f"day+{horizon}: wave MAE model={entry['wave_height_m']['MAE']:.3f} "
            f"persist={b['persistence']['wave_height_m']['MAE']:.3f} clim={b['climatology']['wave_height_m']['MAE']:.3f}m | "
            f"wind MAE model={entry['wind_kmh']['MAE']:.2f} "
            f"persist={b['persistence']['wind_kmh']['MAE']:.2f} clim={b['climatology']['wind_kmh']['MAE']:.2f}km/h "
            f"(train={entry['train_rows']}, test={entry['test_rows']})"
        )

    anchor_points = {n: (_KNOWN_LOCATIONS[n].lat, _KNOWN_LOCATIONS[n].lon) for n in _ANCHOR_NAMES}
    meta = build_meta(dict(zip(_ANCHOR_NAMES, frames)), anchor_points, trained_at)
    meta["target_transform"] = {
        "chosen": chosen,
        "candidates": list(CANDIDATE_TRANSFORMS),
        "reference_only": list(REFERENCE_TRANSFORMS),
        "validation_scores": scores,
        "criterion": "mean over horizons and targets of validation MAE / validation persistence MAE; lowest wins",
    }
    all_metrics = {"_meta": meta, **all_metrics}
    (MODELS_DIR / "forecast_metrics.json").write_text(json.dumps(all_metrics, indent=2))
    (MODELS_DIR / "forecast_feature_importance.json").write_text(json.dumps(all_importance, indent=2))
    print(f"\nSaved 7 models + metrics + feature importance to {MODELS_DIR}")


if __name__ == "__main__":
    main()
