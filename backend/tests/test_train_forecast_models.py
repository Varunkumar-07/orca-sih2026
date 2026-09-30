"""Coverage for the pure evaluation pieces of
backend/scripts/train_forecast_models.py: the date-based split and
calendar-day target/lag joins, target-encoding selection, the
persistence and climatology baselines, the per-horizon metrics entry
(model vs baselines on the same test rows, plus skill) and the _meta block.

Synthetic data only — no Open-Meteo fetch, no model training.

Run: PYTHONPATH=. pytest backend/tests/test_train_forecast_models.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.scripts import train_forecast_models as tfm


def _rows(anchor: str, dates: list[str], wave: list[float], wind: list[float], target_wave=None, target_wind=None):
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "anchor": anchor,
            "wave_height_m": wave,
            "wind_kmh": wind,
            "target_wave_height_m": target_wave if target_wave is not None else wave,
            "target_wind_kmh": target_wind if target_wind is not None else wind,
        }
    )


def _frame(anchor: str, dates: list[str], wave: list[float], wind: list[float]) -> pd.DataFrame:
    """A per-anchor frame shaped like _build_location_frame's output."""
    n = len(dates)
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "anchor": anchor,
            "lat": 13.0,
            "lon": 80.0,
            "month_sin": 0.0,
            "month_cos": 1.0,
            "wave_height_m": wave,
            "wave_height_lag1": [0.5] * n,
            "wind_kmh": wind,
            "wind_kmh_lag1": [5.0] * n,
            "sst_celsius": [28.0] * n,
            "air_temp_celsius": [29.0] * n,
        }
    )


def test_target_is_h_calendar_days_later_not_h_rows_later():
    """2024-01-03 is missing: day0=01-02 with h=2 targets 01-04 (a real
    day), and day0=01-01 with h=2 targets 01-03 (missing) -> dropped, not
    silently given 01-04's value the way a row shift would."""
    frame = _frame("goa", ["2024-01-01", "2024-01-02", "2024-01-04", "2024-01-05"], [1, 2, 4, 5], [10, 20, 40, 50])
    data = tfm._build_horizon_dataset([frame], horizon=2)
    by_date = {d.date().isoformat(): row for d, row in zip(data["date"], data.itertuples())}
    assert "2024-01-01" not in by_date
    assert by_date["2024-01-02"].target_wave_height_m == 4
    assert (data["target_date"] - data["date"]).dt.days.eq(2).all()


def test_lag1_is_the_previous_calendar_day_not_the_previous_row():
    marine = {
        "time": ["2024-01-01", "2024-01-02", "2024-01-04"],
        "wave_height_max": [1.0, 2.0, 4.0],
        "sea_surface_temperature_max": [28.0, 28.0, 28.0],
    }
    archive = {
        "time": ["2024-01-01", "2024-01-02", "2024-01-04"],
        "wind_speed_10m_max": [10.0, 20.0, 40.0],
        "temperature_2m_mean": [29.0, 29.0, 29.0],
    }
    df = tfm._build_location_frame("goa", 15.0, 74.0, marine, archive)
    lag = dict(zip(df["date"].dt.day, df["wave_height_lag1"]))
    assert lag[2] == 1.0
    assert np.isnan(lag[4])  # 01-03 is missing, so there's no real lag


def test_date_split_uses_one_cutoff_and_no_date_is_on_both_sides():
    dates = pd.date_range("2024-01-01", periods=20).strftime("%Y-%m-%d").tolist()
    frames = [_frame(a, dates, list(range(20)), list(range(20))) for a in ("goa", "kochi")]
    data = tfm._build_horizon_dataset(frames, horizon=3)
    cutoff = tfm.date_cutoff(data["date"], 0.8)
    train, test = tfm.split_by_date(data, cutoff)

    assert set(train["date"]).isdisjoint(set(test["date"]))
    assert train["date"].max() < cutoff <= test["date"].min()
    # No training label comes from the test period either.
    assert train["target_date"].max() < cutoff
    # The cutoff is the same for every anchor.
    assert test.groupby("anchor")["date"].min().nunique() == 1


def test_choose_transform_picks_the_lowest_validation_score_among_candidates_only():
    val_mae = {
        "day1": {
            "raw": {"wave_height_m": 0.01, "wind_kmh": 0.1},  # best, but reference only
            "standardized": {"wave_height_m": 0.2, "wind_kmh": 2.0},
            "residual": {"wave_height_m": 0.1, "wind_kmh": 2.2},
            "persistence": {"wave_height_m": 0.2, "wind_kmh": 2.0},
        }
    }
    chosen, scores = tfm.choose_transform(val_mae)
    # standardized: (1.0 + 1.0) / 2 = 1.0; residual: (0.5 + 1.1) / 2 = 0.8
    assert chosen == "residual"
    assert scores["standardized"] == 1.0
    assert scores["residual"] == 0.8
    assert "raw" in scores


def test_fit_and_predict_round_trip_real_units_for_every_candidate():
    """A tiny end-to-end fit: whatever the encoding, predictions come back
    in real units close to an easy-to-learn target."""
    dates = pd.date_range("2024-01-01", periods=60).strftime("%Y-%m-%d").tolist()
    wave = [1.0 + 0.01 * i for i in range(60)]
    wind = [10.0 + 0.1 * i for i in range(60)]
    data = tfm._build_horizon_dataset([_frame("goa", dates, wave, wind)], horizon=1)
    for kind in tfm.CANDIDATE_TRANSFORMS:
        model, transform = tfm.fit_model(data, kind)
        assert transform["kind"] == kind
        pred = tfm.predict_model(model, transform, data)
        assert np.abs(pred[:, 0] - data["target_wave_height_m"]).mean() < 0.05
        assert np.abs(pred[:, 1] - data["target_wind_kmh"]).mean() < 0.5


def test_persistence_predicts_day_zero_values():
    test = _rows("goa", ["2024-06-01", "2024-06-02"], [1.1, 1.4], [10.0, 12.0], [2.0, 2.0], [20.0, 20.0])
    preds = tfm.persistence_predictions(test)
    assert preds["wave_height_m"].tolist() == [1.1, 1.4]
    assert preds["wind_kmh"].tolist() == [10.0, 12.0]


def test_climatology_is_per_anchor_per_target_month_mean_from_training_rows():
    train = pd.concat(
        [
            # horizon=1: 2023-05-31 targets June; 2023-06-10/20 target June too.
            _rows("goa", ["2023-05-31", "2023-06-10", "2023-06-20"], [0, 0, 0], [0, 0, 0], [1.0, 2.0, 3.0], [10.0, 20.0, 30.0]),
            _rows("kochi", ["2023-06-10"], [0], [0], [9.0], [90.0]),
        ]
    )
    test = _rows("goa", ["2024-06-15"], [0.0], [0.0], [100.0], [100.0])
    preds = tfm.climatology_predictions(train, test, horizon=1)
    assert preds["wave_height_m"].tolist() == [pytest.approx(2.0)]
    assert preds["wind_kmh"].tolist() == [pytest.approx(20.0)]


def test_climatology_never_uses_test_rows():
    """A huge outlier in the test set must not move its own baseline."""
    train = _rows("goa", ["2023-07-01", "2023-07-02"], [0, 0], [0, 0], [1.0, 1.0], [5.0, 5.0])
    test = _rows("goa", ["2024-07-01", "2024-07-02"], [0, 0], [0, 0], [1.0, 999.0], [5.0, 999.0])
    preds = tfm.climatology_predictions(train, test, horizon=1)
    assert preds["wave_height_m"].tolist() == [pytest.approx(1.0), pytest.approx(1.0)]


def test_climatology_falls_back_to_anchor_mean_for_a_month_missing_from_training():
    train = _rows("goa", ["2023-01-10", "2023-02-10"], [0, 0], [0, 0], [1.0, 3.0], [10.0, 30.0])
    test = _rows("goa", ["2024-08-10"], [0], [0], [5.0], [50.0])
    preds = tfm.climatology_predictions(train, test, horizon=1)
    assert preds["wave_height_m"].tolist() == [pytest.approx(2.0)]
    assert preds["wind_kmh"].tolist() == [pytest.approx(20.0)]


def test_climatology_uses_the_target_dates_month_not_day_zeros():
    """Day 0 on 2024-01-30 with horizon 3 targets February."""
    train = _rows("goa", ["2023-01-15", "2023-02-15"], [0, 0], [0, 0], [1.0, 7.0], [1.0, 7.0])
    test = _rows("goa", ["2024-01-30"], [0], [0], [0.0], [0.0])
    # Train rows' own target months: Jan 15 + 3 = Jan (1.0), Feb 15 + 3 = Feb (7.0).
    preds = tfm.climatology_predictions(train, test, horizon=3)
    assert preds["wave_height_m"].tolist() == [pytest.approx(7.0)]


def test_evaluate_horizon_scores_model_and_baselines_on_the_same_test_rows():
    train = _rows("goa", ["2023-06-01", "2023-06-02"], [1.0, 1.0], [10.0, 10.0], [1.0, 1.0], [10.0, 10.0])
    test = _rows("goa", ["2024-06-01", "2024-06-02"], [1.0, 3.0], [10.0, 30.0], [2.0, 2.0], [20.0, 20.0])
    # Model predicts the targets exactly.
    model_pred = np.array([[2.0, 20.0], [2.0, 20.0]])
    entry = tfm.evaluate_horizon(train, test, model_pred, horizon=1)

    assert entry["wave_height_m"] == {"MAE": 0.0, "RMSE": 0.0}
    assert entry["test_rows"] == 2 and entry["train_rows"] == 2
    assert entry["test_dates"] == ["2024-06-01", "2024-06-02"]
    # Persistence: |1-2|, |3-2| -> MAE 1.0 wave; |10-20|, |30-20| -> 10.0 wind.
    assert entry["baselines"]["persistence"]["wave_height_m"]["MAE"] == 1.0
    assert entry["baselines"]["persistence"]["wind_kmh"]["MAE"] == 10.0
    # Climatology: June training mean 1.0 / 10.0 -> MAE 1.0 / 10.0.
    assert entry["baselines"]["climatology"]["wave_height_m"]["MAE"] == 1.0
    # Perfect model -> skill 1.0 against both.
    assert entry["skill"]["vs_persistence"]["wave_height_m"] == 1.0
    assert entry["skill"]["vs_climatology"]["wind_kmh"] == 1.0


def test_skill_is_negative_when_the_model_is_worse_than_a_baseline():
    train = _rows("goa", ["2023-06-01"], [2.0], [20.0], [2.0], [20.0])
    test = _rows("goa", ["2024-06-01"], [2.0], [20.0], [2.0], [20.0])
    entry = tfm.evaluate_horizon(train, test, np.array([[3.0, 30.0]]), horizon=1)
    # Persistence is exact here (MAE 0) -> skill undefined, reported as None.
    assert entry["skill"]["vs_persistence"]["wave_height_m"] is None
    train2 = _rows("goa", ["2023-06-01"], [0.0], [0.0], [2.5], [25.0])
    entry2 = tfm.evaluate_horizon(train2, test, np.array([[3.0, 30.0]]), horizon=1)
    # Climatology MAE 0.5 vs model MAE 1.0 -> skill 1 - 1.0/0.5 = -1.0.
    assert entry2["skill"]["vs_climatology"]["wave_height_m"] == -1.0


def test_build_meta_records_window_anchors_split_and_model():
    frames = {
        "goa": pd.DataFrame({"date": pd.to_datetime(["2021-11-01", "2021-11-02"]), "anchor": "goa"}),
        "kochi": pd.DataFrame({"date": pd.to_datetime(["2021-11-03"]), "anchor": "kochi"}),
        "kolkata": pd.DataFrame({"date": pd.to_datetime([]), "anchor": []}),
    }
    points = {"goa": (15.3, 74.1), "kochi": (9.9, 76.3), "kolkata": (22.6, 88.4)}
    meta = tfm.build_meta(frames, points, "2026-09-30T00:00:00Z")
    assert meta["trained_at"] == "2026-09-30T00:00:00Z"
    assert meta["data_window"]["first_date"] == "2021-11-01"
    assert meta["data_window"]["last_date"] == "2021-11-03"
    assert meta["anchors"]["goa"] == {"lat": 15.3, "lon": 74.1, "usable_days": 2}
    # An anchor with no usable data is recorded, not dropped.
    assert meta["anchors"]["kolkata"]["usable_days"] == 0
    assert meta["anchors_used"] == 2
    assert meta["total_usable_days"] == 3
    assert meta["split"]["train_fraction_of_dates"] == tfm.TRAIN_FRACTION
    assert meta["model"]["n_estimators"] == 100
    assert meta["features"] == tfm.FEATURE_COLS


def test_climatology_handles_object_dtype_targets_from_empty_anchor_frames():
    """Concatenating an empty anchor frame leaves target columns as object
    dtype — the real training run hit this (Goa and Kolkata return no
    usable archive days)."""
    train = _rows("goa", ["2023-06-01", "2023-06-02"], [0, 0], [0, 0], [1.0, 3.0], [10.0, 30.0]).astype(
        {"target_wave_height_m": object, "target_wind_kmh": object}
    )
    test = _rows("goa", ["2024-06-01"], [0], [0], [0.0], [0.0])
    preds = tfm.climatology_predictions(train, test, horizon=1)
    assert preds["wave_height_m"].tolist() == [pytest.approx(2.0)]
