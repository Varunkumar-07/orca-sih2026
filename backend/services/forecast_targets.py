"""
Target encoding for the 7-day wave/wind forecast models — the one place
both training (backend/scripts/train_forecast_models.py) and serving
(forecast_service.py) turn real targets into what a model is fit on, and
back. Shared so the two can never drift apart.

Why targets are encoded at all: each horizon is ONE multi-output
RandomForest predicting wave height (metres, errors ~0.2) and wind speed
(km/h, errors ~3) together. Its split criterion sums squared error across
outputs in their raw units, so wind's much larger numbers dominate every
split and wave height is barely optimized. Encoding puts both targets on
the same scale.

Kinds (stored with each model artifact as its "target_transform"):
  - "raw": no encoding (the original models; kept so an artifact without a
    transform still decodes correctly)
  - "standardized": z-score per target, using training-set mean/std
  - "residual": the change from day 0 (value(day+h) - value(day 0)), then
    z-scored the same way; day 0's own value is added back when decoding

Day 0's values are the model's own "wave_height_m" / "wind_kmh" input
features — the same names as the targets, in TARGETS order.
"""

from __future__ import annotations

import numpy as np

TARGETS = ("wave_height_m", "wind_kmh")
KINDS = ("raw", "standardized", "residual")


def fit_encode(y: np.ndarray, day0: np.ndarray, kind: str) -> tuple[np.ndarray, dict]:
    """(encoded targets, transform) for training targets `y` (n x 2, TARGETS
    order) and day 0's values `day0` (same shape). The transform's mean/std
    come from these training rows only."""
    y = np.asarray(y, dtype=float)
    day0 = np.asarray(day0, dtype=float)
    if kind == "raw":
        return y, {"kind": "raw", "targets": list(TARGETS)}
    if kind not in KINDS:
        raise ValueError(f"unknown target transform kind {kind!r}")
    base = y - day0 if kind == "residual" else y
    mean = base.mean(axis=0)
    std = base.std(axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    transform = {"kind": kind, "targets": list(TARGETS), "mean": mean.tolist(), "std": std.tolist()}
    return (base - mean) / std, transform


def decode(pred: np.ndarray, day0: np.ndarray, transform: dict | None) -> np.ndarray:
    """Model output (n x 2, encoded) -> real wave height (m) / wind (km/h).
    A missing transform means a "raw" model."""
    pred = np.asarray(pred, dtype=float)
    if not transform or transform.get("kind") == "raw":
        return pred
    kind = transform["kind"]
    if kind not in KINDS:
        raise ValueError(f"unknown target transform kind {kind!r}")
    out = pred * np.asarray(transform["std"], dtype=float) + np.asarray(transform["mean"], dtype=float)
    if kind == "residual":
        out = out + np.asarray(day0, dtype=float)
    return out
