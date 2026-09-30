"""Coverage for backend/services/forecast_targets.py — the target encoding
shared by forecast training and serving.

Run: PYTHONPATH=. pytest backend/tests/test_forecast_targets.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from backend.services import forecast_targets as ft

_Y = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 60.0]])
_DAY0 = np.array([[0.5, 12.0], [2.5, 18.0], [2.0, 25.0], [3.5, 40.0]])


@pytest.mark.parametrize("kind", ft.KINDS)
def test_encode_then_decode_round_trips_to_the_real_targets(kind):
    encoded, transform = ft.fit_encode(_Y, _DAY0, kind)
    assert transform["kind"] == kind
    np.testing.assert_allclose(ft.decode(encoded, _DAY0, transform), _Y)


def test_standardized_targets_have_zero_mean_unit_std_on_training_rows():
    encoded, transform = ft.fit_encode(_Y, _DAY0, "standardized")
    np.testing.assert_allclose(encoded.mean(axis=0), [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(encoded.std(axis=0), [1.0, 1.0])
    np.testing.assert_allclose(transform["mean"], _Y.mean(axis=0))


def test_residual_encodes_the_change_from_day0_and_decoding_adds_day0_back():
    _encoded, transform = ft.fit_encode(_Y, _DAY0, "residual")
    np.testing.assert_allclose(transform["mean"], (_Y - _DAY0).mean(axis=0))
    # A prediction of exactly the mean residual decodes to day0 + mean.
    day0 = np.array([[1.0, 15.0]])
    decoded = ft.decode(np.array([[0.0, 0.0]]), day0, transform)
    np.testing.assert_allclose(decoded, day0 + np.array(transform["mean"]))


def test_decode_uses_the_stored_training_stats_not_the_new_rows():
    _encoded, transform = ft.fit_encode(_Y, _DAY0, "standardized")
    decoded = ft.decode(np.array([[1.0, -1.0]]), np.array([[99.0, 99.0]]), transform)
    expected = np.array([[_Y[:, 0].mean() + _Y[:, 0].std(), _Y[:, 1].mean() - _Y[:, 1].std()]])
    np.testing.assert_allclose(decoded, expected)


def test_a_missing_transform_decodes_as_raw():
    pred = np.array([[1.5, 20.0]])
    np.testing.assert_allclose(ft.decode(pred, np.array([[9.0, 9.0]]), None), pred)


def test_a_constant_target_does_not_divide_by_zero():
    y = np.array([[1.0, 10.0], [1.0, 20.0]])
    encoded, transform = ft.fit_encode(y, np.zeros_like(y), "standardized")
    assert np.all(np.isfinite(encoded))
    np.testing.assert_allclose(ft.decode(encoded, np.zeros_like(y), transform), y)


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError):
        ft.fit_encode(_Y, _DAY0, "log")
    with pytest.raises(ValueError):
        ft.decode(_Y, _DAY0, {"kind": "log"})
