"""Offline unit tests for calibration math (no network required)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from polycal.calibration import (
    bin_and_aggregate,
    per_lead_mae,
    theoretical_curve,
    wilson_ci,
)


def test_wilson_ci_extremes():
    lo, hi = wilson_ci(0, 0)
    assert (lo, hi) == (0.0, 1.0)

    lo, hi = wilson_ci(10, 10)
    assert lo > 0.6 and hi == 1.0

    lo, hi = wilson_ci(5, 10)
    assert lo < 0.5 < hi


def test_bin_and_aggregate_basic():
    rng = np.random.default_rng(0)
    n = 5000
    spread = rng.uniform(-5, 5, n)
    # Truth: yes_won = 1 iff spread + N(0, 1) > 0.
    noise = rng.normal(0, 1, n)
    yes = (spread + noise > 0).astype(int)
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="h").date,
        "lead_time": 24,
        "threshold": 70.0,
        "spread": spread,
        "yes_won": yes,
        "forecast_high": 70.0 + spread,
        "actual_high": 70.0,
    })
    cal = bin_and_aggregate(df)
    # Around spread=0 the empirical rate should be ~0.5.
    near_zero = cal[(cal["bin_center"] > -0.5) & (cal["bin_center"] < 0.5)]
    assert 0.35 < near_zero["p_hat"].mean() < 0.65
    # Large positive spread → near 1.
    far = cal[cal["bin_center"] > 3.5]
    assert far["p_hat"].mean() > 0.9


def test_theoretical_curve_monotone():
    xs, ys = theoretical_curve(mae_f=1.0)
    assert ys[0] < 0.05 and ys[-1] > 0.95
    assert np.all(np.diff(ys) >= -1e-9)


def test_per_lead_mae():
    df = pd.DataFrame({
        "date": [pd.Timestamp("2024-01-01").date(), pd.Timestamp("2024-01-02").date()],
        "lead_time": [24, 24],
        "threshold": [70, 70],
        "spread": [0, 0],
        "yes_won": [1, 0],
        "forecast_high": [72.0, 68.0],
        "actual_high": [70.0, 70.0],
    })
    mae = per_lead_mae(df)
    assert math.isclose(mae[24], 2.0)
