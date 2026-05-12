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
    xs, ys = theoretical_curve(sigma_f=1.0)
    assert ys[0] < 0.05 and ys[-1] > 0.95
    assert np.all(np.diff(ys) >= -1e-9)


def test_theoretical_curve_bias_shift():
    # Positive bias (forecast > actual on average) shifts the curve right:
    # need more positive spread to reach P=0.5.
    xs, ys0 = theoretical_curve(sigma_f=1.0, bias_f=0.0)
    _, ysp = theoretical_curve(sigma_f=1.0, bias_f=1.0)
    i_mid_0 = int(np.argmin(np.abs(ys0 - 0.5)))
    i_mid_p = int(np.argmin(np.abs(ysp - 0.5)))
    assert xs[i_mid_p] > xs[i_mid_0]


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


# ---------- market_calibration ----------

def test_snapshot_prices_picks_nearest_within_tolerance():
    import datetime as dt
    from polycal.market_calibration import peak_utc, snapshot_prices

    resolve_date = dt.date(2024, 6, 15)
    peak = peak_utc(resolve_date)
    # Hourly price history covering [peak - 48h, peak + 2h]. Lead times 1/6/24
    # fall inside, 72h does not — should be dropped by ±2h tolerance.
    ts = pd.date_range(peak - pd.Timedelta(hours=48), peak + pd.Timedelta(hours=2),
                       freq="h")
    prices = pd.DataFrame({
        "market_id": "m1",
        "ts_utc": ts,
        "p": np.linspace(0.5, 0.9, len(ts)),
        "resolve_date": resolve_date,
        "threshold_f": 85.0,
    })
    snaps = snapshot_prices(prices, lead_times=[1, 6, 24, 72], tolerance_hours=2.0)
    assert set(snaps["lead_time"]) == {1, 6, 24}
    # Snapshot timestamps should each be within 1h of their target (hourly grid).
    for _, row in snaps.iterrows():
        gap = abs((row["snap_ts_utc"] - row["target_ts_utc"]).total_seconds())
        assert gap <= 3600


def test_join_with_actuals_computes_yes_won():
    import datetime as dt
    from polycal.market_calibration import join_with_actuals

    snaps = pd.DataFrame([
        {"market_id": "m1", "resolve_date": dt.date(2024, 6, 1), "threshold_f": 80.0,
         "lead_time": 24, "target_ts_utc": pd.Timestamp("2024-06-01T17:00"),
         "snap_ts_utc": pd.Timestamp("2024-06-01T17:00"), "p_market": 0.7},
        {"market_id": "m2", "resolve_date": dt.date(2024, 6, 1), "threshold_f": 90.0,
         "lead_time": 24, "target_ts_utc": pd.Timestamp("2024-06-01T17:00"),
         "snap_ts_utc": pd.Timestamp("2024-06-01T17:00"), "p_market": 0.4},
    ])
    out = join_with_actuals(snaps, {dt.date(2024, 6, 1): 85.0})
    assert len(out) == 2
    assert out.loc[out["threshold_f"] == 80.0, "yes_won"].iloc[0] == 1
    assert out.loc[out["threshold_f"] == 90.0, "yes_won"].iloc[0] == 0


def test_bin_by_price_detects_overpricing():
    from polycal.market_calibration import bin_by_price

    rng = np.random.default_rng(0)
    n = 5000
    # Market price uniform on [0,1]; truth: YES wins with prob = price * 0.85.
    # → at high prices the empirical winrate is well below the diagonal.
    price = rng.uniform(0, 1, n)
    yes = (rng.uniform(0, 1, n) < price * 0.85).astype(int)
    df = pd.DataFrame({"lead_time": 24, "p_market": price, "yes_won": yes})
    cal = bin_by_price(df)
    high = cal[cal["bin_center"] > 0.85]
    assert (high["p_hat"] < high["mean_price"]).all()
