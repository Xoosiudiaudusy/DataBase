"""Calibration math: spread bucketing, Wilson CI, theoretical normal curve."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import SPREAD_BIN_EDGES_F


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Returns (lo, hi). For n=0 returns (0, 1)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _bin_centers() -> np.ndarray:
    edges = np.asarray(SPREAD_BIN_EDGES_F)
    return 0.5 * (edges[:-1] + edges[1:])


def bin_and_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket rows by (lead_time, spread bin) and compute p_hat + Wilson CI."""
    edges = np.asarray(SPREAD_BIN_EDGES_F)
    centers = _bin_centers()

    out: list[dict] = []
    for lt, sub in df.groupby("lead_time"):
        idx = np.digitize(sub["spread"].to_numpy(), edges) - 1
        sub = sub.assign(_bin=idx)
        for b in range(len(centers)):
            chunk = sub[sub["_bin"] == b]
            n = int(len(chunk))
            if n == 0:
                continue
            k = int(chunk["yes_won"].sum())
            lo, hi = wilson_ci(k, n)
            out.append({
                "lead_time": int(lt),
                "bin_center": float(centers[b]),
                "n": n, "k": k,
                "p_hat": k / n,
                "lo": lo, "hi": hi,
            })
    return pd.DataFrame(out)


def per_lead_error_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-lead error stats. Forecast error = forecast - actual; one row per (date, lead).

    Returns DataFrame indexed by lead_time with columns: n, mae, bias, std.
    The theoretical-normal calibration curve uses ``bias`` and ``std`` directly,
    not MAE — because the NBM/HRRR error distribution has a non-zero mean
    (NBM in NYC is consistently cold-biased), and an unbiased Gaussian
    overstates uncertainty.
    """
    per_day = df.drop_duplicates(["date", "lead_time"]).copy()
    per_day["err"] = per_day["forecast_high"] - per_day["actual_high"]
    per_day["ae"] = per_day["err"].abs()
    return per_day.groupby("lead_time").agg(
        n=("date", "nunique"),
        mae=("ae", "mean"),
        bias=("err", "mean"),
        std=("err", "std"),
    )


def per_lead_mae(df: pd.DataFrame) -> dict[int, float]:
    """Backward-compatible MAE-only helper (used by older plot code)."""
    return per_lead_error_stats(df)["mae"].to_dict()


def theoretical_curve(
    sigma_f: float, bias_f: float = 0.0,
    spreads: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """P(YES | spread) under bias-adjusted normal forecast error.

    ``sigma_f`` and ``bias_f`` are the std and mean of (forecast - actual)
    over the relevant lead time. Then
        P(actual >= threshold | spread) = Phi( (spread - bias) / sigma ).
    With bias=0 this reduces to the standard zero-bias normal calibration.
    """
    if spreads is None:
        spreads = np.linspace(-5.25, 5.25, 200)
    if sigma_f <= 0:
        return spreads, np.full_like(spreads, 0.5)
    return spreads, norm.cdf((spreads - bias_f) / sigma_f)
