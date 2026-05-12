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


def per_lead_mae(df: pd.DataFrame) -> dict[int, float]:
    """MAE of forecast vs actual high per lead time. One row per (date, lead)."""
    per_day = df.drop_duplicates(["date", "lead_time"])
    per_day = per_day.assign(abs_err=(per_day["forecast_high"] - per_day["actual_high"]).abs())
    return per_day.groupby("lead_time")["abs_err"].mean().to_dict()


def theoretical_curve(
    mae_f: float, spreads: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """P(YES | spread) under zero-bias normal error with given MAE."""
    if spreads is None:
        spreads = np.linspace(-5.25, 5.25, 200)
    sigma = mae_f * math.sqrt(math.pi / 2.0)
    if sigma <= 0:
        return spreads, np.full_like(spreads, 0.5)
    return spreads, norm.cdf(spreads / sigma)
