"""Polymarket market-calibration: empirical winrate vs YES price.

Given the cached Polymarket markets + price histories and KLGA actuals, snapshot
each YES price at fixed lead times before peak heating (17:00 local on
resolve_date), join with the realized daily high, then bucket by price to get
the empirical P(YES | price) curve. A perfectly calibrated market sits on the
y = x diagonal; if the curve sags below the diagonal at high prices, the
market is overpricing YES (the original hypothesis).
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import wilson_ci
from .config import (
    ACTUALS_SOURCE_DEFAULT,
    LEAD_TIMES_HOURS,
    LOCAL_TZ,
    PEAK_HOUR_LOCAL,
    POLYMARKET_CACHE,
)
from .fetch_actuals import daily_highs


# 20 buckets of width 0.05 over [0, 1].
PRICE_BIN_EDGES = [round(0.05 * i, 4) for i in range(21)]


def peak_utc(resolve_date: dt.date) -> pd.Timestamp:
    """17:00 local time on ``resolve_date`` → tz-naive UTC timestamp."""
    return (
        pd.Timestamp(dt.datetime.combine(resolve_date, dt.time(PEAK_HOUR_LOCAL)))
        .tz_localize(LOCAL_TZ)
        .tz_convert("UTC")
        .tz_localize(None)
    )


def snapshot_prices(
    prices: pd.DataFrame,
    lead_times: list[int] = LEAD_TIMES_HOURS,
    tolerance_hours: float = 2.0,
) -> pd.DataFrame:
    """For each market × lead-time, take the YES price closest to ``peak − T``.

    ``prices`` is the long-form table produced by ``fetch_polymarket.run`` —
    columns: market_id, ts_utc, p, resolve_date, threshold_f. Snapshots that
    fall further than ``tolerance_hours`` from any cached price are dropped
    (sparse markets, no forward-fill).
    """
    cols = [
        "market_id", "resolve_date", "threshold_f", "lead_time",
        "target_ts_utc", "snap_ts_utc", "p_market",
    ]
    if prices.empty:
        return pd.DataFrame(columns=cols)

    prices = prices.sort_values(["market_id", "ts_utc"]).reset_index(drop=True)
    tol_ns = int(tolerance_hours * 3_600 * 10**9)
    out: list[dict] = []
    for market_id, grp in prices.groupby("market_id", sort=False):
        # Force ns precision on both sides so int64 differences are comparable.
        ts = pd.to_datetime(grp["ts_utc"]).to_numpy(dtype="datetime64[ns]")
        ts_int = ts.astype("int64")
        p_arr = grp["p"].to_numpy()
        resolve_date = grp["resolve_date"].iloc[0]
        threshold_f = grp["threshold_f"].iloc[0]
        if resolve_date is None or pd.isna(threshold_f):
            continue
        peak_ns = np.datetime64(peak_utc(resolve_date), "ns")
        for T in lead_times:
            target = peak_ns - np.timedelta64(int(T * 3600 * 10**9), "ns")
            diffs = np.abs(ts_int - target.astype("int64"))
            idx = int(np.argmin(diffs))
            if int(diffs[idx]) > tol_ns:
                continue
            out.append({
                "market_id": market_id,
                "resolve_date": resolve_date,
                "threshold_f": float(threshold_f),
                "lead_time": int(T),
                "target_ts_utc": pd.Timestamp(target),
                "snap_ts_utc": pd.Timestamp(ts[idx]),
                "p_market": float(p_arr[idx]),
            })
    return pd.DataFrame(out, columns=cols)


def join_with_actuals(
    snapshots: pd.DataFrame,
    actuals: pd.DataFrame | dict[dt.date, float],
    markets: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach ``actual_high_f`` (sanity) and ``yes_won`` (calibration target).

    The 2025 ``nyc-daily-weather`` series mixes three market shapes — "X°F or
    below", "X°F or above", "between X-Y°F" — so the legacy rule
    ``yes_won = actual_high >= threshold`` is wrong for two of them. When
    ``markets`` is supplied we read the resolved outcome directly from
    ``outcomePrices`` (1 = YES won) which is correct for all shapes. The
    actuals join is kept for diagnostics.
    """
    if isinstance(actuals, pd.DataFrame):
        actuals_map = dict(zip(actuals["local_date"], actuals["actual_high_f"]))
    else:
        actuals_map = dict(actuals)
    df = snapshots.copy()
    df["actual_high_f"] = df["resolve_date"].map(actuals_map)

    if markets is not None and "yes_won_final" in markets.columns:
        won = markets.set_index("market_id")["yes_won_final"]
        df["yes_won"] = df["market_id"].map(won)
        df = df.dropna(subset=["yes_won"]).copy()
        df["yes_won"] = df["yes_won"].astype(int)
    else:
        df = df.dropna(subset=["actual_high_f"]).copy()
        df["yes_won"] = (df["actual_high_f"] >= df["threshold_f"]).astype(int)
    return df.reset_index(drop=True)


def bin_by_price(
    df: pd.DataFrame, edges: list[float] | None = None,
) -> pd.DataFrame:
    """Bucket (lead_time, p_market) → empirical P(YES) with Wilson 95% CI."""
    edges_arr = np.asarray(edges if edges is not None else PRICE_BIN_EDGES)
    centers = 0.5 * (edges_arr[:-1] + edges_arr[1:])
    out: list[dict] = []
    for lt, sub in df.groupby("lead_time"):
        idx = np.digitize(sub["p_market"].to_numpy(), edges_arr) - 1
        idx = np.clip(idx, 0, len(centers) - 1)
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
                "mean_price": float(chunk["p_market"].mean()),
            })
    return pd.DataFrame(out)


def run(
    start: dt.date, end: dt.date,
    lead_times: list[int] = LEAD_TIMES_HOURS,
    actuals_source: str = ACTUALS_SOURCE_DEFAULT,
    tolerance_hours: float = 2.0,
    polymarket_dir: Path = POLYMARKET_CACHE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """End-to-end: cached prices → snapshots → join actuals → bin by price.

    Returns ``(joined_rows, calibration_table)``. Both are ready to persist.
    """
    prices_path = polymarket_dir / "prices.parquet"
    markets_path = polymarket_dir / "markets.parquet"
    if not prices_path.exists():
        raise FileNotFoundError(
            f"Polymarket cache missing at {prices_path}. "
            "Run `polycal fetch-polymarket --start ... --end ...` first."
        )
    prices = pd.read_parquet(prices_path)
    prices = prices[
        (prices["resolve_date"] >= start) & (prices["resolve_date"] <= end)
    ].copy()
    if prices.empty:
        raise RuntimeError(f"No cached Polymarket prices in [{start}, {end}]")

    snaps = snapshot_prices(prices, lead_times=lead_times,
                            tolerance_hours=tolerance_hours)
    if snaps.empty:
        raise RuntimeError("No price snapshots within tolerance — widen --tolerance?")

    markets = pd.read_parquet(markets_path) if markets_path.exists() else None
    try:
        actuals = daily_highs(start, end, source=actuals_source)
    except Exception as e:
        print(f"  actuals unavailable ({e}); proceeding with market-outcome resolution only.")
        actuals = {}
    joined = join_with_actuals(snaps, actuals, markets=markets)
    if joined.empty:
        raise RuntimeError("Snapshots present but no resolvable markets.")

    cal = bin_by_price(joined)
    return joined, cal
