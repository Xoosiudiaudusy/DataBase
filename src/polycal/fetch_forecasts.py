"""NOAA forecast retrieval via Herbie.

NBM (CONUS, hourly) is the primary model; HRRR is the fallback. We pull the
2 m temperature field at the station grid point for every hourly step in the
12–23 UTC daily window of day D, then take the max as the forecast daily high.

Lead-time semantics (peak-anchored):
    issue_utc = floor_hour( local(D, PEAK_HOUR_LOCAL) - lead_hours )
where local() converts a naive timestamp in America/New_York to UTC. The
floor handles DST and the fact that NBM/HRRR run on the hour. "T hours
before peak heating" is what we test against — a real forecast question,
not a nowcast already settled by post-peak observations.

When ``issue_utc`` lies *inside* the daily window (short lead times such as
T=1h, T=3h with morning hours already observed), we splice: ISD/Mesonet
observations cover ``valid_time < issue_utc`` and the NBM forecast covers
``valid_time >= issue_utc``. The trader at issue_time knows what's already
been observed. Set ``allow_obs_splice=False`` for pure-forecast mode.
"""
from __future__ import annotations

import datetime as dt
import functools
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import (
    ACTUALS_SOURCE_DEFAULT,
    DAILY_WINDOW_UTC_HOURS,
    FORECAST_CACHE,
    LOCAL_TZ,
    PEAK_HOUR_LOCAL,
    STATION_LAT,
    STATION_LON,
    kelvin_to_f,
)

# Variable regex shared by NBM CONUS and HRRR sfc products.
TMP_2M_REGEX = ":TMP:2 m above ground:"


def issue_time_for(local_date: dt.date, lead_hours: int) -> pd.Timestamp:
    """Return the (hour-floored, tz-naive UTC) NBM run time for (date, lead).

    Anchored on peak heating: T hours before ``PEAK_HOUR_LOCAL`` on day D.
    DST is handled by ``tz_localize(LOCAL_TZ)``.
    """
    peak_local = pd.Timestamp(
        year=local_date.year, month=local_date.month, day=local_date.day,
        hour=PEAK_HOUR_LOCAL,
    ).tz_localize(LOCAL_TZ)
    issue = peak_local - pd.Timedelta(hours=lead_hours)
    issue_utc = issue.tz_convert("UTC").tz_localize(None)
    return issue_utc.floor("h")


def fxx_list_for(issue_utc: pd.Timestamp, local_date: dt.date) -> list[int]:
    """fxx values whose valid_time falls in [12Z(D), 23Z(D)].

    Returns an empty list if the issue time is past the end of the window.
    """
    out = []
    for h in DAILY_WINDOW_UTC_HOURS:
        valid = pd.Timestamp(local_date) + pd.Timedelta(hours=h)
        fxx = int((valid - issue_utc).total_seconds() // 3600)
        if fxx >= 0:
            out.append(fxx)
    return out


@functools.lru_cache(maxsize=4)
def _grid_index(model: str) -> tuple[int, int]:
    """Locate the nearest grid index to the station once per model.

    We do a one-shot Herbie pull of a tiny known-good run to learn (i, j).
    """
    from herbie import Herbie  # local import keeps top-level import cheap

    # Recent NBM/HRRR run that is virtually guaranteed to exist.
    sample_run = pd.Timestamp.utcnow().tz_localize(None).floor("h") - pd.Timedelta(hours=6)
    product = "co" if model == "nbm" else "sfc"
    h = Herbie(sample_run, model=model, product=product, fxx=1,
               priority=["aws"], verbose=False)
    ds = h.xarray(TMP_2M_REGEX, remove_grib=False)
    lat = np.asarray(ds["latitude"])
    lon = np.asarray(ds["longitude"])
    # longitude can be 0-360; normalize to (-180, 180].
    lon = np.where(lon > 180, lon - 360, lon)
    dist = (lat - STATION_LAT) ** 2 + (lon - STATION_LON) ** 2
    i, j = np.unravel_index(int(np.argmin(dist)), dist.shape)
    return int(i), int(j)


def _extract_t2m_f(ds, model: str) -> float:
    """Pull the station-pixel 2m temperature in °F from a Herbie xarray ds."""
    i, j = _grid_index(model)
    # 2m temp variable name from cfgrib is conventionally "t2m".
    arr = ds["t2m"] if "t2m" in ds else ds[list(ds.data_vars)[0]]
    val_k = float(np.asarray(arr)[i, j])
    return kelvin_to_f(val_k)


def fetch_forecast_temps(
    issue_utc: pd.Timestamp, fxx_iter: Iterable[int], model: str = "nbm"
) -> pd.DataFrame:
    """For one model run, fetch t2m at the station for each fxx.

    Returns DataFrame: valid_time_utc, fxx, t2m_f. Per-fxx failures (file
    missing on S3, GRIB extraction errors) are skipped, not raised — the
    daily-max max() will still work on whatever survives.
    """
    from herbie import Herbie

    product = "co" if model == "nbm" else "sfc"
    rows: list[dict] = []
    for fxx in fxx_iter:
        try:
            h = Herbie(
                issue_utc.to_pydatetime(),
                model=model, product=product, fxx=fxx,
                priority=["aws"],  # skip NOMADS — broken SSL in some sandboxes
                verbose=False,
            )
            if getattr(h, "grib", None) is None:
                continue
            ds = h.xarray(TMP_2M_REGEX, remove_grib=False)
            t2m_f = _extract_t2m_f(ds, model)
        except Exception as e:
            print(f"  [skip] {model} run={issue_utc:%Y-%m-%dT%H} fxx={fxx}: {type(e).__name__}: {e}")
            continue
        valid = issue_utc + pd.Timedelta(hours=fxx)
        rows.append({"valid_time_utc": valid, "fxx": fxx, "t2m_f": t2m_f})
    return pd.DataFrame(rows)


def _per_day_cache(model: str, local_date: dt.date, issue_utc: pd.Timestamp) -> Path:
    sub = FORECAST_CACHE / model / f"{local_date:%Y}" / f"{local_date:%m}"
    sub.mkdir(parents=True, exist_ok=True)
    return sub / f"forecast_high_{local_date:%Y-%m-%d}_{issue_utc:%Y%m%dT%HZ}.parquet"


def _observed_max_before(
    local_date: dt.date, issue_utc: pd.Timestamp,
    actuals_source: str,
) -> tuple[float | None, int]:
    """Max ISD/Mesonet temperature for valid times in [12Z(D), issue_utc)."""
    from .fetch_actuals import fetch_actuals_year

    raw = fetch_actuals_year(local_date.year, source=actuals_source)
    window_start = pd.Timestamp(local_date).tz_localize("UTC") + pd.Timedelta(hours=DAILY_WINDOW_UTC_HOURS[0])
    issue_utc_aware = issue_utc.tz_localize("UTC")
    mask = (raw["valid_utc"] >= window_start) & (raw["valid_utc"] < issue_utc_aware)
    sub = raw.loc[mask, "tmpf"]
    if sub.empty:
        return None, 0
    return float(sub.max()), int(len(sub))


def forecast_high_for_day(
    local_date: dt.date, lead_hours: int, model: str = "nbm",
    allow_obs_splice: bool = True,
    actuals_source: str = ACTUALS_SOURCE_DEFAULT,
) -> dict | None:
    """Compute (and cache) the forecast daily high for one (date, lead).

    For (date, lead) where the issue time still leaves forecast steps inside
    the daily window, this is a pure NBM/HRRR forecast. For short lead times
    where the issue time has passed (part of) the window, the daily high is
    spliced as max( observed temps for valid<issue_utc, forecast for valid>=issue_utc ) —
    matching what a Polymarket trader knows at that moment. Set
    ``allow_obs_splice=False`` for the strict-spec interpretation (returns
    ``None`` in that case).
    """
    issue_utc = issue_time_for(local_date, lead_hours)
    cache_path = _per_day_cache(model, local_date, issue_utc)

    if cache_path.exists():
        steps = pd.read_parquet(cache_path)
    else:
        fxx_iter = fxx_list_for(issue_utc, local_date)
        if fxx_iter:
            steps = fetch_forecast_temps(issue_utc, fxx_iter, model=model)
            steps.insert(0, "local_date", local_date)
            steps.insert(1, "model", model)
            steps.insert(2, "run_hour_utc", issue_utc)
            steps.to_parquet(cache_path, index=False)
        else:
            steps = pd.DataFrame(columns=["valid_time_utc", "fxx", "t2m_f"])

    forecast_max = float(steps["t2m_f"].max()) if not steps.empty else None
    n_forecast = int(len(steps))

    observed_max = None
    n_observed = 0
    if allow_obs_splice:
        observed_max, n_observed = _observed_max_before(
            local_date, issue_utc, actuals_source=actuals_source,
        )

    candidates = [v for v in (forecast_max, observed_max) if v is not None]
    if not candidates:
        return None

    return {
        "local_date": local_date,
        "lead_time": lead_hours,
        "run_hour_utc": issue_utc,
        "forecast_high_f": max(candidates),
        "n_steps": n_forecast,
        "n_obs_used": n_observed,
        "spliced": observed_max is not None and n_forecast > 0,
        "obs_only": observed_max is not None and n_forecast == 0,
    }
