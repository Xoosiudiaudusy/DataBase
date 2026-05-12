"""METAR / ASOS observations from Iowa State Mesonet.

One HTTP request per calendar year, cached as parquet. Daily highs are
computed in America/New_York local time so they line up with the Polymarket
resolution day.
"""
from __future__ import annotations

import datetime as dt
import io

import pandas as pd
import requests

from .config import (
    ACTUALS_CACHE,
    LOCAL_TZ,
    STATION_ID,
)

IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"


def _raw_parquet_path(year: int) -> "Path":  # noqa: F821
    return ACTUALS_CACHE / f"asos_{STATION_ID.lower()}_{year}.parquet"


def fetch_asos_year(year: int, force: bool = False) -> pd.DataFrame:
    """Download one calendar year of ASOS observations for the configured station.

    Returns a DataFrame with columns ``valid_utc`` (tz-aware UTC) and ``tmpf``.
    Cached as parquet under ``cache/actuals/``.
    """
    path = _raw_parquet_path(year)
    if path.exists() and not force:
        return pd.read_parquet(path)

    today = dt.date.today()
    start = dt.date(year, 1, 1)
    end = min(dt.date(year + 1, 1, 1), today + dt.timedelta(days=1))

    params = {
        "station": STATION_ID,
        "data": "tmpf",
        "year1": start.year, "month1": start.month, "day1": start.day,
        "year2": end.year, "month2": end.month, "day2": end.day,
        "tz": "Etc/UTC",
        "format": "onlycomma",
        "latlon": "no",
        "missing": "null",
        "trace": "null",
        "direct": "no",
        "report_type": [3, 4],  # routine + special METAR
    }
    resp = requests.get(IEM_URL, params=params, timeout=120)
    resp.raise_for_status()

    df = pd.read_csv(io.StringIO(resp.text))
    if df.empty:
        raise RuntimeError(f"Iowa Mesonet returned empty CSV for {STATION_ID} {year}")

    df["valid_utc"] = pd.to_datetime(df["valid"], utc=True, errors="coerce")
    df["tmpf"] = pd.to_numeric(df["tmpf"], errors="coerce")
    df = df.dropna(subset=["valid_utc", "tmpf"])[["valid_utc", "tmpf"]]
    df.to_parquet(path, index=False)
    return df


def daily_highs(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Daily max temperature in °F, grouped by local NY calendar day.

    ``end`` is inclusive. Returns DataFrame with columns
    ``local_date`` (date), ``actual_high_f`` (float), ``n_obs`` (int).
    """
    years = list(range(start.year, end.year + 1))
    frames = [fetch_asos_year(y) for y in years]
    raw = pd.concat(frames, ignore_index=True)

    local = raw["valid_utc"].dt.tz_convert(LOCAL_TZ)
    raw = raw.assign(local_date=local.dt.date)

    daily = (
        raw.groupby("local_date", as_index=False)
        .agg(actual_high_f=("tmpf", "max"), n_obs=("tmpf", "size"))
    )
    mask = (daily["local_date"] >= start) & (daily["local_date"] <= end)
    return daily.loc[mask].reset_index(drop=True)
