"""METAR / ASOS observations.

Two interchangeable sources:

- ``mesonet`` (default): Iowa State Mesonet ASOS download. Rich quality,
  hourly granularity, one CSV per calendar year.
- ``isd``: NOAA Integrated Surface Database via AWS S3
  (``s3://noaa-global-hourly-pds/{year}/{usaf+wban}.csv``). Public,
  no auth, always reachable from AWS-friendly networks. Used as a
  fallback when ``mesonet`` is blocked.

Both are cached as parquet under ``cache/actuals/``. Daily highs are
computed in America/New_York local time to line up with the Polymarket
resolution day.
"""
from __future__ import annotations

import datetime as dt
import io

import pandas as pd
import requests

from .config import (
    ACTUALS_CACHE,
    ACTUALS_SOURCE_DEFAULT,
    ISD_STATION,
    LOCAL_TZ,
    STATION_ID,
)

IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
ISD_S3_URL = "https://noaa-global-hourly-pds.s3.amazonaws.com/{year}/{station}.csv"


def _raw_parquet_path(year: int, source: str) -> "Path":  # noqa: F821
    return ACTUALS_CACHE / f"{source}_{STATION_ID.lower()}_{year}.parquet"


def fetch_mesonet_year(year: int, force: bool = False) -> pd.DataFrame:
    """Download one calendar year of ASOS observations from Iowa State Mesonet.

    Returns a DataFrame with columns ``valid_utc`` (tz-aware UTC) and ``tmpf``.
    """
    path = _raw_parquet_path(year, "mesonet")
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
        "report_type": [3, 4],
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


# ISD quality codes considered acceptable (passed checks or gross-limits only).
_ISD_OK_QC = {"0", "1", "4", "5", "9"}


def fetch_isd_year(year: int, force: bool = False) -> pd.DataFrame:
    """Download one calendar year of NOAA ISD hourly obs from S3.

    The TMP field is ``±SDDDD,Q`` — signed degrees Celsius × 10, plus a
    one-char quality code. Missing values are encoded as ``+9999``. We keep
    METAR-like report types (FM-15 hourly, FM-16 special) and SYNOP (FM-12).
    """
    path = _raw_parquet_path(year, "isd")
    if path.exists() and not force:
        return pd.read_parquet(path)

    url = ISD_S3_URL.format(year=year, station=ISD_STATION)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    raw = pd.read_csv(io.StringIO(resp.text), low_memory=False)

    parts = raw["TMP"].astype(str).str.split(",", n=1, expand=True)
    val_str = parts[0]
    qc = parts[1]
    raw_c = pd.to_numeric(val_str, errors="coerce") / 10.0
    raw_c = raw_c.where(val_str != "+9999")
    raw_c = raw_c.where(qc.isin(_ISD_OK_QC))

    df = pd.DataFrame({
        "valid_utc": pd.to_datetime(raw["DATE"], utc=True, errors="coerce"),
        "tmpf": raw_c * 9.0 / 5.0 + 32.0,
        "report_type": raw["REPORT_TYPE"].astype(str).str.strip(),
    })
    df = df.dropna(subset=["valid_utc", "tmpf"])
    df = df[df["report_type"].isin({"FM-12", "FM-15", "FM-16"})]
    df = df[["valid_utc", "tmpf"]].reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"ISD returned no usable rows for {ISD_STATION} {year}")
    df.to_parquet(path, index=False)
    return df


def fetch_actuals_year(year: int, source: str = ACTUALS_SOURCE_DEFAULT) -> pd.DataFrame:
    if source == "mesonet":
        return fetch_mesonet_year(year)
    if source == "isd":
        return fetch_isd_year(year)
    raise ValueError(f"unknown actuals source: {source!r}")


def daily_highs(
    start: dt.date, end: dt.date,
    source: str = ACTUALS_SOURCE_DEFAULT,
) -> pd.DataFrame:
    """Daily max temperature in °F, grouped by local NY calendar day."""
    years = list(range(start.year, end.year + 1))
    frames = [fetch_actuals_year(y, source=source) for y in years]
    raw = pd.concat(frames, ignore_index=True)

    local = raw["valid_utc"].dt.tz_convert(LOCAL_TZ)
    raw = raw.assign(local_date=local.dt.date)

    daily = (
        raw.groupby("local_date", as_index=False)
        .agg(actual_high_f=("tmpf", "max"), n_obs=("tmpf", "size"))
    )
    mask = (daily["local_date"] >= start) & (daily["local_date"] <= end)
    return daily.loc[mask].reset_index(drop=True)
