"""Batched NBM forecast extraction. One GRIB pull -> t2m at all 9 stations.

Caches per (issue_utc, fxx) the all-city extraction. This is the key
optimization: 9 cities share the same GRIB file per (issue, fxx).
"""
from __future__ import annotations
import functools, datetime as dt, warnings, os
from pathlib import Path
import numpy as np, pandas as pd

from .stations import STATIONS

TMP_REGEX = ":TMP:2 m above ground:"
CACHE = Path("cache/forecasts/grib_extracts")
CACHE.mkdir(parents=True, exist_ok=True)
warnings.filterwarnings("ignore")


@functools.lru_cache(maxsize=1)
def _grid_indices():
    from herbie import Herbie
    sample = pd.Timestamp.utcnow().tz_localize(None).floor("h") - pd.Timedelta(hours=6)
    h = Herbie(sample, model="nbm", product="co", fxx=1, priority=["aws"], verbose=False)
    ds = h.xarray(TMP_REGEX, remove_grib=False)
    lat = np.asarray(ds["latitude"]); lon = np.asarray(ds["longitude"])
    lon = np.where(lon > 180, lon - 360, lon)
    idx = {}
    for c, info in STATIONS.items():
        d2 = (lat - info["lat"])**2 + (lon - info["lon"])**2
        i, j = np.unravel_index(int(np.argmin(d2)), d2.shape)
        idx[c] = (int(i), int(j))
    return idx


def extract_all_cities(issue_utc: pd.Timestamp, fxx: int) -> dict | None:
    """Pull one (issue, fxx) GRIB, extract t2m °F at all 9 cities. Cached on disk."""
    fp = CACHE / f"{issue_utc:%Y%m%dT%HZ}_f{fxx:02d}.parquet"
    if fp.exists():
        d = pd.read_parquet(fp).iloc[0].to_dict()
        return {k: v for k, v in d.items() if k in STATIONS}
    from herbie import Herbie
    try:
        h = Herbie(issue_utc.to_pydatetime(), model="nbm", product="co", fxx=fxx,
                   priority=["aws"], verbose=False)
        if getattr(h, "grib", None) is None: return None
        ds = h.xarray(TMP_REGEX, remove_grib=False)
        arr = np.asarray(ds["t2m"])
        idx = _grid_indices()
        out = {c: (float(arr[i, j]) - 273.15) * 9/5 + 32 for c, (i, j) in idx.items()}
        pd.DataFrame([out]).to_parquet(fp)
        return out
    except Exception as e:
        return None


def fxx_window(issue_utc: pd.Timestamp, day: dt.date) -> list[int]:
    """Return fxx values whose valid_time lands inside the 12-23Z daylight
    window of `day`."""
    out = []
    for h in range(12, 24):
        valid = pd.Timestamp(day) + pd.Timedelta(hours=h)
        fxx = int((valid - issue_utc).total_seconds() // 3600)
        if 1 <= fxx <= 264:  # NBM CONUS max horizon ~ 264h
            out.append(fxx)
    return out
