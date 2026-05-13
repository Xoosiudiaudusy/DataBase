"""NBM forecast extraction for multiple stations from a single GRIB file.

The key efficiency: one Herbie pull (= one S3 download + cfgrib parse) ->
extract T2m at all 9 station (lat,lon) pixels in the same xarray. Subsequent
issue-times re-use the AWS-side caching.
"""
from __future__ import annotations
import functools, datetime as dt, warnings
from pathlib import Path
import numpy as np, pandas as pd

from .stations import STATIONS, peak_utc

TMP_REGEX = ":TMP:2 m above ground:"
CACHE = Path("cache/forecasts/multicity")
CACHE.mkdir(parents=True, exist_ok=True)
warnings.filterwarnings("ignore")


@functools.lru_cache(maxsize=1)
def _grid_indices():
    """One-shot: load any recent NBM run, find (i,j) for each station."""
    from herbie import Herbie
    sample = pd.Timestamp.utcnow().tz_localize(None).floor("h") - pd.Timedelta(hours=6)
    h = Herbie(sample, model="nbm", product="co", fxx=1, priority=["aws"], verbose=False)
    ds = h.xarray(TMP_REGEX, remove_grib=False)
    lat = np.asarray(ds["latitude"]); lon = np.asarray(ds["longitude"])
    lon = np.where(lon > 180, lon - 360, lon)
    idx = {}
    for city, info in STATIONS.items():
        d2 = (lat - info["lat"])**2 + (lon - info["lon"])**2
        i, j = np.unravel_index(int(np.argmin(d2)), d2.shape)
        idx[city] = (int(i), int(j))
    return idx


def extract_one_run(issue_utc: pd.Timestamp, fxx: int):
    """Pull one NBM run+fxx, extract t2m (°F) at all 9 stations."""
    from herbie import Herbie
    h = Herbie(issue_utc.to_pydatetime(), model="nbm", product="co", fxx=fxx,
               priority=["aws"], verbose=False)
    if getattr(h, "grib", None) is None: return None
    ds = h.xarray(TMP_REGEX, remove_grib=False)
    arr = np.asarray(ds["t2m"])
    idx = _grid_indices()
    out = {}
    for city, (i, j) in idx.items():
        k = float(arr[i, j])
        out[city] = (k - 273.15) * 9.0 / 5.0 + 32.0  # °F
    return out


def forecast_max_for_day(city: str, day: dt.date, issue_utc: pd.Timestamp,
                         allow_obs: bool = True) -> dict | None:
    """For city/day, given issue_utc, max(forecast_t2m over the 12-23 UTC window of day).

    Splice with observations if issue_utc is past start of window (obs first,
    forecast after).
    """
    fp = CACHE / f"{city}_{day}_{issue_utc:%Y%m%dT%HZ}.parquet"
    if fp.exists():
        return pd.read_parquet(fp).iloc[0].to_dict()

    # Compute fxx values that fall within day's 12-23 UTC daylight window
    day_start = pd.Timestamp(day) + pd.Timedelta(hours=12)
    day_end   = pd.Timestamp(day) + pd.Timedelta(hours=23)
    fxx_list = []
    for h in range(12, 24):
        valid = pd.Timestamp(day) + pd.Timedelta(hours=h)
        fxx = int((valid - issue_utc).total_seconds() // 3600)
        if fxx >= 1: fxx_list.append((h, fxx))

    fc_max = None
    n_fc = 0
    if fxx_list:
        vals = []
        for _, fxx in fxx_list:
            try:
                res = extract_one_run(issue_utc, fxx)
                if res and res.get(city) is not None:
                    vals.append(res[city]); n_fc += 1
            except Exception:
                continue
        if vals:
            fc_max = max(vals)

    obs_max, n_obs = None, 0
    if allow_obs:
        try:
            obs_max, n_obs = _observed_max(city, day, issue_utc)
        except Exception:
            pass

    candidates = [v for v in (fc_max, obs_max) if v is not None]
    if not candidates:
        return None
    result = {
        "city": city, "date": str(day), "issue_utc": issue_utc,
        "forecast_max_f": max(candidates),
        "n_fc_steps": n_fc, "n_obs": n_obs,
        "spliced": (obs_max is not None and n_fc > 0),
    }
    pd.DataFrame([result]).to_parquet(fp)
    return result


def _observed_max(city: str, day: dt.date, issue_utc: pd.Timestamp) -> tuple:
    """Observed Mesonet ASOS max for [12Z, issue_utc) at the city's station."""
    import requests, io
    icao = STATIONS[city]["icao"][1:]  # 4-char ICAO -> 3-char Mesonet id
    url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
    params = {"station": icao, "data": "tmpf",
              "year1": day.year, "month1": day.month, "day1": day.day,
              "year2": day.year, "month2": day.month, "day2": day.day,
              "tz": "Etc/UTC", "format": "onlycomma", "missing": "empty"}
    r = requests.get(url, params=params, timeout=20)
    if not r.ok: return None, 0
    df = pd.read_csv(io.StringIO(r.text))
    if df.empty or "tmpf" not in df.columns: return None, 0
    df["valid_utc"] = pd.to_datetime(df["valid"])
    win_start = pd.Timestamp(day) + pd.Timedelta(hours=12)
    sub = df[(df["valid_utc"] >= win_start) & (df["valid_utc"] < issue_utc)]
    sub = sub.dropna(subset=["tmpf"])
    if sub.empty: return None, 0
    return float(sub["tmpf"].max()), int(len(sub))
