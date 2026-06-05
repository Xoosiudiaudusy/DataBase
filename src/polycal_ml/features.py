"""Feature engineering for bucket-market YES-probability prediction.

One row = one decision moment for one bucket market.

Required raw inputs per row:
  - city, resolve_date, bucket_lo, bucket_hi, decision_lead_h
  - NBM forecast caches: hourly issue_time × (city, fxx)
  - Mesonet ASOS caches: per-station hourly temperature observations
  - Market microstructure: bucket prices over time (optional)

Conventions:
  - All times are tz-naive UTC.
  - decision_moment = peak_utc(city, resolve_date) − decision_lead_h
  - Features are computed using ONLY data observable at or before
    decision_moment (no future leakage).
"""
from __future__ import annotations
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from polycal_lag.stations import STATIONS, peak_utc
from polycal_lag.forecast_batch import fxx_window

GRIB_EXT = Path("cache/forecasts/grib_extracts")
ASOS_CACHE = Path("cache/actuals")
TICKS_CACHE = Path("cache/polymarket/ticks")


# ---------------- NBM forecast helpers ----------------

NBM_PARQUET = Path("cache/forecasts/nbm_extracts.parquet")
_NBM_CACHE: dict | None = None


def _nbm_index() -> dict:
    """Lazy-load the big NBM extract parquet into a dict keyed by
    (issue_utc, fxx) -> {city: t2m_f}. Falls back to per-file GRIB extracts
    if the consolidated parquet is absent."""
    global _NBM_CACHE
    if _NBM_CACHE is not None:
        return _NBM_CACHE
    _NBM_CACHE = {}
    if NBM_PARQUET.exists():
        df = pd.read_parquet(NBM_PARQUET)
        df["issue_utc"] = pd.to_datetime(df["issue_utc"])
        cities = [c for c in STATIONS if c in df.columns]
        for r in df.itertuples(index=False):
            key = (getattr(r, "issue_utc"), int(getattr(r, "fxx")))
            _NBM_CACHE[key] = {c: getattr(r, c) for c in cities}
    return _NBM_CACHE


def _forecast_steps(city: str, date: dt.date, lead_h: int) -> list[tuple]:
    """Return [(valid_time, t2m_f), ...] for the daylight window of `date`
    given the issue time `lead_h` before peak. Reads the consolidated
    parquet; falls back to per-file extracts."""
    p = peak_utc(city, date)
    issue = (p - pd.Timedelta(hours=lead_h)).floor("h")
    idx = _nbm_index()
    rows = []
    for fx in fxx_window(issue, date):
        v = None
        if idx:
            rec = idx.get((issue, fx))
            if rec is not None:
                v = rec.get(city)
        else:  # fallback: per-file extracts
            fp = GRIB_EXT / f"{issue:%Y%m%dT%HZ}_f{fx:02d}.parquet"
            if fp.exists():
                v = pd.read_parquet(fp).iloc[0].get(city)
        if v is not None and pd.notna(v):
            rows.append((issue + pd.Timedelta(hours=fx), float(v)))
    return rows


def nbm_forecast_max(city: str, date: dt.date, lead_h: int) -> float | None:
    """Forecast max over the daylight window at `lead_h` before peak.

    NOTE: at short leads this only covers post-issue hours, so it can miss a
    peak that already happened (cold bias). build_features combines it with
    observed max-so-far into `fc_spliced_daily_max`."""
    rows = _forecast_steps(city, date, lead_h)
    return max(v for _, v in rows) if rows else None


def nbm_forecast_stats(city: str, date: dt.date, lead_h: int) -> dict:
    """max, min, std, argmax_hour over the daylight-window forecast steps."""
    rows = _forecast_steps(city, date, lead_h)
    if not rows:
        return {"max": None, "min": None, "std": None, "argmax_hour_utc": None}
    arr = np.array([v for _, v in rows])
    argmax_idx = int(arr.argmax())
    return {
        "max": float(arr.max()),
        "min": float(arr.min()),
        "std": float(arr.std(ddof=0)),
        "argmax_hour_utc": rows[argmax_idx][0].hour,
    }


# ---------------- ASOS observation helpers ----------------

def _load_asos_year(city: str, year: int) -> pd.DataFrame | None:
    """Cached hourly observations for (city, year). Returns DataFrame with
    columns valid_utc, tmpf or None if absent."""
    icao = STATIONS[city]["icao"][1:]
    fp = ASOS_CACHE / f"mesonet_{icao.lower()}_{year}.parquet"
    if not fp.exists():
        return None
    df = pd.read_parquet(fp)
    if "valid_utc" not in df.columns:
        if "valid" in df.columns:
            df["valid_utc"] = pd.to_datetime(df["valid"]).dt.tz_localize(None)
        else:
            return None
    return df[["valid_utc", "tmpf"]].dropna().sort_values("valid_utc")


def obs_features(city: str, decision_moment: pd.Timestamp,
                 resolve_date: dt.date) -> dict:
    """Observation features visible at `decision_moment`. No future leakage."""
    out = {
        "temp_now": None, "temp_lag_1h": None, "temp_lag_3h": None,
        "temp_lag_6h": None, "temp_lag_12h": None,
        "obs_max_so_far_today": None, "obs_min_so_far_today": None,
        "rate_1h": None, "rate_3h": None, "rate_6h": None,
        "yesterday_max": None, "last_7day_max_mean": None,
        "last_7day_max_std": None,
    }
    df = _load_asos_year(city, resolve_date.year)
    if df is None or df.empty:
        return out

    cutoff = decision_moment
    sub = df[df["valid_utc"] <= cutoff]
    if sub.empty:
        return out

    out["temp_now"] = float(sub["tmpf"].iloc[-1])
    for lag_h, key in [(1, "temp_lag_1h"), (3, "temp_lag_3h"),
                       (6, "temp_lag_6h"), (12, "temp_lag_12h")]:
        target = cutoff - pd.Timedelta(hours=lag_h)
        nearby = sub[sub["valid_utc"] <= target]
        if not nearby.empty:
            out[key] = float(nearby["tmpf"].iloc[-1])
    for key, lag in [("rate_1h", 1), ("rate_3h", 3), ("rate_6h", 6)]:
        if out[f"temp_lag_{lag}h"] is not None and out["temp_now"] is not None:
            out[key] = (out["temp_now"] - out[f"temp_lag_{lag}h"]) / lag

    # Today's stats so far (12 UTC daylight window start)
    day_start = pd.Timestamp(resolve_date) + pd.Timedelta(hours=12)
    today_obs = sub[(sub["valid_utc"] >= day_start) & (sub["valid_utc"] <= cutoff)]
    if not today_obs.empty:
        out["obs_max_so_far_today"] = float(today_obs["tmpf"].max())
        out["obs_min_so_far_today"] = float(today_obs["tmpf"].min())

    # Yesterday's max (full prior day, 12-23 UTC window)
    yest_start = pd.Timestamp(resolve_date - dt.timedelta(days=1)) + pd.Timedelta(hours=12)
    yest_end = pd.Timestamp(resolve_date - dt.timedelta(days=1)) + pd.Timedelta(hours=23)
    yest = sub[(sub["valid_utc"] >= yest_start) & (sub["valid_utc"] <= yest_end)]
    if not yest.empty:
        out["yesterday_max"] = float(yest["tmpf"].max())

    # Last 7 days daily-max
    week_start = pd.Timestamp(resolve_date - dt.timedelta(days=7))
    week = sub[(sub["valid_utc"] >= week_start) & (sub["valid_utc"] < pd.Timestamp(resolve_date))]
    if not week.empty:
        daily = week.set_index("valid_utc")["tmpf"].resample("D").max().dropna()
        if not daily.empty:
            out["last_7day_max_mean"] = float(daily.mean())
            out["last_7day_max_std"] = float(daily.std(ddof=0)) if len(daily) > 1 else 0.0
    return out


# ---------------- Market microstructure helpers ----------------

def market_features(condition_id: str, yes_token: str,
                    target_ts: pd.Timestamp) -> dict:
    """YES-price + staleness at decision moment.

    Rule: market_yes_price = last cached trade with ts_utc <= target_ts,
    AFTER sorting by ts_utc (the /trades endpoint returns reverse-chrono;
    forgetting to sort silently picks the OLDEST trade and was the source
    of the +97% in-sample illusion fixed earlier in the session).

    price_staleness_min is given to the model as a feature so it can
    discount very-stale prices.
    """
    out = {"market_yes_price": None, "price_staleness_min": None,
           "n_ticks_before_target": None}
    fp = TICKS_CACHE / f"{condition_id}.parquet"
    if not fp.exists(): return out
    t = pd.read_parquet(fp)
    if t.empty: return out
    t = t.copy()
    t["ts_utc"] = pd.to_datetime(t["ts_utc"])
    t["yes_price"] = np.where(
        t["asset"].astype(str) == str(yes_token), t["price"], 1 - t["price"]
    )
    t = t.sort_values("ts_utc").reset_index(drop=True)
    before = t[t["ts_utc"] <= target_ts]
    if before.empty: return out
    last_ts = before["ts_utc"].iloc[-1]
    out["market_yes_price"] = float(before["yes_price"].iloc[-1])
    out["price_staleness_min"] = float((target_ts - last_ts).total_seconds() / 60)
    out["n_ticks_before_target"] = int(len(before))
    return out


# ---------------- Top-level row builder ----------------

@dataclass
class FeatureRow:
    city: str
    resolve_date: dt.date
    bucket_lo: int
    bucket_hi: int
    decision_lead_h: int
    yes_won: int | None = None  # target
    condition_id: str | None = None  # for market price lookup
    yes_token: str | None = None     # for market price lookup


def build_features(row: FeatureRow) -> dict:
    """Compute the full feature vector for one decision."""
    city = row.city
    d = row.resolve_date
    lead = row.decision_lead_h
    p = peak_utc(city, d)
    decision_moment = p - pd.Timedelta(hours=lead)

    f = {
        "city": city,
        "resolve_date": str(d),
        "lead_h": lead,
        "bucket_lo": row.bucket_lo,
        "bucket_hi": row.bucket_hi,
        "bucket_width": row.bucket_hi - row.bucket_lo,
        "bucket_mid": (row.bucket_lo + row.bucket_hi) / 2,
        "month": d.month,
        "day_of_year": d.timetuple().tm_yday,
        "is_transition_month": 1 if d.month in (3, 4, 5, 9, 10, 11) else 0,
        "weekday": d.weekday(),
        "lat": STATIONS[city]["lat"],
        "lon": STATIONS[city]["lon"],
        "yes_won": row.yes_won,
    }
    # NBM features at the standard lead set (always all of them — that way
    # feature dict has a stable schema regardless of decision lead).
    for L in (1, 3, 6, 12, 24, 48):
        s = nbm_forecast_stats(city, d, L)
        suffix = f"L{L}h"
        f[f"mu_{suffix}"] = s["max"]
        f[f"mu_min_{suffix}"] = s["min"]
        f[f"mu_std_{suffix}"] = s["std"]
        f[f"mu_argmax_hr_{suffix}"] = s["argmax_hour_utc"]

    mu_now = f[f"mu_L{lead}h"]
    f["delta_mu_24h"] = (mu_now - f["mu_L24h"]) if mu_now and f["mu_L24h"] else None
    f["delta_mu_48h"] = (mu_now - f["mu_L48h"]) if mu_now and f["mu_L48h"] else None
    f["bucket_mid_minus_mu"] = (f["bucket_mid"] - mu_now) if mu_now else None
    f["fc_window_range"] = (f[f"mu_L{lead}h"] - f[f"mu_min_L{lead}h"]
                            ) if f[f"mu_L{lead}h"] and f[f"mu_min_L{lead}h"] else None

    obs = obs_features(city, decision_moment, d)
    f.update(obs)
    f["obs_minus_fc_now"] = (
        obs["temp_now"] - mu_now
        if obs["temp_now"] is not None and mu_now is not None else None
    )
    f["bucket_lo_minus_obs_max"] = (
        row.bucket_lo - obs["obs_max_so_far_today"]
        if obs["obs_max_so_far_today"] is not None else None
    )

    # Spliced daily-max estimate: the trader's actual best guess of the
    # realized daily high at decision time. At short leads the pure forecast
    # `mu_now` only covers the post-issue hours and can MISS a peak that
    # already happened (causes a cold bias) — so we take the max of the
    # forecast and the observed daily max so far. This is what
    # forecast_high_for_day(allow_obs_splice=True) does. No leakage: both
    # inputs are knowable at decision_moment.
    splice_parts = [v for v in (mu_now, obs.get("obs_max_so_far_today")) if v is not None]
    f["fc_spliced_daily_max"] = max(splice_parts) if splice_parts else None
    f["bucket_mid_minus_spliced"] = (
        f["bucket_mid"] - f["fc_spliced_daily_max"]
        if f["fc_spliced_daily_max"] is not None else None
    )
    f["bucket_hi_minus_yesterday_max"] = (
        row.bucket_hi - obs["yesterday_max"]
        if obs["yesterday_max"] is not None else None
    )

    # Market microstructure (price + staleness)
    if row.condition_id and row.yes_token:
        mkt = market_features(row.condition_id, row.yes_token, decision_moment)
    else:
        mkt = {"market_yes_price": None, "price_staleness_min": None,
               "n_ticks_before_target": None}
    f.update(mkt)
    return f


# Feature names (for sklearn pipeline)
FEATURE_COLS = [
    "lead_h", "bucket_lo", "bucket_hi", "bucket_width", "bucket_mid",
    "month", "day_of_year", "is_transition_month", "weekday", "lat", "lon",
    "mu_L1h", "mu_L3h", "mu_L6h", "mu_L12h", "mu_L24h", "mu_L48h",
    "mu_min_L6h", "mu_std_L6h", "mu_argmax_hr_L6h",
    "delta_mu_24h", "delta_mu_48h",
    "bucket_mid_minus_mu", "fc_window_range",
    "fc_spliced_daily_max", "bucket_mid_minus_spliced",
    "temp_now", "temp_lag_1h", "temp_lag_3h", "temp_lag_6h", "temp_lag_12h",
    "obs_max_so_far_today", "obs_min_so_far_today",
    "rate_1h", "rate_3h", "rate_6h",
    "yesterday_max", "last_7day_max_mean", "last_7day_max_std",
    "obs_minus_fc_now", "bucket_lo_minus_obs_max",
    "bucket_hi_minus_yesterday_max",
    # Market microstructure (with staleness fed in so model can discount it)
    "market_yes_price", "price_staleness_min", "n_ticks_before_target",
]
# `city` enters via one-hot or target-encoded separately
