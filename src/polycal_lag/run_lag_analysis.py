"""Multi-city lag analysis: Polymarket tick price vs NBM forecast-implied bucket probability.

For each bucket market with substantial price movement, compute:
- Forecast μ(T) trajectory at hourly issue times.
- σ(T) = 0.5 + 0.45*sqrt(T) (empirical from NYC July 2025).
- P(YES) = Φ((upper + 0.5 - μ)/σ) - Φ((lower - 0.5 - μ)/σ).
- Cross-correlate tick price vs forecast P(YES) curve on a 5-min grid.
- Report lag (positive: market lags forecast).
"""
from __future__ import annotations
import re, datetime as dt, json, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd, requests
from scipy.stats import norm

from .stations import STATIONS, peak_utc
from .forecast_multi import forecast_max_for_day

GAMMA = "https://gamma-api.polymarket.com"
DATA  = "https://data-api.polymarket.com"
TICKS_CACHE = Path("cache/polymarket/ticks")
TICKS_CACHE.mkdir(parents=True, exist_ok=True)

_session = requests.Session()
_session.headers.update({"User-Agent":"polycal/0.2"})


def sigma_for(T_h):
    return 0.5 + 0.45 * np.sqrt(max(0.5, T_h))


def bucket_prob(mu, lo, hi, T):
    s = sigma_for(T)
    return float(norm.cdf((hi + 0.5 - mu) / s) - norm.cdf((lo - 0.5 - mu) / s))


def parse_bucket(q):
    m = re.search(r"between\s+(\d+)\s*-\s*(\d+)", q or "", re.I)
    if m: return int(m.group(1)), int(m.group(2))
    return None


def discover_events(city, start, end):
    out, offset = [], 0
    while True:
        r = _session.get(f"{GAMMA}/events", params={
            "series_slug": f"{city}-daily-weather", "closed":"true", "archived":"false",
            "end_date_min": f"{start}T00:00:00Z",
            "end_date_max": f"{(end + dt.timedelta(days=1)).isoformat()}T00:00:00Z",
            "limit": 100, "offset": offset, "order":"endDate", "ascending":"true",
        }, timeout=30)
        if not r.ok: break
        batch = r.json()
        if not batch: break
        out.extend(batch)
        if len(batch) < 100: break
        offset += 100
    return out


def fetch_ticks(market):
    """Fetch all trades for one market via data-api."""
    cid = market["conditionId"]
    fp = TICKS_CACHE / f"{cid}.parquet"
    if fp.exists():
        return pd.read_parquet(fp)
    rows, offset = [], 0
    while True:
        r = _session.get(f"{DATA}/trades", params={
            "market": cid, "limit": 1000, "offset": offset, "takerOnly":"false"
        }, timeout=30)
        if not r.ok: break
        batch = r.json()
        if not batch: break
        rows.extend(batch)
        if len(batch) < 1000 or offset >= 9000: break
        offset += 1000
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["ts_utc"] = pd.to_datetime(df["timestamp"], unit="s")
    df["price"] = df["price"].astype(float)
    df["size"] = df["size"].astype(float)
    # YES asset = the YES token (outcomeIndex 0 typically)
    df.to_parquet(fp, index=False)
    return df


def yes_price_series(ticks, yes_token):
    """Return DataFrame ts_utc, yes_price reflecting last-trade YES price."""
    if ticks.empty: return ticks
    t = ticks.copy()
    # Each trade is either on YES asset (price = YES price) or NO asset (price = NO price)
    # Convert NO trades to YES-equivalent: yes_price = 1 - no_price
    t["yes_price"] = np.where(t["asset"].astype(str) == str(yes_token),
                              t["price"], 1.0 - t["price"])
    t = t.sort_values("ts_utc")[["ts_utc","yes_price"]].reset_index(drop=True)
    return t


def market_lag(city, ev, m, ticks, forecasts_by_lead):
    """Compute lag for one market. Return dict or None."""
    bucket = parse_bucket(m.get("question"))
    if not bucket: return None
    lo, hi = bucket
    resolve_date = pd.to_datetime(ev["endDate"]).date()
    peak = peak_utc(city, resolve_date)
    yes_token = m.get("clobTokenIds")
    if isinstance(yes_token, str):
        import json as _json
        try: yes_token = _json.loads(yes_token)[0]
        except: pass

    yp = yes_price_series(ticks, yes_token)
    if len(yp) < 30: return None
    span = yp["yes_price"].max() - yp["yes_price"].min()
    if span < 0.3: return None

    # Build forecast trajectory: a sequence of (lead_h, forecast μ, P_yes)
    fc_points = []
    for issue_utc, mu in forecasts_by_lead.items():
        if mu is None: continue
        lead_h = (peak - issue_utc).total_seconds() / 3600
        if lead_h < -1 or lead_h > 96: continue
        fc_points.append({"issue_utc": issue_utc, "lead_h": lead_h,
                          "mu": mu, "p_yes": bucket_prob(mu, lo, hi, max(0.5, lead_h))})
    if len(fc_points) < 6: return None
    fc = pd.DataFrame(fc_points).sort_values("issue_utc").reset_index(drop=True)

    # Resample both onto 5-min grid covering the overlap window
    t_start = max(yp["ts_utc"].iloc[0], fc["issue_utc"].iloc[0])
    t_end   = min(yp["ts_utc"].iloc[-1], fc["issue_utc"].iloc[-1])
    if (t_end - t_start).total_seconds() < 6*3600: return None
    grid = pd.date_range(t_start, t_end, freq="5min")
    yp_g = pd.Series(np.interp(grid.astype("int64"),
                                yp["ts_utc"].astype("int64").values,
                                yp["yes_price"].values), index=grid)
    fc_g = pd.Series(np.interp(grid.astype("int64"),
                                fc["issue_utc"].astype("int64").values,
                                fc["p_yes"].values), index=grid)

    # Cross-correlation at lags in [-180, +180] minutes (36 steps of 5min)
    yp_arr, fc_arr = yp_g.values, fc_g.values
    if yp_arr.std() < 0.02 or fc_arr.std() < 0.02: return None
    best_lag_min, best_corr = 0, -2
    for lag_steps in range(-36, 37):
        if lag_steps > 0:
            a, b = fc_arr[:-lag_steps], yp_arr[lag_steps:]
        elif lag_steps < 0:
            a, b = fc_arr[-lag_steps:], yp_arr[:lag_steps]
        else:
            a, b = fc_arr, yp_arr
        if len(a) < 24: continue
        if a.std() < 1e-6 or b.std() < 1e-6: continue
        c = float(np.corrcoef(a, b)[0,1])
        if c > best_corr:
            best_corr, best_lag_min = c, lag_steps * 5

    return {
        "city": city, "resolve_date": str(resolve_date),
        "question": m["question"], "market_id": m["id"], "bucket_lo": lo, "bucket_hi": hi,
        "n_ticks": int(len(yp)), "price_span": float(span),
        "lag_minutes": int(best_lag_min), "correlation": float(best_corr),
        "yes_won": int(m.get("outcomePrices", '["0","0"]')[1:-1].split(",")[0].strip('"') == "1"),
    }


def fc_for_date(args):
    city, date, issue_utc = args
    try:
        res = forecast_max_for_day(city, date, issue_utc, allow_obs=True)
        return (city, date, issue_utc, res["forecast_max_f"] if res else None)
    except Exception as e:
        return (city, date, issue_utc, None)


def run(start: dt.date, end: dt.date, cities: list[str]):
    print(f"Window: {start} → {end}, cities: {','.join(cities)}")

    # 1) Discover events per city
    print("\n[1/4] Discovering events...")
    events = {}
    for city in cities:
        evs = discover_events(city, start, end)
        events[city] = evs
        print(f"  {city:>8}: {len(evs)} events")

    # 2) Tick-fetch all bucket markets in parallel
    print("\n[2/4] Fetching ticks (parallel threads)...")
    tasks = []
    for city, evs in events.items():
        for ev in evs:
            for m in (ev.get("markets") or []):
                if parse_bucket(m.get("question")):
                    tasks.append((city, ev, m))
    print(f"  total bucket markets to fetch: {len(tasks):,}")
    ticks_by_market = {}
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = {ex.submit(fetch_ticks, m): (city, ev, m) for city, ev, m in tasks}
        done = 0
        for f in as_completed(futs):
            city, ev, m = futs[f]
            try: ticks_by_market[m["id"]] = f.result()
            except Exception as e: ticks_by_market[m["id"]] = pd.DataFrame()
            done += 1
            if done % 200 == 0:
                print(f"    {done}/{len(tasks)}", flush=True)
    print(f"  done: {len(ticks_by_market):,} markets")

    # 3) Compute forecasts for each (city, date, hourly issue)
    # For each event date, we need forecast μ at issue_times from -72h to 0h before peak,
    # at hourly fidelity. Use processes for GRIB extraction.
    print("\n[3/4] Pulling NBM forecasts...")
    fc_tasks = []
    for city, evs in events.items():
        for ev in evs:
            d = pd.to_datetime(ev["endDate"]).date()
            p = peak_utc(city, d)
            for k in range(0, 73, 3):  # every 3h saves time, still informative
                issue = p - pd.Timedelta(hours=k)
                fc_tasks.append((city, d, issue))
    # Dedup
    fc_tasks = list({(c,d,i): (c,d,i) for c,d,i in fc_tasks}.values())
    print(f"  {len(fc_tasks)} (city,date,issue) triples")

    fc_results = {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(fc_for_date, t) for t in fc_tasks]
        done = 0
        for f in as_completed(futs):
            try:
                c, d, i, mu = f.result()
                fc_results[(c, str(d), i)] = mu
            except Exception: pass
            done += 1
            if done % 200 == 0:
                print(f"    {done}/{len(fc_tasks)}", flush=True)
    print(f"  done: {sum(1 for v in fc_results.values() if v is not None):,} non-null forecasts")

    # 4) Per-market lag analysis
    print("\n[4/4] Computing lag per market...")
    results = []
    for city, ev, m in tasks:
        d = str(pd.to_datetime(ev["endDate"]).date())
        # Collect forecasts for this (city, date)
        fc_for_market = {i: mu for (c, dd, i), mu in fc_results.items()
                         if c == city and dd == d}
        ticks = ticks_by_market.get(m["id"], pd.DataFrame())
        if ticks.empty: continue
        try:
            r = market_lag(city, ev, m, ticks, fc_for_market)
            if r: results.append(r)
        except Exception as e:
            continue
    print(f"  markets with computable lag: {len(results):,}")

    out = pd.DataFrame(results)
    out.to_parquet("results/lag_analysis_apr_may_2026.parquet", index=False)
    print(f"\nSaved results/lag_analysis_apr_may_2026.parquet ({len(out)} rows)")
    return out


if __name__ == "__main__":
    import sys
    start = dt.date.fromisoformat(sys.argv[1] if len(sys.argv) > 1 else "2026-04-01")
    end   = dt.date.fromisoformat(sys.argv[2] if len(sys.argv) > 2 else "2026-05-12")
    cities = sys.argv[3].split(",") if len(sys.argv) > 3 else list(STATIONS.keys())
    df = run(start, end, cities)
    if not df.empty:
        print("\n=== Lag distribution (positive: price LAGS forecast; negative: price LEADS) ===")
        print(df["lag_minutes"].describe(percentiles=[.1,.25,.5,.75,.9]).round(1).to_string())
        print(f"\nMedian lag: {df['lag_minutes'].median():.0f} min  |  Mean corr: {df['correlation'].mean():.2f}")
        print("\nLag by city:")
        print(df.groupby("city").agg(
            n=("lag_minutes","size"),
            lag_median=("lag_minutes","median"),
            lag_mean=("lag_minutes","mean"),
            corr_mean=("correlation","mean"),
        ).round(1).to_string())
