"""Multi-city lag analysis, v2: batched GRIB pulls.

Strategy:
  - Per (city, date), pick issue times at peak - {1, 3, 6, 12, 24, 48}h.
  - Determine ALL unique (issue_utc, fxx) GRIB requests across all (city, date).
  - Pull each GRIB ONCE, extract t2m at all 9 cities, cache.
  - Aggregate per (city, date, issue) forecast_max from the extracts.
"""
from __future__ import annotations
import re, datetime as dt, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd, requests
from scipy.stats import norm

from .stations import STATIONS, peak_utc
from .forecast_batch import extract_all_cities, fxx_window

GAMMA = "https://gamma-api.polymarket.com"
DATA  = "https://data-api.polymarket.com"
TICKS = Path("cache/polymarket/ticks"); TICKS.mkdir(parents=True, exist_ok=True)
LEADS = [1, 3, 6, 12, 24, 48]

_sess = requests.Session()
_sess.headers.update({"User-Agent":"polycal/0.3"})


def parse_bucket(q):
    m = re.search(r"between\s+(\d+)\s*-\s*(\d+)", q or "", re.I)
    return (int(m.group(1)), int(m.group(2))) if m else None


def discover(city, start, end):
    out, offset = [], 0
    while True:
        r = _sess.get(f"{GAMMA}/events", params={
            "series_slug": f"{city}-daily-weather", "closed":"true", "archived":"false",
            "end_date_min": f"{start}T00:00:00Z",
            "end_date_max": f"{(end + dt.timedelta(days=1)).isoformat()}T00:00:00Z",
            "limit": 100, "offset": offset, "order":"endDate", "ascending":"true",
        }, timeout=30)
        if not r.ok: break
        b = r.json()
        if not b: break
        out.extend(b)
        if len(b) < 100: break
        offset += 100
    return out


def fetch_ticks(cid):
    fp = TICKS / f"{cid}.parquet"
    if fp.exists(): return pd.read_parquet(fp)
    rows, offset = [], 0
    while True:
        r = _sess.get(f"{DATA}/trades",
                      params={"market": cid, "limit": 1000, "offset": offset, "takerOnly":"false"},
                      timeout=30)
        if not r.ok: break
        b = r.json()
        if not b: break
        rows.extend(b)
        if len(b) < 1000 or offset >= 9000: break
        offset += 1000
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["ts_utc"] = pd.to_datetime(df["timestamp"], unit="s")
    df["price"] = df["price"].astype(float)
    df.to_parquet(fp, index=False)
    return df


def sigma_for(T):
    return 0.5 + 0.45 * np.sqrt(max(0.5, T))


def bucket_prob(mu, lo, hi, T):
    s = sigma_for(T)
    return float(norm.cdf((hi + 0.5 - mu) / s) - norm.cdf((lo - 0.5 - mu) / s))


def pull_one(issue_fxx):
    issue_utc, fxx = issue_fxx
    return (issue_utc, fxx, extract_all_cities(issue_utc, fxx))


def run(start, end, cities):
    print(f"Window: {start} → {end}, cities: {','.join(cities)}\n")

    # Discover events
    print("[1/4] Discovering events...")
    events = {c: discover(c, start, end) for c in cities}
    for c, evs in events.items(): print(f"  {c:>8}: {len(evs)} events")

    # Tick-fetch all bucket markets
    print("\n[2/4] Fetching ticks...")
    tasks = []
    for c, evs in events.items():
        for ev in evs:
            for m in (ev.get("markets") or []):
                if parse_bucket(m.get("question")):
                    tasks.append((c, ev, m))
    print(f"  total bucket markets: {len(tasks):,}")
    ticks_by = {}
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = {ex.submit(fetch_ticks, m["conditionId"]): (c, ev, m) for c, ev, m in tasks}
        done = 0
        for f in as_completed(futs):
            c, ev, m = futs[f]
            try: ticks_by[m["id"]] = f.result()
            except: ticks_by[m["id"]] = pd.DataFrame()
            done += 1
            if done % 500 == 0: print(f"    {done}/{len(tasks)}", flush=True)
    print(f"  done")

    # Compute unique (issue_utc, fxx) pairs across all (city, date, lead)
    print("\n[3/4] Pulling NBM forecasts (batched GRIBs)...")
    pairs = set()
    target_index = []  # rows of (city, date, issue_utc, [fxx_list])
    for c, evs in events.items():
        for ev in evs:
            d = pd.to_datetime(ev["endDate"]).date()
            p = peak_utc(c, d)
            for L in LEADS:
                issue = (p - pd.Timedelta(hours=L)).floor("h")
                fxxs = fxx_window(issue, d)
                target_index.append((c, d, issue, L, fxxs))
                for fx in fxxs: pairs.add((issue, fx))
    pairs = sorted(pairs, key=lambda x: (x[0], x[1]))
    print(f"  unique (issue, fxx) GRIB pulls: {len(pairs):,}")
    print(f"  per-(city,date,lead) targets: {len(target_index):,}")

    grib_results = {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(pull_one, p) for p in pairs]
        done = 0
        t0 = time.time()
        for f in as_completed(futs):
            try:
                issue, fxx, res = f.result()
                if res: grib_results[(issue, fxx)] = res
            except Exception: pass
            done += 1
            if done % 100 == 0:
                rate = done / (time.time()-t0+0.1)
                eta = (len(pairs)-done) / max(rate, 0.1)
                print(f"    {done}/{len(pairs)}  rate={rate:.1f}/s  eta={eta/60:.1f}min", flush=True)
    print(f"  done: {len(grib_results):,}/{len(pairs):,} successful GRIBs")

    # Aggregate per (city, date, lead) forecast max
    print("\n[4/4] Computing forecast μ per (city, date, lead) and lag per market...")
    fc_max = {}  # (city, date, lead) -> max forecast °F
    for c, d, issue, L, fxxs in target_index:
        vals = []
        for fx in fxxs:
            r = grib_results.get((issue, fx))
            if r and r.get(c) is not None: vals.append(r[c])
        if vals:
            fc_max[(c, str(d), L)] = max(vals)

    print(f"  non-null forecast points: {len(fc_max):,}")

    # Per-market lag
    results = []
    for c, ev, m in tasks:
        d = pd.to_datetime(ev["endDate"]).date()
        bucket = parse_bucket(m.get("question"))
        if not bucket: continue
        lo, hi = bucket
        ticks = ticks_by.get(m["id"], pd.DataFrame())
        if len(ticks) < 30: continue

        # YES side: m has clobTokenIds JSON; outcome[0]=Yes index gives yes_token
        import json
        try:
            tokens = json.loads(m["clobTokenIds"]) if isinstance(m["clobTokenIds"], str) else m["clobTokenIds"]
            outcomes = json.loads(m["outcomes"]) if isinstance(m["outcomes"], str) else m["outcomes"]
            yes_token = next(t for t, o in zip(tokens, outcomes) if str(o).lower()=="yes")
        except Exception:
            yes_token = tokens[0]

        t = ticks.copy()
        t["yes_price"] = np.where(t["asset"].astype(str)==str(yes_token), t["price"], 1-t["price"])
        t = t.sort_values("ts_utc").reset_index(drop=True)
        span = float(t["yes_price"].max() - t["yes_price"].min())
        if span < 0.3: continue

        # Forecast trajectory: 6 points at lead = 1,3,6,12,24,48h
        peak = peak_utc(c, d)
        fc_pts = []
        for L in LEADS:
            mu = fc_max.get((c, str(d), L))
            if mu is None: continue
            t_issue = (peak - pd.Timedelta(hours=L)).floor("h")
            fc_pts.append({"ts": t_issue, "lead_h": L,
                           "p_yes": bucket_prob(mu, lo, hi, L)})
        if len(fc_pts) < 4: continue
        fc = pd.DataFrame(fc_pts).sort_values("ts").reset_index(drop=True)

        # 5-min grid on overlap.
        # CRITICAL: ts_utc in parquet is datetime64[ms]; pd.date_range gives ns.
        # Force both to ns before astype(int64), else np.interp sees 1e6× scale
        # mismatch and clips everything to the endpoint value (silent bug).
        t_start = max(t["ts_utc"].iloc[0], fc["ts"].iloc[0])
        t_end   = min(t["ts_utc"].iloc[-1], fc["ts"].iloc[-1])
        if (t_end - t_start).total_seconds() < 6*3600: continue
        grid = pd.date_range(t_start, t_end, freq="5min")
        gx = grid.values.astype("datetime64[ns]").astype("int64")
        xp_t = t["ts_utc"].values.astype("datetime64[ns]").astype("int64")
        xp_f = fc["ts"].values.astype("datetime64[ns]").astype("int64")
        yp = np.interp(gx, xp_t, t["yes_price"].values)
        fp = np.interp(gx, xp_f, fc["p_yes"].values)
        if yp.std() < 0.02 or fp.std() < 0.02: continue

        best_lag, best_c = 0, -2.0
        for k in range(-36, 37):  # ±180 min in 5-min steps
            if k > 0: a, b = fp[:-k], yp[k:]
            elif k < 0: a, b = fp[-k:], yp[:k]
            else: a, b = fp, yp
            if len(a) < 24 or a.std() < 1e-6 or b.std() < 1e-6: continue
            cc = float(np.corrcoef(a, b)[0,1])
            if cc > best_c: best_c, best_lag = cc, k*5

        # YES won — derive from outcomePrices
        op = m.get("outcomePrices")
        try:
            op = json.loads(op) if isinstance(op, str) else op
            yes_won = 1 if (op and float(op[0]) > 0.5) else 0
        except: yes_won = None

        results.append({
            "city": c, "resolve_date": str(d),
            "question": m["question"], "market_id": m["id"],
            "bucket_lo": lo, "bucket_hi": hi,
            "n_ticks": int(len(t)), "price_span": span,
            "lag_minutes": int(best_lag), "correlation": float(best_c),
            "yes_won": yes_won,
        })

    out = pd.DataFrame(results)
    Path("results").mkdir(exist_ok=True)
    out.to_parquet("results/lag_analysis_apr_may_2026.parquet", index=False)
    print(f"\nSaved results/lag_analysis_apr_may_2026.parquet  rows={len(out)}")
    return out


if __name__ == "__main__":
    start = dt.date.fromisoformat(sys.argv[1] if len(sys.argv) > 1 else "2026-04-01")
    end   = dt.date.fromisoformat(sys.argv[2] if len(sys.argv) > 2 else "2026-05-12")
    cities = sys.argv[3].split(",") if len(sys.argv) > 3 else list(STATIONS.keys())
    df = run(start, end, cities)
    if not df.empty:
        print("\n=== Lag distribution (+ = price LAGS forecast, − = price LEADS) ===")
        print(df["lag_minutes"].describe(percentiles=[.1,.25,.5,.75,.9]).round(1).to_string())
        print(f"\nMedian lag: {df['lag_minutes'].median():.0f} min  |  Mean corr: {df['correlation'].mean():.2f}")
        print("\nBy city:")
        print(df.groupby("city").agg(
            n=("lag_minutes","size"),
            lag_median_min=("lag_minutes","median"),
            corr_mean=("correlation","mean"),
            span_mean=("price_span","mean"),
        ).round(2).to_string())
