"""Does Polymarket price the NBM cold bias?

For each bucket market in cache, compare:
  - market YES price at some lead time
  - naive NBM forecast P(YES)            (σ from RMSE, mu uncorrected)
  - bias-corrected forecast P(YES)       (μ + 0.81 °F)
  - empirical winrate

Aggregate by bucket_offset = bucket_midpoint − forecast_μ. Save a parquet
that downstream scripts can join with.
"""
from __future__ import annotations
import json, re, sys, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd, requests
from scipy.stats import norm

from .stations import STATIONS, peak_utc
from .forecast_batch import fxx_window

EXT = Path("cache/forecasts/grib_extracts")
TICKS = Path("cache/polymarket/ticks")

_sess = requests.Session()
_sess.headers.update({"User-Agent":"polycal/0.4"})


def fc_max_for(city, date, lead_h):
    p = peak_utc(city, date)
    issue = (p - pd.Timedelta(hours=lead_h)).floor("h")
    vals = []
    for fx in fxx_window(issue, date):
        fp = EXT / f"{issue:%Y%m%dT%HZ}_f{fx:02d}.parquet"
        if fp.exists():
            v = pd.read_parquet(fp).iloc[0].get(city)
            if v is not None: vals.append(float(v))
    return max(vals) if vals else None


def sigma_for(T): return 0.5 + 0.45 * np.sqrt(max(0.5, T))


def naive_p(mu, lo, hi, T):
    s = sigma_for(T)
    return float(norm.cdf((hi+0.5-mu)/s) - norm.cdf((lo-0.5-mu)/s))


def yes_price_at(condition_id, yes_token, target_ts):
    fp = TICKS / f"{condition_id}.parquet"
    if not fp.exists(): return None
    t = pd.read_parquet(fp)
    if t.empty: return None
    t = t.copy()
    t["yes_price"] = np.where(t["asset"].astype(str)==str(yes_token), t["price"], 1-t["price"])
    t["ts_utc"] = pd.to_datetime(t["ts_utc"])
    before = t[t["ts_utc"] <= target_ts]
    return float(before["yes_price"].iloc[-1]) if len(before) else None


def discover_bucket_markets(start, end):
    out = []
    for c in STATIONS:
        offset = 0
        while True:
            r = _sess.get("https://gamma-api.polymarket.com/events", params={
                "series_slug": f"{c}-daily-weather", "closed":"true",
                "end_date_min": f"{start}T00:00:00Z",
                "end_date_max": f"{(end + dt.timedelta(days=1)).isoformat()}T00:00:00Z",
                "limit":100,"offset":offset}, timeout=30)
            b = r.json()
            if not b: break
            for ev in b:
                for m in (ev.get("markets") or []):
                    qm = re.search(r"between\s+(\d+)\s*-\s*(\d+)", m.get("question") or "", re.I)
                    if not qm: continue
                    try:
                        op = json.loads(m["outcomePrices"])
                        yes_won = int(float(op[0]) > 0.5)
                    except: continue
                    out.append({
                        "city": c, "date": pd.to_datetime(ev["endDate"]).date(),
                        "lo": int(qm.group(1)), "hi": int(qm.group(2)),
                        "yes_won": yes_won, "cid": m["conditionId"],
                        "tokens": json.loads(m["clobTokenIds"]),
                        "outcomes": json.loads(m["outcomes"]),
                    })
            if len(b) < 100: break
            offset += 100
    return out


def run(start, end, leads=(1, 3, 6, 12, 24, 48)):
    markets = discover_bucket_markets(start, end)
    print(f"discovered {len(markets):,} bucket markets")
    rows = []
    for m in markets:
        yes_token = next(t for t, o in zip(m["tokens"], m["outcomes"])
                         if str(o).lower()=="yes")
        for L in leads:
            mu = fc_max_for(m["city"], m["date"], L)
            if mu is None: continue
            target = peak_utc(m["city"], m["date"]) - pd.Timedelta(hours=L)
            yp = yes_price_at(m["cid"], yes_token, target)
            if yp is None: continue
            mid = (m["lo"] + m["hi"]) / 2
            rows.append({
                "city": m["city"], "date": str(m["date"]),
                "lo": m["lo"], "hi": m["hi"],
                "lead_h": L, "forecast_mu": mu, "bucket_offset": mid - mu,
                "market_yes_price": yp, "yes_won": m["yes_won"],
                "naive_p": naive_p(mu, m["lo"], m["hi"], L),
                "bias_p":  naive_p(mu + 0.81, m["lo"], m["hi"], L),
            })
    df = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    df.to_parquet("results/forecast_vs_market_apr_may_2026.parquet", index=False)
    print(f"saved results/forecast_vs_market_apr_may_2026.parquet rows={len(df)}")
    return df


if __name__ == "__main__":
    s = dt.date.fromisoformat(sys.argv[1] if len(sys.argv) > 1 else "2026-04-01")
    e = dt.date.fromisoformat(sys.argv[2] if len(sys.argv) > 2 else "2026-05-12")
    df = run(s, e)

    # Headline aggregate at T=12h
    sub = df[df["lead_h"] == 12].copy()
    sub["off_bin"] = np.round(sub["bucket_offset"]).astype(int).clip(-5, 5)
    print(f"\nAt T=12h, n={len(sub):,}:")
    print(f"{'off':>5} {'n':>5} {'mkt_p':>8} {'naive':>8} {'bias_p':>8} {'empir':>8}  {'EV_yes%':>8}")
    print("-"*60)
    for off, g in sorted(sub.groupby("off_bin"), key=lambda x: x[0]):
        if len(g) < 15: continue
        mp = g["market_yes_price"].mean()
        emp = g["yes_won"].mean()
        ev = (emp - mp) / mp * 100 if mp > 0 else 0
        print(f"{off:>+5} {len(g):>5} {mp:>8.3f} {g['naive_p'].mean():>8.3f} "
              f"{g['bias_p'].mean():>8.3f} {emp:>8.3f}  {ev:>+7.1f}%")

    from sklearn.metrics import brier_score_loss
    print(f"\nBrier scores at T=12h (lower=better):")
    print(f"  naive NBM:           {brier_score_loss(sub['yes_won'], sub['naive_p']):.4f}")
    print(f"  bias-corrected NBM:  {brier_score_loss(sub['yes_won'], sub['bias_p']):.4f}")
    print(f"  Polymarket price:    {brier_score_loss(sub['yes_won'], sub['market_yes_price']):.4f}")
