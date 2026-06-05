"""Assemble training dataset from cached weather markets + features."""
from __future__ import annotations
import ast, json, re, datetime as dt
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from polycal_lag.stations import STATIONS, peak_utc
from .features import FeatureRow, build_features

WEATHER_LIST = Path("results/weather_markets_canonical.parquet")


def parse_bucket(question: str) -> tuple[int, int] | None:
    m = re.search(r"between\s+(\d+)\s*-\s*(\d+)", question or "", re.I)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _parse_outcome_prices(v):
    """outcome_prices is stored as either JSON (`["1","0"]`) or Python-repr
    (`['1', '0']`) depending on source. Handle both."""
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return list(v)
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(v)
            except Exception:
                return None
    return None


def discover_event_buckets(start: dt.date, end: dt.date) -> pd.DataFrame:
    """From the canonical HF list, return one row per (city, date, bucket, yes_won).

    Filters to closed bucket markets in the window with parseable buckets and
    a known resolution.
    """
    if not WEATHER_LIST.exists():
        raise FileNotFoundError(f"{WEATHER_LIST} missing — run the HF download first.")
    w = pd.read_parquet(WEATHER_LIST)
    w["end_date_d"] = pd.to_datetime(w["end_date"]).dt.date
    w = w[(w["end_date_d"] >= start) & (w["end_date_d"] <= end)]
    rows = []
    for _, m in w.iterrows():
        b = parse_bucket(m["question"])
        if not b:
            continue
        op = _parse_outcome_prices(m["outcome_prices"])
        if not op:
            continue
        try:
            yes_won = int(float(op[0]) > 0.5)
        except Exception:
            continue
        rows.append({
            "city": m["city"], "date": m["end_date_d"],
            "bucket_lo": b[0], "bucket_hi": b[1],
            "yes_won": yes_won,
            "condition_id": m["condition_id"],
            "yes_token": m.get("token1"),  # Polymarket convention: token1 = YES
        })
    return pd.DataFrame(rows)


def build_dataset(start: dt.date, end: dt.date, leads: list[int] = (6,),
                  show_progress: bool = True) -> pd.DataFrame:
    """For every bucket market in [start, end], compute feature row at each lead."""
    bucket_events = discover_event_buckets(start, end)
    out = []
    iterator = bucket_events.iterrows()
    if show_progress:
        iterator = tqdm(iterator, total=len(bucket_events), desc="features")
    for _, ev in iterator:
        for L in leads:
            r = FeatureRow(
                city=ev["city"], resolve_date=ev["date"],
                bucket_lo=ev["bucket_lo"], bucket_hi=ev["bucket_hi"],
                decision_lead_h=L, yes_won=int(ev["yes_won"]),
                condition_id=ev.get("condition_id"),
                yes_token=ev.get("yes_token"),
            )
            try:
                f = build_features(r)
                f["condition_id"] = ev["condition_id"]
                out.append(f)
            except Exception as e:
                # skip rows where forecast cache is missing — we'll handle
                # missingness downstream
                continue
    df = pd.DataFrame(out)
    return df


if __name__ == "__main__":
    import sys
    s = dt.date.fromisoformat(sys.argv[1] if len(sys.argv) > 1 else "2025-01-02")
    e = dt.date.fromisoformat(sys.argv[2] if len(sys.argv) > 2 else "2026-05-05")
    df = build_dataset(s, e)
    Path("results").mkdir(exist_ok=True)
    out_path = f"results/ml_dataset_{s}_{e}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved {out_path}  rows={len(df):,}, cols={len(df.columns)}")
    print(f"Date range: {df['resolve_date'].min()} → {df['resolve_date'].max()}")
    print(f"Cities: {df['city'].value_counts().to_dict()}")
    print(f"Missing features (% null per col):")
    nulls = (df.isnull().mean() * 100).sort_values(ascending=False)
    print(nulls[nulls > 0].head(20).round(1).to_string())
