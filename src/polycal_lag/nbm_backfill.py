"""Full NBM backfill — one big parquet, no per-file droppings.

For every (city, date) in the canonical weather window, pull NBM forecasts
at standard leads (1, 3, 6, 12, 24, 48 h). Dedup at (issue_utc, fxx) so a
single GRIB is fetched once even when used by multiple (city, lead) targets.

Strategy:
  - Discover unique (issue_utc, fxx) targets from the canonical event list
  - Skip those already in cache parquet (resumable)
  - ProcessPool workers fetch, parse, extract 9 cities, return rows
  - Main process accumulates in RAM, flushes every CHUNK_ROWS to one parquet
  - remove_grib=True so the ~3 MB GRIB subsets get deleted right after parse
"""
from __future__ import annotations
import datetime as dt, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd
import warnings

from .stations import STATIONS, peak_utc
from .forecast_batch import _grid_indices, fxx_window

OUT = Path("cache/forecasts/nbm_extracts.parquet")
TMP = OUT.with_suffix(".tmp.parquet")
TMP_REGEX = ":TMP:2 m above ground:"
LEADS = [1, 3, 6, 12, 24, 48]
CHUNK_ROWS = 2000

warnings.filterwarnings("ignore")


def pull_one(args) -> dict | None:
    """Worker: pull one GRIB, extract t2m at all 9 stations, return one row.
    remove_grib=True keeps disk clean."""
    issue_utc, fxx = args
    from herbie import Herbie
    try:
        h = Herbie(issue_utc.to_pydatetime(), model="nbm", product="co", fxx=fxx,
                   priority=["aws"], verbose=False)
        if getattr(h, "grib", None) is None:
            return None
        ds = h.xarray(TMP_REGEX, remove_grib=True)  # delete subset after parse
        arr = np.asarray(ds["t2m"])
        idx = _grid_indices()
        row = {"issue_utc": issue_utc, "fxx": fxx,
               "valid_utc": issue_utc + pd.Timedelta(hours=fxx)}
        for c, (i, j) in idx.items():
            row[c] = (float(arr[i, j]) - 273.15) * 9 / 5 + 32  # K -> °F
        return row
    except Exception as e:
        return {"issue_utc": issue_utc, "fxx": fxx, "_error": str(e)[:120]}


def discover_targets(start: dt.date, end: dt.date) -> set:
    """Set of unique (issue_utc, fxx) pairs needed across all cities/dates/leads."""
    pairs = set()
    n_days = (end - start).days + 1
    for d in (start + dt.timedelta(days=i) for i in range(n_days)):
        for city in STATIONS:
            p = peak_utc(city, d)
            for L in LEADS:
                issue = (p - pd.Timedelta(hours=L)).floor("h")
                for fx in fxx_window(issue, d):
                    pairs.add((issue, fx))
    return pairs


def load_existing() -> tuple[pd.DataFrame, set]:
    """Resume: load already-pulled rows + set of (issue_utc, fxx) keys done."""
    if not OUT.exists():
        return pd.DataFrame(), set()
    df = pd.read_parquet(OUT)
    if df.empty:
        return df, set()
    done = set(zip(pd.to_datetime(df["issue_utc"]), df["fxx"].astype(int)))
    return df, done


def flush(rows: list[dict], existing: pd.DataFrame):
    """Append new rows to OUT atomically (write to TMP, rename)."""
    if not rows:
        return existing
    new = pd.DataFrame(rows)
    # Drop error rows from main parquet — keep only successful
    if "_error" in new.columns:
        new = new[new["_error"].isna()].drop(columns=["_error"])
    if new.empty:
        return existing
    combined = pd.concat([existing, new], ignore_index=True) if not existing.empty else new
    OUT.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(TMP, index=False, compression="zstd")
    TMP.replace(OUT)
    return combined


def run(start: dt.date, end: dt.date, workers: int = 8):
    print(f"NBM full backfill: {start} → {end}, {len(STATIONS)} cities, leads {LEADS}")
    targets = discover_targets(start, end)
    print(f"unique (issue, fxx) targets:  {len(targets):,}")

    existing, done = load_existing()
    todo = sorted(targets - done, key=lambda x: (x[0], x[1]))
    print(f"already pulled:               {len(done):,}")
    print(f"to pull this run:             {len(todo):,}")
    if not todo:
        print("nothing to do.")
        return existing

    pending_rows = []
    t0 = time.time()
    completed = 0
    errors = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(pull_one, t) for t in todo]
        for f in as_completed(futs):
            r = f.result()
            if r is not None:
                pending_rows.append(r)
                if "_error" in r and r.get("_error"):
                    errors += 1
            completed += 1
            if completed % 200 == 0:
                rate = completed / (time.time() - t0)
                eta = (len(todo) - completed) / max(rate, 0.01) / 60
                print(f"  {completed:>6}/{len(todo)}  rate={rate:.2f}/s  "
                      f"eta={eta:.1f}min  errors={errors}  buf={len(pending_rows)}",
                      flush=True)
            if len(pending_rows) >= CHUNK_ROWS:
                existing = flush(pending_rows, existing)
                pending_rows = []

    existing = flush(pending_rows, existing)
    print(f"\nDone. Total rows: {len(existing):,}, errors: {errors}")
    print(f"Output: {OUT} ({OUT.stat().st_size/1024/1024:.1f} MB)")
    return existing


if __name__ == "__main__":
    import sys
    s = dt.date.fromisoformat(sys.argv[1] if len(sys.argv) > 1 else "2025-01-21")
    e = dt.date.fromisoformat(sys.argv[2] if len(sys.argv) > 2 else "2026-05-05")
    w = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    run(s, e, workers=w)
