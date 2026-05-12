"""CLI orchestration: build dataset → calibrate → plot."""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from .build_dataset import build_rows
from .calibration import bin_and_aggregate
from .config import (
    ACTUALS_SOURCE_DEFAULT,
    DERIVED_CACHE,
    LEAD_TIMES_HOURS,
    MODEL_DEFAULT,
)
from .plot import plot_calibration


def _parse_date(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def _parse_lead_times(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def run(
    start: dt.date, end: dt.date,
    lead_times: list[int], model: str,
    out_dir: Path,
    actuals_source: str = ACTUALS_SOURCE_DEFAULT,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Building dataset: {start} → {end}, model={model}, "
          f"actuals={actuals_source}, lead_times={lead_times}")
    ds = build_rows(start, end, lead_times, model=model, actuals_source=actuals_source)
    if ds.empty:
        print("No rows built (no forecast/actuals overlap). Exiting.")
        return

    tag = f"{model}_{start.isoformat()}_{end.isoformat()}"
    ds_path = DERIVED_CACHE / f"dataset_{tag}.parquet"
    ds.to_parquet(ds_path, index=False)
    print(f"Dataset: {len(ds)} rows → {ds_path}")

    cal = bin_and_aggregate(ds)
    cal_path = DERIVED_CACHE / f"calibration_{tag}.parquet"
    cal.to_parquet(cal_path, index=False)
    print(f"Calibration table: {len(cal)} buckets → {cal_path}")

    png_path = out_dir / f"calibration_{tag}.png"
    plot_calibration(cal, ds, png_path,
                     title=f"NBM forecast calibration — NYC (KLGA) — {start}…{end}")
    print(f"Plot: {png_path}")


def cli() -> None:
    p = argparse.ArgumentParser(description="Polymarket NYC temperature calibration backtest")
    p.add_argument("--start", type=_parse_date, required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", type=_parse_date, required=True, help="End date YYYY-MM-DD (inclusive)")
    p.add_argument("--lead-times", type=_parse_lead_times,
                   default=LEAD_TIMES_HOURS, help="Comma-separated, hours (e.g. 1,3,6,12,24)")
    p.add_argument("--model", choices=["nbm", "hrrr"], default=MODEL_DEFAULT)
    p.add_argument("--actuals-source", choices=["mesonet", "isd"],
                   default=ACTUALS_SOURCE_DEFAULT,
                   help="mesonet=Iowa State Mesonet (default); isd=NOAA ISD on S3 fallback")
    p.add_argument("--out-dir", type=Path, default=DERIVED_CACHE)
    args = p.parse_args()
    run(args.start, args.end, args.lead_times, args.model, args.out_dir,
        actuals_source=args.actuals_source)


if __name__ == "__main__":
    cli()
