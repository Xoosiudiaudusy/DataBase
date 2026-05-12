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
    sub = p.add_subparsers(dest="cmd")

    # Default subcommand: backfill (forecast vs actual calibration).
    bp = sub.add_parser("backfill", help="Build forecast/actuals dataset + calibration plot")
    bp.add_argument("--start", type=_parse_date, required=True)
    bp.add_argument("--end", type=_parse_date, required=True)
    bp.add_argument("--lead-times", type=_parse_lead_times, default=LEAD_TIMES_HOURS)
    bp.add_argument("--model", choices=["nbm", "hrrr"], default=MODEL_DEFAULT)
    bp.add_argument("--actuals-source", choices=["mesonet", "isd"], default=ACTUALS_SOURCE_DEFAULT)
    bp.add_argument("--out-dir", type=Path, default=DERIVED_CACHE)

    # Polymarket fetcher: discover NYC temp events, save metadata + hourly prices.
    fp = sub.add_parser("fetch-polymarket",
                        help="Discover NYC temperature markets and fetch hourly price history")
    fp.add_argument("--start", type=_parse_date, required=True)
    fp.add_argument("--end", type=_parse_date, required=True)
    fp.add_argument("--sleep", type=float, default=0.1,
                    help="Pause between CLOB requests, seconds")

    # Market calibration: snapshot Polymarket prices at lead times, join actuals,
    # bin by price → empirical winrate vs market YES price.
    mp = sub.add_parser("market-calibration",
                        help="Bucket cached Polymarket prices by lead time and "
                             "compute empirical winrate vs market YES price")
    mp.add_argument("--start", type=_parse_date, required=True)
    mp.add_argument("--end", type=_parse_date, required=True)
    mp.add_argument("--lead-times", type=_parse_lead_times, default=LEAD_TIMES_HOURS)
    mp.add_argument("--actuals-source", choices=["mesonet", "isd"],
                    default=ACTUALS_SOURCE_DEFAULT)
    mp.add_argument("--tolerance-hours", type=float, default=2.0,
                    help="Max gap between target lead-time and nearest cached price")
    mp.add_argument("--out-dir", type=Path, default=DERIVED_CACHE)

    # Back-compat: bare `polycal --start ... --end ...` runs the backfill.
    p.add_argument("--start", type=_parse_date, help=argparse.SUPPRESS)
    p.add_argument("--end", type=_parse_date, help=argparse.SUPPRESS)
    p.add_argument("--lead-times", type=_parse_lead_times,
                   default=LEAD_TIMES_HOURS, help=argparse.SUPPRESS)
    p.add_argument("--model", choices=["nbm", "hrrr"], default=MODEL_DEFAULT, help=argparse.SUPPRESS)
    p.add_argument("--actuals-source", choices=["mesonet", "isd"],
                   default=ACTUALS_SOURCE_DEFAULT, help=argparse.SUPPRESS)
    p.add_argument("--out-dir", type=Path, default=DERIVED_CACHE, help=argparse.SUPPRESS)

    args = p.parse_args()

    if args.cmd == "fetch-polymarket":
        from .fetch_polymarket import run as run_poly
        run_poly(args.start, args.end, sleep_s=args.sleep)
        return

    if args.cmd == "market-calibration":
        from .market_calibration import run as run_market
        from .plot_market import plot_market_calibration
        joined, cal = run_market(
            args.start, args.end,
            lead_times=args.lead_times,
            actuals_source=args.actuals_source,
            tolerance_hours=args.tolerance_hours,
        )
        args.out_dir.mkdir(parents=True, exist_ok=True)
        tag = f"polymarket_{args.start.isoformat()}_{args.end.isoformat()}"
        joined_path = args.out_dir / f"market_dataset_{tag}.parquet"
        cal_path = args.out_dir / f"market_calibration_{tag}.parquet"
        png_path = args.out_dir / f"market_calibration_{tag}.png"
        joined.to_parquet(joined_path, index=False)
        cal.to_parquet(cal_path, index=False)
        plot_market_calibration(
            cal, joined, png_path,
            title=f"Polymarket NYC temp — empirical vs price — {args.start}…{args.end}",
        )
        print(f"Snapshots: {len(joined)} rows → {joined_path}")
        print(f"Calibration: {len(cal)} buckets → {cal_path}")
        print(f"Plot: {png_path}")
        return

    # Default / backfill
    if args.start is None or args.end is None:
        p.error("--start and --end are required for backfill")
    run(args.start, args.end, args.lead_times, args.model, args.out_dir,
        actuals_source=args.actuals_source)


if __name__ == "__main__":
    cli()
