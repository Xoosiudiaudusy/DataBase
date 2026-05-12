# polycal — Polymarket NYC temperature-market calibration backtest

Investigates the hypothesis that Polymarket temperature markets systematically
overprice the tails: that when the price of YES is 0.92, the realized winrate
of YES is closer to 0.80. This first iteration builds the **forecast** calibration
curve — i.e. given a forecast daily high and a threshold, how often does the
realized daily high clear that threshold? — without yet hitting Polymarket
price data.

Station: **KLGA** (LaGuardia Airport), the Iowa-Mesonet `LGA` ASOS site that
Polymarket resolves NYC temperature markets against.

## Install

`cfgrib` needs the system **eccodes** library:

```bash
# Debian/Ubuntu
sudo apt-get install -y libeccodes0
# or via conda
conda install -c conda-forge eccodes
```

Then:

```bash
pip install -e .
```

## Run

Smoke test (one week, two lead times — should finish in a few minutes):

```bash
polycal --start 2024-06-01 --end 2024-06-07 --lead-times 1,24
```

Six-month backfill (all five lead times):

```bash
polycal --start 2024-11-01 --end 2025-05-01
```

Output goes to `cache/derived/`:

- `dataset_<model>_<start>_<end>.parquet` — joined per-row dataset.
- `calibration_<model>_<start>_<end>.parquet` — bucketed calibration table.
- `calibration_<model>_<start>_<end>.png` — empirical vs theoretical-normal curve, one line per lead time.

## Module layout

```
src/polycal/
  config.py            station / coords / lead times / cache layout
  fetch_actuals.py     Iowa Mesonet ASOS (cached per year)
  fetch_forecasts.py   NBM (default) or HRRR via Herbie, per-day parquet cache
  build_dataset.py     join forecasts × actuals, expand thresholds
  calibration.py       Wilson CI, spread bucketing, theoretical normal curve
  plot.py              matplotlib figure
  main.py              CLI entry point
```

## Lead-time semantics

For day D, lead time T:

```
issue_utc = floor_hour( America/New_York(D, 23:59) − T hours )
fxx_list  = { fxx : issue_utc + fxx ∈ [12Z(D), 23Z(D)] }
forecast_high = max( t2m_f at fxx_list )
```

For T ∈ {1, 3, 6} the issue time falls inside or after the daily window, so
fxx covers only the tail (or nothing). That is intentional — the resulting
"nowcast collapse" of the spread is part of the signal we're after.

## Hypothesis check

After running the 180-day backfill, look at the `T = 3h` line near
`spread = +3°F`. If the empirical P(YES) is around 0.80 while the dashed
theoretical-normal curve (with `σ = MAE × √(π/2)`) sits near 0.95, the
forecast has fatter tails than a Gaussian — and a 92¢ Polymarket YES contract
in that situation really is overpriced.

## Out of scope (this iteration)

- Polymarket price data — next step, after the calibration curve is solid.
- ML / regression — straight bucket aggregation only.
- Other cities (London EGLC, Miami KMIA, LA KLAX). NBM/HRRR are CONUS only;
  London needs GFS or ECMWF IFS.
