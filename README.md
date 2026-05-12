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
  config.py             station / coords / lead times / cache layout
  fetch_actuals.py      Iowa Mesonet ASOS or NOAA ISD S3 (cached per year)
  fetch_forecasts.py    NBM (default) or HRRR via Herbie, per-day parquet cache
  fetch_polymarket.py   Polymarket Gamma + CLOB, event/market metadata + prices
  build_dataset.py      join forecasts × actuals, expand thresholds
  calibration.py        Wilson CI, spread bucketing, theoretical normal curve
  market_calibration.py snapshot prices at lead times, join actuals, bin by price
  plot.py               forecast calibration figure
  plot_market.py        market calibration figure (winrate vs YES price)
  main.py               CLI entry point
```

## Market calibration

The forecast calibration above is the prior. The actual hypothesis under test —
"Polymarket overprices the tails" — needs the market itself. Two-step:

```bash
# 1. Fetch cached prices+markets for the window (writes cache/polymarket/).
polycal fetch-polymarket --start 2024-06-01 --end 2024-10-01

# 2. Snapshot YES price at each lead time, join with actuals, bin by price.
polycal market-calibration --start 2024-06-01 --end 2024-10-01
```

Output goes to `cache/derived/`:

- `market_dataset_polymarket_<start>_<end>.parquet` — per-(market, lead_time)
  snapshot rows: market_id, threshold_f, lead_time, p_market, actual_high_f,
  yes_won.
- `market_calibration_polymarket_<start>_<end>.parquet` — bucketed table:
  bin_center, n, k, p_hat, Wilson lo/hi, mean_price.
- `market_calibration_polymarket_<start>_<end>.png` — empirical P(YES) vs
  market YES price, one line per lead time, with the y = x diagonal overlaid.

A perfectly calibrated market sits on the diagonal. If the curve at `T = 1h`
sags below the diagonal at price ≥ 0.85 — say empirical winrate is 0.78 when
the market is at 0.92 — that's the overpricing signal we were after.

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

- ML / regression — straight bucket aggregation only.
- Trading P&L sizing / Kelly. Calibration is a precondition, not a strategy.
- Other cities (London EGLC, Miami KMIA, LA KLAX). NBM/HRRR are CONUS only;
  London needs GFS or ECMWF IFS.
