# polycal — session briefing

Pick up here. This file is the handoff between sessions; read it first.

## What this is

A calibration backtest. The thesis: **Polymarket NYC daily-high temperature
markets overprice the tails** — when YES trades at 0.92 the realized winrate
is closer to 0.80. Single station: KLGA (LaGuardia). Two pipelines:

1. **Forecast calibration** — NBM/HRRR daily-high forecast at lead time T vs
   realized ASOS daily-high; bucket by `spread = forecast - threshold`, fit
   Wilson CI per bucket, compare to a bias-adjusted normal curve. Done.
2. **Market calibration** — Polymarket YES price snapshot at T hours before
   peak heating (17:00 local) vs realized outcome; bucket by `p_market`,
   plot vs `y = x` diagonal. Sagging below the diagonal at high price is the
   overpricing signal. Added in this branch.

## Repo layout, in short

```
src/polycal/
  config.py              station, lead times, cache paths, Polymarket URLs
  fetch_actuals.py       Iowa Mesonet ASOS (default) or NOAA ISD S3 (--actuals-source isd)
  fetch_forecasts.py     Herbie NBM/HRRR via AWS S3, pinned with priority=["aws"]
  fetch_polymarket.py    Gamma events + markets, CLOB hourly prices
  build_dataset.py       join forecasts × actuals, threshold expansion
  calibration.py         Wilson CI, spread binning, bias-adjusted theoretical curve
  market_calibration.py  price snapshots, join actuals, price binning
  plot.py / plot_market.py
  main.py                CLI (subcommands: backfill / fetch-polymarket / market-calibration)
tests/test_smoke.py      offline unit tests (synthetic data, 8 tests)
results/                 frozen plots and parquet from earlier backfills
cache/                   gitignored: actuals/, forecasts/, polymarket/, derived/
```

## Active question (status: open)

**Does Polymarket overprice high-price YES contracts?** The math is wired up,
but no real result yet — the previous session couldn't reach Polymarket/Mesonet
from its sandbox. Network allowlist has now been extended (see below). The next
session should be able to actually run the pipeline end to end.

## How to run

```bash
pip install -e .                    # cfgrib needs system libeccodes; see README
PYTHONPATH=src python -m pytest tests/ -v    # 8/8 should pass

# Fetch real data (requires the allowlisted hosts):
polycal fetch-polymarket --start 2024-06-01 --end 2024-10-31
polycal market-calibration --start 2024-06-01 --end 2024-10-31

# Outputs to cache/derived/:
#   market_dataset_polymarket_<start>_<end>.parquet
#   market_calibration_polymarket_<start>_<end>.parquet
#   market_calibration_polymarket_<start>_<end>.png
```

The plot is the deliverable. Look at the `T = 1h` and `T = 3h` lines around
`price ≥ 0.85` — if `p_hat` lies clearly below `mean_price` with Wilson CI not
crossing the diagonal, the overpricing claim holds.

## Network expectations

The sandbox proxy blocks everything not in `.claude/settings.json →
sandbox.network.allowedDomains`. Current list (do not narrow without reason):

- `gamma-api.polymarket.com`, `clob.polymarket.com`, `data-api.polymarket.com`,
  `docs.polymarket.com` — Polymarket APIs.
- `mesonet.agron.iastate.edu` — Iowa State Mesonet ASOS (default actuals).
- `noaa-global-hourly-pds.s3.amazonaws.com` — NOAA ISD S3 (fallback actuals,
  reach via `--actuals-source isd`).
- `noaa-nbm-grib2-pds.s3.amazonaws.com`, `noaa-hrrr-bdp-pds.s3.amazonaws.com`
  — Herbie pulls NBM/HRRR GRIB2 from these.

If a request returns HTTP 403 with header `x-deny-reason: host_not_allowed`,
it is the **sandbox proxy** rejecting, not the upstream API. The user has to
re-trust this project's `.claude/settings.json` for the current session/
container (a fresh cloud container does not inherit trust from earlier ones).

## Gotchas worth remembering

- `theoretical_curve` signature was changed from `mae_f=` to
  `sigma_f, bias_f=0.0` when the bias-adjusted curve landed. The smoke test
  was caught out by this and is now fixed; if another caller breaks, look
  here first.
- `snapshot_prices` compares `datetime64` timestamps via int64. Parquet often
  stores `ts_utc` as `datetime64[us]`; `pd.Timestamp.to_datetime64()` returns
  `ns`. Mixed-unit subtraction silently inflates the diff. The function now
  forces both sides to `ns` — preserve that if you refactor.
- Lead-time anchor is `PEAK_HOUR_LOCAL = 17` (NY local), not market close.
  This is intentional: it measures forecast skill at peak heating instead of
  collapsing into a post-peak nowcast.
- `fetch_forecasts.py` calls Herbie with `priority=["aws"]` — NOMADS had
  broken SSL in some sandboxes. Don't add NOMADS back unless you verify TLS.
- `actuals_source` default is `mesonet`; switch to `isd` only if Mesonet is
  blocked. ISD on S3 is more austere (quality codes, fewer obs) but always
  reachable from AWS-friendly networks.

## Branch convention

Active branch for the current handoff session: as specified in the session
prompt (typically `claude/continue-previous-work-*`). Never push to `master`
without explicit user instruction. Open a PR only if the user asks for one.
