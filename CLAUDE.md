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

## Active question (status: first result, hypothesis NOT confirmed)

**Does Polymarket overprice high-price YES contracts?** First end-to-end run on
**summer 2025** (Jun 1 – Sep 30, KLGA, 122 events, 854 binary markets, 4,114
price snapshots across 5 lead times). Frozen artifacts:
`results/market_calibration_polymarket_summer2025.{png,parquet}`.

Findings (lead times T = 1h / 3h, where signal is sharpest):
  - **High-price tail (p ≥ 0.85)**: `p_hat` lies AT or ABOVE `mean_price`, not
    below. At T=1h, n=33 markets at mean 0.984 → realized winrate 1.00. At
    T=1h, n=14 at mean 0.93 → 1.00. Favorites are not overpriced; if anything
    slightly underpriced (the NO side at p≈0.02 never won).
  - **Low-price tail (p ≤ 0.05)**: tiny overpricing of YES. At T=1h, n=549,
    mean 0.0045 → realized 0.0036. Real but the gap is well within Wilson CI.
  - **Mid range (0.2–0.8)** scatters around the diagonal; CIs wide due to
    bucket-market sparsity.

So the original "tails are overpriced" thesis is **not** supported in this
window. The pattern is asymmetric and well-known from prediction-market
literature: longshots slightly overpriced, favorites at or above fair. Both
effects are small at peak-heating lead times and noisy at longer leads. Worth
a longer window before concluding (only ~120 days here).

**Trading-strategy implication.** Buying NO on YES-favorite markets to "catch
reversals" — the obvious play if the tail-overpricing thesis held — is a money
loser in this sample:

  * T=1h, buy NO when YES ≥ 0.70 (n=93 trades): realized NO winrate 7.5%
    vs avg NO cost 10.1% → **ROI −26%**.
  * T=3h, same rule (n=41): ROI −26%.
  * Only the narrow band p_market ∈ [0.72, 0.78] shows a hint of +EV NO,
    but Wilson CI is wide (n ≤ 8 per bucket).

## NBM forecast accuracy at peak-heating lead times

A 31-day NBM backfill on July 2025 (`results/dataset_nbm_jul2025.parquet`)
shows why high-confidence Polymarket favorites resolve at 100%:

  * **T = 1h**: MAE = 0.27 °F, median |err| = 0 °F, 84% of days within ±0.5 °F.
    The forecast at 16:00 EDT (obs-spliced through 16Z) essentially knows
    the answer. Only 1/31 days had |err| > 1 °F.
  * **T = 3h**: MAE = 1.15 °F, **bias = −0.81 °F** (consistent under-
    forecast), 45% of days |err| > 1 °F, 19% > 2 °F. Worst miss 4.2 °F low.

The −0.81 °F bias at T=3h is the more actionable finding. On "≤ X°F or below"
markets where X is just above NBM's T=3h forecast, the actual high comes in
above X more often than the market prices imply. Pricing this out vs the
cached Polymarket data is the obvious next step.

## How to run

```bash
pip install -e .                    # cfgrib needs system libeccodes; see README
python -m pytest tests/ -v          # 8/8 should pass

# Fetch real data (requires the allowlisted hosts). The Polymarket
# "nyc-daily-weather" series started 2025-01-21 — earlier dates won't return
# anything.
polycal fetch-polymarket --start 2025-06-01 --end 2025-09-30
polycal market-calibration --start 2025-06-01 --end 2025-09-30

# Outputs to cache/derived/:
#   market_dataset_polymarket_<start>_<end>.parquet
#   market_calibration_polymarket_<start>_<end>.parquet
#   market_calibration_polymarket_<start>_<end>.png
```

The plot is the deliverable. Look at the `T = 1h` and `T = 3h` lines around
`price ≥ 0.85` — if `p_hat` lies clearly below `mean_price` with Wilson CI not
crossing the diagonal, the overpricing claim holds. (Current run: it doesn't.)

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
- Polymarket NYC daily-temp markets live under series `nyc-daily-weather`
  (gamma param `series_slug=nyc-daily-weather`). Tag-based discovery
  (`tag_slug=weather`) misses them entirely. Series started 2025-01-21.
- The 2025 series mixes three market shapes — "X°F or below", "X°F or above",
  "between X-Y°F". So `yes_won = (actual_high >= threshold)` is wrong for
  most rows. `extract_markets` parses the resolved outcome directly from
  `outcomePrices` (`["1","0"]` → YES won) and stores it as `yes_won_final`;
  `join_with_actuals` reads that column when `markets` is passed in.
- CLOB `prices-history` caps `(endTs − startTs) ≤ 360h (15 days)` at
  `fidelity=60`. Window in `fetch_polymarket.run` is `[resolve − 14d,
  resolve + 1d]` = exactly 15 days; do not enlarge or every request 400s.

## Branch convention

Active branch for the current handoff session: as specified in the session
prompt (typically `claude/continue-previous-work-*`). Never push to `master`
without explicit user instruction. Open a PR only if the user asks for one.
