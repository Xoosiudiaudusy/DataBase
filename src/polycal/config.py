"""Project-wide constants.

Single-city for now (NYC = KLGA = LaGuardia Airport, the station Polymarket
resolves NYC temperature markets against). To add more cities later, extend
STATIONS and pick the right NOAA model per region (NBM/HRRR for CONUS,
GFS/IFS for the rest of the world).
"""
from __future__ import annotations

from pathlib import Path

# --- Station / market ---
STATION_ID = "LGA"            # Iowa Mesonet id (no K-prefix)
STATION_ICAO = "KLGA"         # ICAO; some endpoints want this form
STATION_LAT = 40.7773
STATION_LON = -73.8726
LOCAL_TZ = "America/New_York"

# --- Forecast model defaults ---
MODEL_DEFAULT = "nbm"         # one of {"nbm", "hrrr"}
LEAD_TIMES_HOURS = [1, 3, 6, 12, 24]

# Daily-max forecast window, UTC. NY 07:00 EST → 18:00 EST (or 08-19 EDT).
DAILY_WINDOW_UTC_HOURS = list(range(12, 24))  # 12,13,...,23

# --- Threshold expansion (Polymarket-style binary thresholds) ---
THRESHOLD_OFFSETS_F = list(range(-5, 6))  # actual_high + offset, °F

# --- Calibration binning ---
SPREAD_BIN_EDGES_F = [round(-5.25 + 0.5 * i, 2) for i in range(22)]  # -5.25..+5.25, step 0.5

# --- Cache layout ---
REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "cache"
FORECAST_CACHE = CACHE_DIR / "forecasts"
ACTUALS_CACHE = CACHE_DIR / "actuals"
DERIVED_CACHE = CACHE_DIR / "derived"

for _d in (FORECAST_CACHE, ACTUALS_CACHE, DERIVED_CACHE):
    _d.mkdir(parents=True, exist_ok=True)


def kelvin_to_f(k: float) -> float:
    return (k - 273.15) * 9.0 / 5.0 + 32.0
