"""Per-city resolution stations for Polymarket daily-weather series.

Source: each market's `resolutionSource` field on gamma-api.polymarket.com.
Coordinates are the official ICAO station locations (FAA/NOAA published).
"""
STATIONS = {
    "nyc":     {"icao": "KLGA", "lat": 40.7773, "lon": -73.8726, "tz": "America/New_York"},
    "chicago": {"icao": "KORD", "lat": 41.9786, "lon": -87.9048, "tz": "America/Chicago"},
    "miami":   {"icao": "KMIA", "lat": 25.7959, "lon": -80.2870, "tz": "America/New_York"},
    "dallas":  {"icao": "KDAL", "lat": 32.8471, "lon": -96.8518, "tz": "America/Chicago"},
    "atlanta": {"icao": "KATL", "lat": 33.6407, "lon": -84.4277, "tz": "America/New_York"},
    "seattle": {"icao": "KSEA", "lat": 47.4502, "lon": -122.3088,"tz": "America/Los_Angeles"},
    "denver":  {"icao": "KBKF", "lat": 39.7017, "lon": -104.7517,"tz": "America/Denver"},  # Buckley AFB / Aurora, NOT KDEN
    "houston": {"icao": "KHOU", "lat": 29.6454, "lon":  -95.2789,"tz": "America/Chicago"}, # Hobby, NOT KIAH
    "austin":  {"icao": "KAUS", "lat": 30.1945, "lon":  -97.6700,"tz": "America/Chicago"}, # Bergstrom
}
PEAK_HOUR_LOCAL = 17  # 5pm local — typical daily-high time

import pandas as pd
def peak_utc(city: str, date) -> pd.Timestamp:
    """tz-naive UTC ts of peak heating (17:00 local) for the city on the date."""
    tz = STATIONS[city]["tz"]
    p = pd.Timestamp.combine(date, pd.Timestamp("17:00").time()).tz_localize(tz)
    return p.tz_convert("UTC").tz_localize(None)
