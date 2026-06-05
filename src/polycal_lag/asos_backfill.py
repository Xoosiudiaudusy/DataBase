"""Pull ASOS observations for all 9 stations × all years in window."""
from __future__ import annotations
import datetime as dt, io, time
from pathlib import Path
import pandas as pd, requests
from .stations import STATIONS

OUT_DIR = Path("cache/actuals")
URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"


def pull_station_year(icao_3: str, year: int, max_retries: int = 5) -> pd.DataFrame:
    params = {
        "station": icao_3, "data": "tmpf",
        "year1": year, "month1": 1, "day1": 1,
        "year2": year, "month2": 12, "day2": 31,
        "tz": "Etc/UTC", "format": "onlycomma", "missing": "empty",
    }
    delay = 2
    for _ in range(max_retries):
        try:
            r = requests.get(URL, params=params, timeout=120)
            if r.status_code == 429:
                time.sleep(delay); delay = min(delay * 2, 60); continue
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            if df.empty: return df
            df["valid_utc"] = pd.to_datetime(df["valid"])
            return df[["valid_utc", "tmpf"]].dropna().sort_values("valid_utc").reset_index(drop=True)
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                time.sleep(delay); delay = min(delay * 2, 60); continue
            raise
    raise RuntimeError("mesonet 429 after retries")


def run(years: list[int], inter_request_delay_s: float = 2.0):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for city, info in STATIONS.items():
        icao_3 = info["icao"][1:]
        for y in years:
            fp = OUT_DIR / f"mesonet_{icao_3.lower()}_{y}.parquet"
            if fp.exists():
                print(f"  {city:>8} ({icao_3}) {y}: cached"); continue
            print(f"  {city:>8} ({icao_3}) {y}: pulling ...", end=" ", flush=True)
            try:
                df = pull_station_year(icao_3, y)
                df.to_parquet(fp, index=False)
                print(f"{len(df):,} rows")
            except Exception as e:
                print(f"ERR {e}")
            time.sleep(inter_request_delay_s)


if __name__ == "__main__":
    import sys
    years = [int(y) for y in sys.argv[1:]] if len(sys.argv) > 1 else [2025, 2026]
    run(years)
