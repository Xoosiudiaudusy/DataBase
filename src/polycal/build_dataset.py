"""Join forecasts × actuals and expand thresholds.

For each (date, lead_time) we pair the forecast daily high with the realized
daily high, then emit one row per Polymarket-style threshold around the
realized value. ``spread = forecast_high - threshold``, ``yes_won = 1`` if
the realized high cleared the threshold.
"""
from __future__ import annotations

import datetime as dt
from typing import Iterable

import pandas as pd
from tqdm import tqdm

from .config import ACTUALS_SOURCE_DEFAULT, THRESHOLD_OFFSETS_F
from .fetch_actuals import daily_highs
from .fetch_forecasts import forecast_high_for_day


def _iter_dates(start: dt.date, end: dt.date) -> Iterable[dt.date]:
    d = start
    one = dt.timedelta(days=1)
    while d <= end:
        yield d
        d += one


def build_rows(
    start: dt.date, end: dt.date,
    lead_times: list[int], model: str = "nbm",
    actuals_source: str = ACTUALS_SOURCE_DEFAULT,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Build the day × lead × threshold dataset for the requested window."""
    actuals = (
        daily_highs(start, end, source=actuals_source)
        .set_index("local_date")["actual_high_f"].to_dict()
    )

    pairs: list[dict] = []
    dates = list(_iter_dates(start, end))
    iterator = tqdm(dates, desc=f"{model} forecasts", disable=not show_progress)
    for d in iterator:
        if d not in actuals:
            continue
        actual = actuals[d]
        for T in lead_times:
            row = forecast_high_for_day(d, T, model=model)
            if row is None:
                continue
            for off in THRESHOLD_OFFSETS_F:
                threshold = actual + off
                spread = row["forecast_high_f"] - threshold
                pairs.append({
                    "date": d,
                    "lead_time": T,
                    "threshold": threshold,
                    "spread": spread,
                    "yes_won": int(actual >= threshold),
                    "forecast_high": row["forecast_high_f"],
                    "actual_high": actual,
                    "n_steps": row["n_steps"],
                })
    return pd.DataFrame(pairs)
