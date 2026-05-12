"""Fetch Polymarket NYC daily-high temperature markets and their price history.

Two APIs:
  * Gamma (gamma-api.polymarket.com): event/market metadata.
  * CLOB  (clob.polymarket.com):       price-history time series per token.

The Polymarket "highest temperature in NYC on <date>" series resolves against
LaGuardia's NWS daily-max. Each event is a single calendar date and contains
multiple binary markets (one per threshold, e.g. "≥ 85°F"). We harvest:
  - event slug, end_date, conditionId
  - market question (we parse the threshold from it), YES token id
  - price history at 60-min fidelity for the YES token

Idempotent: re-running only fetches markets/prices that aren't already cached.
"""
from __future__ import annotations

import datetime as dt
import re
import time
from typing import Iterable

import pandas as pd
import requests
from tqdm import tqdm

from .config import CLOB_API, GAMMA_API, LOCAL_TZ, POLYMARKET_CACHE


# ---------- low-level API helpers ----------

_session = requests.Session()
_session.headers.update({"User-Agent": "polycal-research/0.1"})


def _gamma_get(path: str, params: dict | None = None) -> list | dict:
    r = _session.get(f"{GAMMA_API}{path}", params=params or {}, timeout=30)
    r.raise_for_status()
    return r.json()


def _clob_get(path: str, params: dict | None = None) -> dict:
    r = _session.get(f"{CLOB_API}{path}", params=params or {}, timeout=30)
    r.raise_for_status()
    return r.json()


def _paginate_events(params: dict, page_size: int = 100) -> Iterable[dict]:
    offset = 0
    while True:
        batch = _gamma_get("/events", {**params, "limit": page_size, "offset": offset})
        if not isinstance(batch, list) or not batch:
            return
        yield from batch
        if len(batch) < page_size:
            return
        offset += page_size


# ---------- discovery: NYC daily-high temperature events ----------

# Polymarket event slugs we've seen: "highest-temperature-in-nyc-on-<month>-<day>",
# also "what-will-be-the-highest-temperature-in-nyc-on-<date>". Be permissive.
_NYC_TEMP_PATTERNS = [
    re.compile(r"highest[- ]temperature[- ]in[- ]nyc", re.I),
    re.compile(r"highest[- ]temperature[- ]in[- ]new[- ]york", re.I),
    re.compile(r"nyc.*high.*temp", re.I),
    re.compile(r"new[- ]york.*high.*temp", re.I),
]

_THRESHOLD_PATTERNS = [
    re.compile(r"(?:≥|>=|at\s+or\s+above|or\s+higher|reach(?:es)?\s+(?:above|or\s+above)?)\s*(\d{2,3})\s*°?\s*F?", re.I),
    re.compile(r"(?:above|over|exceed(?:s)?)\s*(\d{2,3})\s*°?\s*F?", re.I),
    re.compile(r"(\d{2,3})\s*°?\s*F\s*(?:or\s+higher|or\s+above|or\s+more)", re.I),
    re.compile(r"\b(\d{2,3})\s*°F\b"),  # last resort: any temperature reference
]


def _looks_like_nyc_temp_event(ev: dict) -> bool:
    slug = (ev.get("slug") or "").lower()
    title = (ev.get("title") or ev.get("question") or "").lower()
    for pat in _NYC_TEMP_PATTERNS:
        if pat.search(slug) or pat.search(title):
            return True
    return False


def parse_threshold(question: str) -> int | None:
    for pat in _THRESHOLD_PATTERNS:
        m = pat.search(question or "")
        if m:
            try:
                return int(m.group(1))
            except (ValueError, IndexError):
                continue
    return None


def discover_events(start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    """Return DataFrame of NYC daily-high temperature events resolving in [start, end]."""
    # Use end_date filters (event resolves on end_date)
    params = {
        "closed": "true",
        "archived": "false",
        "tag_slug": "weather",
        "end_date_min": f"{start_date.isoformat()}T00:00:00Z",
        "end_date_max": f"{(end_date + dt.timedelta(days=1)).isoformat()}T00:00:00Z",
        "order": "endDate",
        "ascending": "true",
    }
    rows = []
    for ev in _paginate_events(params):
        if not _looks_like_nyc_temp_event(ev):
            continue
        rows.append({
            "event_id": ev.get("id"),
            "slug": ev.get("slug"),
            "title": ev.get("title") or ev.get("question"),
            "end_date": pd.to_datetime(ev.get("endDate")).tz_convert(LOCAL_TZ).date()
                if ev.get("endDate") else None,
            "start_date": pd.to_datetime(ev.get("startDate")).tz_convert(LOCAL_TZ).date()
                if ev.get("startDate") else None,
            "n_markets": len(ev.get("markets", [])),
            "raw": ev,
        })
    return pd.DataFrame(rows)


# ---------- markets & price history ----------

def extract_markets(events_df: pd.DataFrame) -> pd.DataFrame:
    """Flatten events into one row per binary market with threshold parsed."""
    import json
    out = []
    for _, ev in events_df.iterrows():
        for m in (ev["raw"].get("markets") or []):
            q = m.get("question") or ""
            thr = parse_threshold(q)
            token_ids = m.get("clobTokenIds")
            if isinstance(token_ids, str):
                try:
                    token_ids = json.loads(token_ids)
                except Exception:
                    token_ids = None
            outcomes = m.get("outcomes")
            if isinstance(outcomes, str):
                try:
                    outcomes = json.loads(outcomes)
                except Exception:
                    outcomes = None
            yes_token = None
            if token_ids and outcomes and len(token_ids) == len(outcomes):
                for o, tid in zip(outcomes, token_ids):
                    if str(o).strip().lower() in ("yes", "true", "1"):
                        yes_token = tid
                        break
            if yes_token is None and token_ids:
                yes_token = token_ids[0]  # convention: index 0 is YES
            out.append({
                "event_id": ev["event_id"],
                "event_slug": ev["slug"],
                "resolve_date": ev["end_date"],
                "market_id": m.get("id"),
                "condition_id": m.get("conditionId"),
                "question": q,
                "threshold_f": thr,
                "yes_token_id": str(yes_token) if yes_token else None,
                "resolved_outcome": m.get("umaResolutionStatus") or m.get("resolvedBy") or None,
                "outcome_prices_final": m.get("outcomePrices"),
            })
    df = pd.DataFrame(out)
    return df.dropna(subset=["threshold_f", "yes_token_id"]).reset_index(drop=True)


def fetch_price_history(
    token_id: str, start_ts: int, end_ts: int, fidelity_min: int = 60,
) -> pd.DataFrame:
    """Return hourly price series for a YES token over [start_ts, end_ts] UNIX."""
    data = _clob_get(
        "/prices-history",
        {"market": token_id, "startTs": start_ts, "endTs": end_ts, "fidelity": fidelity_min},
    )
    hist = data.get("history") or []
    if not hist:
        return pd.DataFrame(columns=["ts_utc", "p"])
    df = pd.DataFrame(hist)
    df["ts_utc"] = pd.to_datetime(df["t"], unit="s", utc=True).dt.tz_localize(None)
    df["p"] = df["p"].astype(float)
    return df[["ts_utc", "p"]].sort_values("ts_utc").reset_index(drop=True)


# ---------- orchestration ----------

def run(start_date: dt.date, end_date: dt.date, sleep_s: float = 0.1) -> None:
    """Discover events in window, persist metadata + price histories."""
    POLYMARKET_CACHE.mkdir(parents=True, exist_ok=True)
    events_path = POLYMARKET_CACHE / "events.parquet"
    markets_path = POLYMARKET_CACHE / "markets.parquet"
    prices_path = POLYMARKET_CACHE / "prices.parquet"

    print(f"Discovering NYC temp events in [{start_date}, {end_date}]...")
    ev_df = discover_events(start_date, end_date)
    print(f"  found {len(ev_df)} events")
    if ev_df.empty:
        print("Nothing to do.")
        return
    ev_df.drop(columns=["raw"]).to_parquet(events_path)

    mkt_df = extract_markets(ev_df)
    print(f"  parsed {len(mkt_df)} binary markets with threshold + YES token")
    mkt_df.to_parquet(markets_path)

    # Fetch price history per market. The window: from event open (start_date)
    # to event end (resolve_date end-of-day UTC). We pull at hourly fidelity.
    all_prices = []
    for _, row in tqdm(list(mkt_df.iterrows()), desc="price history"):
        resolve = row["resolve_date"]
        if resolve is None:
            continue
        start_ts = int(pd.Timestamp(resolve - dt.timedelta(days=14)).timestamp())
        end_ts = int((pd.Timestamp(resolve) + pd.Timedelta(days=2)).timestamp())
        try:
            ph = fetch_price_history(row["yes_token_id"], start_ts, end_ts, fidelity_min=60)
        except requests.HTTPError as e:
            print(f"  {row['market_id']}: HTTP {e.response.status_code}, skipping")
            continue
        if ph.empty:
            continue
        ph["market_id"] = row["market_id"]
        ph["resolve_date"] = resolve
        ph["threshold_f"] = row["threshold_f"]
        all_prices.append(ph)
        time.sleep(sleep_s)
    if all_prices:
        prices = pd.concat(all_prices, ignore_index=True)
        prices.to_parquet(prices_path)
        print(f"Saved {len(prices):,} price rows -> {prices_path}")
    else:
        print("No price rows collected.")
