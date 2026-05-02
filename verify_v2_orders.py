"""Sanity-check that we can sign + post limit orders against Polymarket CLOB V2
via our proxy (Gnosis Safe) wallet.

Buys then immediately sells a tiny position on a London weather market.
Hard budget cap: $1.50.
"""

import json
import os
import sys
import time
from decimal import Decimal, ROUND_DOWN

import requests
from dotenv import load_dotenv
from py_clob_client_v2 import (
    ApiCreds,
    ClobClient,
    OrderArgs,
    OrderType,
    PartialCreateOrderOptions,
    Side,
    SignatureTypeV2,
)

load_dotenv()

HOST = "https://clob.polymarket.com"
GAMMA = "https://gamma-api.polymarket.com"
CHAIN_ID = 137

MAX_BUDGET_USD = Decimal("1.50")
TARGET_SPEND_USD = Decimal("1.20")
MAX_PRICE_CAP = Decimal("0.20")
FILL_POLL_SEC = 2
FILL_TIMEOUT_S = 60

PRIVATE_KEY = os.environ["PRIVATE_KEY"]
FUNDER_ADDRESS = os.environ["FUNDER_ADDRESS"]
# Default to Gnosis Safe (legacy browser-wallet proxy created by Polymarket UI).
# Override with SIGNATURE_TYPE=POLY_1271 if you have a new V2-style deposit wallet.
SIGNATURE_TYPE_NAME = os.environ.get("SIGNATURE_TYPE", "POLY_GNOSIS_SAFE").upper()
SIGNATURE_TYPE = getattr(SignatureTypeV2, SIGNATURE_TYPE_NAME)

WEATHER_KEYWORDS = ("weather", "temperature", "temp ", "rain", "snow", "hottest", "coldest", "wettest", "degrees", "celsius", "fahrenheit", "°")
LOCATION_KEYWORDS = ("london",)
BIG_CITIES = ("london", "new york", "nyc", "los angeles", "chicago", "tokyo", "paris", "moscow", "berlin", "madrid", "miami", "boston")


def make_client(creds=None):
    return ClobClient(
        host=HOST,
        chain_id=CHAIN_ID,
        key=PRIVATE_KEY,
        signature_type=SIGNATURE_TYPE,
        funder=FUNDER_ADDRESS,
        creds=creds,
    )


def _silence_sdk_http_logs():
    """SDK uses logger.error() to dump raw HTTP error bodies (incl. Cloudflare HTML
    pages). Install a filter that collapses HTML bodies to a one-line summary."""
    import logging

    class _StripHtml(logging.Filter):
        def filter(self, record):
            try:
                msg = record.getMessage()
            except Exception:
                return True
            if "<!DOCTYPE" in msg or "<html" in msg or "Cloudflare" in msg:
                head = msg.split("body=", 1)[0].rstrip()
                record.msg = f"{head} body=<{len(msg)} bytes html, suppressed>"
                record.args = ()
            return True

    f = _StripHtml()
    for name in (
        "py_clob_client_v2",
        "py_clob_client_v2.http_helpers",
        "py_clob_client_v2.http_helpers.helpers",
    ):
        logging.getLogger(name).addFilter(f)
    # ensure root sees the filter too in case loggers propagate
    logging.getLogger().addFilter(f)
    # Make sure the logger has at least one handler so output is visible
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def get_creds():
    if all(os.environ.get(k) for k in ("CLOB_API_KEY", "CLOB_SECRET", "CLOB_PASS_PHRASE")):
        return ApiCreds(
            api_key=os.environ["CLOB_API_KEY"],
            api_secret=os.environ["CLOB_SECRET"],
            api_passphrase=os.environ["CLOB_PASS_PHRASE"],
        )
    print("[creds] deriving L2 credentials from PRIVATE_KEY...")
    _silence_sdk_http_logs()
    return make_client().create_or_derive_api_key()


def _parse_json_field(val):
    if isinstance(val, list):
        return val
    if isinstance(val, str) and val:
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            return []
    return []


def _to_market_record(m):
    token_ids = _parse_json_field(m.get("clobTokenIds"))
    outcomes = _parse_json_field(m.get("outcomes"))
    prices = _parse_json_field(m.get("outcomePrices"))
    if len(token_ids) != 2 or len(prices) != 2:
        return None
    return {
        "question": m.get("question"),
        "slug": m.get("slug"),
        "tick_size": str(m.get("orderPriceMinTickSize") or m.get("minimumTickSize") or "0.01"),
        "min_size": Decimal(str(m.get("orderMinSize") or m.get("minimumOrderSize") or "5")),
        "tokens": [
            {"id": str(token_ids[0]), "outcome": outcomes[0] if outcomes else "0", "approx_price": Decimal(prices[0])},
            {"id": str(token_ids[1]), "outcome": outcomes[1] if len(outcomes) > 1 else "1", "approx_price": Decimal(prices[1])},
        ],
    }


def _flatten_event(ev):
    out = []
    for m in ev.get("markets", []) or []:
        m["_event_slug"] = ev.get("slug")
        m["_event_title"] = ev.get("title")
        out.append(m)
    return out


def _fetch_event_by_slug(slug):
    try:
        r = requests.get(f"{GAMMA}/events", params={"slug": slug}, timeout=15)
        r.raise_for_status()
        batch = r.json()
        if isinstance(batch, dict):
            batch = [batch]
        out = []
        for ev in batch:
            out.extend(_flatten_event(ev))
        if out:
            print(f"[gamma /events?slug={slug}] {len(out)} child markets")
        return out
    except Exception as e:
        print(f"[gamma /events?slug={slug}] error: {e}")
        return []


def _fetch_events_markets():
    """Pull markets via /events (which exposes child markets for multi-outcome events)
    filtered to ones whose event slug looks weather-related. Returns flat market dicts."""
    # Direct lookup of any explicitly requested event
    out = []
    explicit = os.environ.get("EVENT_SLUG")
    if explicit:
        out.extend(_fetch_event_by_slug(explicit))
    # Try a couple of common London weather event slug patterns (today/tomorrow)
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    for d in (now, now + timedelta(days=1), now - timedelta(days=1)):
        slug = f"highest-temperature-in-london-on-{d.strftime('%B').lower()}-{d.day}-{d.year}"
        out.extend(_fetch_event_by_slug(slug))

    seen_event_keys = set()
    london_events_seen = []
    for offset in (0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000):
        try:
            r = requests.get(
                f"{GAMMA}/events",
                params={"closed": "false", "archived": "false", "limit": 500, "offset": offset},
                timeout=30,
            )
            r.raise_for_status()
        except Exception as e:
            print(f"[gamma /events] offset={offset} error: {e}")
            break
        batch = r.json()
        if not batch:
            break
        for ev in batch:
            ev_slug = ev.get("slug") or ""
            ev_text = (ev_slug + " " + (ev.get("title") or "")).lower()
            if "london" in ev_text:
                london_events_seen.append(ev_slug)
            if not (any(c in ev_text for c in BIG_CITIES) or
                    any(w in ev_text for w in WEATHER_KEYWORDS)):
                continue
            key = ev.get("id") or ev_slug
            if key in seen_event_keys:
                continue
            seen_event_keys.add(key)
            out.extend(_flatten_event(ev))
    print(f"[gamma /events] {len(out)} child markets from weather/city events")
    if london_events_seen:
        print(f"[gamma /events] london-mentioning event slugs seen: {london_events_seen[:10]}")
    return out


def _fetch_market_by_slug(slug):
    try:
        r = requests.get(f"{GAMMA}/markets", params={"slug": slug}, timeout=15)
        r.raise_for_status()
        batch = r.json()
        if isinstance(batch, dict):
            batch = [batch]
        return batch
    except Exception as e:
        print(f"[gamma /markets?slug={slug}] error: {e}")
        return []


def find_candidate_markets():
    explicit_market = os.environ.get("MARKET_SLUG")
    if explicit_market:
        print(f"[gamma] explicit MARKET_SLUG={explicit_market}")
        ms = _fetch_market_by_slug(explicit_market)
        recs = []
        for m in ms:
            rec = _to_market_record(m)
            if rec:
                recs.append(rec)
                print(f"  -> {rec['question']!r}  prices={rec['tokens'][0]['approx_price']}/{rec['tokens'][1]['approx_price']}  tick={rec['tick_size']}")
        if recs:
            return recs
        print("[gamma] explicit slug returned nothing — falling back to search")

    print("[gamma] fetching active markets...")
    out = []
    for offset in (0, 500, 1000, 1500, 2000):
        r = requests.get(
            f"{GAMMA}/markets",
            params={
                "active": "true",
                "closed": "false",
                "archived": "false",
                "limit": 500,
                "offset": offset,
                "order": "volumeNum",
                "ascending": "false",
            },
            timeout=30,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        out.extend(batch)
    print(f"[gamma /markets] got {len(out)} markets")
    out.extend(_fetch_events_markets())
    # dedup by id
    seen = set()
    deduped = []
    for m in out:
        mid = m.get("id") or m.get("conditionId") or m.get("slug")
        if mid in seen:
            continue
        seen.add(mid)
        deduped.append(m)
    out = deduped
    print(f"[gamma] {len(out)} unique markets total")

    accepting = [m for m in out if m.get("acceptingOrders")]
    print(f"[gamma] {len(accepting)} acceptingOrders")

    def passes(m, loc_kw, weather_kw):
        text = " ".join([
            m.get("question", "") or "",
            m.get("slug", "") or "",
            m.get("_event_slug", "") or "",
        ]).lower()
        if loc_kw and not any(k in text for k in loc_kw):
            return False
        if weather_kw and not any(k in text for k in weather_kw):
            return False
        return True

    tiers = [
        ("london+weather", LOCATION_KEYWORDS, WEATHER_KEYWORDS),
        ("any city + weather", BIG_CITIES, WEATHER_KEYWORDS),
        ("any weather", (), WEATHER_KEYWORDS),
        ("london only", LOCATION_KEYWORDS, ()),
        ("any binary market", (), ()),
    ]

    for label, loc_kw, weather_kw in tiers:
        cands = []
        for m in accepting:
            if not passes(m, loc_kw, weather_kw):
                continue
            rec = _to_market_record(m)
            if rec is None:
                continue
            # at least one side priced in our budget zone
            if not any(t["approx_price"] <= MAX_PRICE_CAP for t in rec["tokens"]):
                continue
            cands.append(rec)
        print(f"[gamma] tier '{label}': {len(cands)} candidates")
        if cands:
            print("[gamma] sample of candidates:")
            for c in cands[:5]:
                p0, p1 = c["tokens"][0]["approx_price"], c["tokens"][1]["approx_price"]
                print(f"  - {c['question']!r}  prices={p0}/{p1}  tick={c['tick_size']}")
            return cands
    return []


def _ob_get(ob, key):
    if isinstance(ob, dict):
        return ob.get(key) or []
    return getattr(ob, key, None) or []


def _entry(e, key):
    if isinstance(e, dict):
        return e.get(key)
    return getattr(e, key, None)


def pick_token(client, candidates):
    """For each candidate, query the orderbook and pick the cheapest actionable side."""
    best = None
    for m in candidates:
        for tok in m["tokens"]:
            print(f"\n[pick] {m['slug']}/{tok['outcome']} approx_price={tok['approx_price']}  token_id={tok['id']}")
            if tok["approx_price"] > MAX_PRICE_CAP:
                print(f"  skip: approx_price {tok['approx_price']} > MAX_PRICE_CAP {MAX_PRICE_CAP}")
                continue
            try:
                ob = client.get_order_book(tok["id"])
            except Exception as e:
                print(f"  skip: get_order_book failed: {e!r}")
                continue
            keys = list(ob.keys()) if isinstance(ob, dict) else [a for a in dir(ob) if not a.startswith('_')][:12]
            print(f"  ob type={type(ob).__name__} keys/attrs={keys}")
            if isinstance(ob, dict) and not ob:
                print("  /book returned empty dict — probing /book directly via raw HTTP")
                try:
                    raw = requests.get("https://clob.polymarket.com/book",
                                       params={"token_id": tok["id"]}, timeout=15)
                    print(f"  raw /book status={raw.status_code} len={len(raw.text)} body[:300]={raw.text[:300]!r}")
                except Exception as e:
                    print(f"  raw /book error: {e!r}")
            asks = _ob_get(ob, "asks")
            bids = _ob_get(ob, "bids")
            print(f"  ob: asks={len(asks)} bids={len(bids)}")
            if asks:
                print(f"  first asks: {[(str(_entry(a,'price')), str(_entry(a,'size'))) for a in asks[:3]]}")
            if bids:
                print(f"  first bids: {[(str(_entry(b,'price')), str(_entry(b,'size'))) for b in bids[:3]]}")
            if not asks or not bids:
                print(f"  skip: empty side(s)")
                continue
            ask_prices = sorted(Decimal(str(_entry(a, "price"))) for a in asks)
            bid_prices = sorted((Decimal(str(_entry(b, "price"))) for b in bids), reverse=True)
            best_ask = ask_prices[0]
            best_bid = bid_prices[0]
            ask_size_at_top = sum(Decimal(str(_entry(a, "size"))) for a in asks if Decimal(str(_entry(a, "price"))) == best_ask)
            bid_size_at_top = sum(Decimal(str(_entry(b, "size"))) for b in bids if Decimal(str(_entry(b, "price"))) == best_bid)
            print(f"  best_bid={best_bid}@{bid_size_at_top}  best_ask={best_ask}@{ask_size_at_top}")
            if best_ask > MAX_PRICE_CAP:
                print(f"  skip: best_ask {best_ask} > MAX_PRICE_CAP {MAX_PRICE_CAP}")
                continue
            # Pick a size that fits our budget, but cap at what's available at top of book
            target_size = (TARGET_SPEND_USD / best_ask).quantize(Decimal("1"), rounding=ROUND_DOWN)
            size = min(target_size, ask_size_at_top, bid_size_at_top)
            print(f"  target_size={target_size}  ask_top={ask_size_at_top}  bid_top={bid_size_at_top}  -> size={size}")
            if size < m["min_size"]:
                print(f"  skip: size {size} < market min {m['min_size']}")
                continue
            cost = best_ask * size
            if cost > MAX_BUDGET_USD:
                print(f"  skip: cost {cost} > MAX_BUDGET_USD {MAX_BUDGET_USD}")
                continue
            score = (best_ask - best_bid)
            print(f"  ACCEPT  size={size}  cost={cost}  spread={score}")
            if best is None or score < best["score"]:
                best = {
                    "score": score,
                    "market": m,
                    "token": tok,
                    "best_ask": best_ask,
                    "best_bid": best_bid,
                    "size": size,
                }
    return best


def place_order(client, token_id, price, side, size, tick_size):
    return client.create_and_post_order(
        order_args=OrderArgs(token_id=token_id, price=float(price), side=side, size=float(size)),
        options=PartialCreateOrderOptions(tick_size=str(tick_size)),
        order_type=OrderType.GTC,
    )


def order_id_from_resp(resp):
    if isinstance(resp, dict):
        return resp.get("orderID") or resp.get("orderId") or resp.get("id")
    return getattr(resp, "orderID", None) or getattr(resp, "orderId", None) or getattr(resp, "id", None)


def wait_filled(client, order_id, label):
    deadline = time.time() + FILL_TIMEOUT_S
    last_status = None
    while time.time() < deadline:
        try:
            o = client.get_order(order_id)
        except Exception as e:
            print(f"  [{label}] get_order error: {e}")
            time.sleep(FILL_POLL_SEC)
            continue
        status = o.get("status") if isinstance(o, dict) else getattr(o, "status", None)
        size_matched = o.get("size_matched") if isinstance(o, dict) else getattr(o, "size_matched", None)
        if status != last_status:
            print(f"  [{label}] status={status} size_matched={size_matched}")
            last_status = status
        if status in ("MATCHED", "FILLED"):
            return o
        time.sleep(FILL_POLL_SEC)
    print(f"  [{label}] timeout — cancelling")
    try:
        client.cancel(order_id=order_id)
    except Exception as e:
        print(f"  [{label}] cancel error: {e}")
    return None


def main():
    print(f"funder (proxy): {FUNDER_ADDRESS}")
    print(f"signature type: {SIGNATURE_TYPE_NAME} ({SIGNATURE_TYPE.value})")
    creds = get_creds()
    client = make_client(creds)
    print(f"L2 creds: api_key={creds.api_key[:8]}...")

    candidates = find_candidate_markets()
    if not candidates:
        sys.exit("no London weather markets found")

    pick = pick_token(client, candidates)
    if not pick:
        sys.exit("no actionable cheap side on any London weather market")

    market = pick["market"]
    tok = pick["token"]
    best_ask = pick["best_ask"]
    best_bid = pick["best_bid"]
    size = pick["size"]
    tick_size = market["tick_size"]

    print()
    print(f"PICKED: {market['question']} / {tok['outcome']}")
    print(f"  token_id: {tok['id']}")
    print(f"  best_bid={best_bid}  best_ask={best_ask}  tick={tick_size}  size={size}")

    total_cost = best_ask * size
    if total_cost > MAX_BUDGET_USD:
        sys.exit(f"ABORT: total_cost {total_cost} > MAX_BUDGET_USD {MAX_BUDGET_USD}")

    print()
    print(f"PLAN: BUY {size} @ {best_ask} = ${total_cost}")
    print(f"      SELL {size} @ {best_bid} = ${best_bid * size}")
    print(f"      worst-case loss = ${total_cost - best_bid * size}")
    print()

    # ---- BUY ----
    print(f"[BUY] posting GTC limit {size}@{best_ask}...")
    buy_resp = place_order(client, tok["id"], best_ask, Side.BUY, size, tick_size)
    print(f"[BUY] resp: {buy_resp}")
    buy_id = order_id_from_resp(buy_resp)
    if not buy_id:
        sys.exit("BUY: no order id in response")
    buy_filled = wait_filled(client, buy_id, "BUY")
    if not buy_filled:
        sys.exit("BUY: not filled in time, cancelled")

    # ---- SELL ----
    # Re-quote the bid in case the book moved
    ob = client.get_order_book(tok["id"])
    raw_bids = _ob_get(ob, "bids")
    bids = sorted((Decimal(str(_entry(b, "price"))) for b in raw_bids), reverse=True)
    if not bids:
        sys.exit("SELL: empty bid side after buy — leaving the position to user")
    sell_price = bids[0]
    print()
    print(f"[SELL] posting GTC limit {size}@{sell_price}...")
    sell_resp = place_order(client, tok["id"], sell_price, Side.SELL, size, tick_size)
    print(f"[SELL] resp: {sell_resp}")
    sell_id = order_id_from_resp(sell_resp)
    if not sell_id:
        sys.exit("SELL: no order id in response")
    sell_filled = wait_filled(client, sell_id, "SELL")
    if not sell_filled:
        print("SELL: not fully filled, cancelled what was open. Position may remain.")
        sys.exit(2)

    print()
    print("=" * 60)
    print("OK — V2 ордера работают: подписали, запостили, исполнили BUY+SELL")
    print(f"BUY  {size}@{best_ask}  -> {buy_id}")
    print(f"SELL {size}@{sell_price} -> {sell_id}")
    print("=" * 60)


if __name__ == "__main__":
    main()
