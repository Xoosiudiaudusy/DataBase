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

WEATHER_KEYWORDS = ("weather", "temperature", "temp ", "rain", "snow", "hottest", "coldest", "wettest")
LOCATION_KEYWORDS = ("london",)


def make_client(creds=None):
    return ClobClient(
        host=HOST,
        chain_id=CHAIN_ID,
        key=PRIVATE_KEY,
        signature_type=SignatureTypeV2.POLY_1271,
        funder=FUNDER_ADDRESS,
        creds=creds,
    )


def get_creds():
    if all(os.environ.get(k) for k in ("CLOB_API_KEY", "CLOB_SECRET", "CLOB_PASS_PHRASE")):
        return ApiCreds(
            api_key=os.environ["CLOB_API_KEY"],
            api_secret=os.environ["CLOB_SECRET"],
            api_passphrase=os.environ["CLOB_PASS_PHRASE"],
        )
    print("[creds] deriving L2 credentials from PRIVATE_KEY...")
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


def find_candidate_markets():
    print("[gamma] fetching active markets...")
    out = []
    for offset in (0, 500, 1000):
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
    print(f"[gamma] got {len(out)} markets")

    candidates = []
    for m in out:
        if not m.get("acceptingOrders"):
            continue
        text = (m.get("question", "") + " " + m.get("slug", "")).lower()
        if not any(k in text for k in LOCATION_KEYWORDS):
            continue
        if not any(k in text for k in WEATHER_KEYWORDS):
            continue
        token_ids = _parse_json_field(m.get("clobTokenIds"))
        outcomes = _parse_json_field(m.get("outcomes"))
        prices = _parse_json_field(m.get("outcomePrices"))
        if len(token_ids) != 2 or len(prices) != 2:
            continue
        candidates.append({
            "question": m.get("question"),
            "slug": m.get("slug"),
            "tick_size": str(m.get("orderPriceMinTickSize") or m.get("minimumTickSize") or "0.01"),
            "min_size": Decimal(str(m.get("orderMinSize") or m.get("minimumOrderSize") or "5")),
            "tokens": [
                {"id": str(token_ids[0]), "outcome": outcomes[0] if outcomes else "0", "approx_price": Decimal(prices[0])},
                {"id": str(token_ids[1]), "outcome": outcomes[1] if len(outcomes) > 1 else "1", "approx_price": Decimal(prices[1])},
            ],
        })
    print(f"[gamma] {len(candidates)} candidate London-weather markets")
    return candidates


def pick_token(client, candidates):
    """For each candidate, query the orderbook and pick the cheapest actionable side."""
    best = None
    for m in candidates:
        for tok in m["tokens"]:
            if tok["approx_price"] > MAX_PRICE_CAP:
                continue
            try:
                ob = client.get_order_book(tok["id"])
            except Exception as e:
                print(f"  [skip] {m['slug']}/{tok['outcome']}: get_order_book failed: {e}")
                continue
            asks = getattr(ob, "asks", None) or []
            bids = getattr(ob, "bids", None) or []
            if not asks or not bids:
                continue
            ask_prices = sorted(Decimal(str(a.price)) for a in asks)
            bid_prices = sorted((Decimal(str(b.price)) for b in bids), reverse=True)
            best_ask = ask_prices[0]
            best_bid = bid_prices[0]
            ask_size_at_top = sum(Decimal(str(a.size)) for a in asks if Decimal(str(a.price)) == best_ask)
            bid_size_at_top = sum(Decimal(str(b.size)) for b in bids if Decimal(str(b.price)) == best_bid)
            print(f"  {m['slug']}/{tok['outcome']}: bid={best_bid}@{bid_size_at_top} ask={best_ask}@{ask_size_at_top}")
            if best_ask > MAX_PRICE_CAP:
                continue
            need_size = (TARGET_SPEND_USD / best_ask).quantize(Decimal("1"), rounding=ROUND_DOWN)
            if need_size < m["min_size"] or ask_size_at_top < need_size or bid_size_at_top < need_size:
                continue
            score = (best_ask - best_bid)  # tighter spread is better
            if best is None or score < best["score"]:
                best = {
                    "score": score,
                    "market": m,
                    "token": tok,
                    "best_ask": best_ask,
                    "best_bid": best_bid,
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
    tick_size = market["tick_size"]

    print()
    print(f"PICKED: {market['question']} / {tok['outcome']}")
    print(f"  token_id: {tok['id']}")
    print(f"  best_bid={best_bid}  best_ask={best_ask}  tick={tick_size}")

    size = (TARGET_SPEND_USD / best_ask).quantize(Decimal("1"), rounding=ROUND_DOWN)
    total_cost = best_ask * size

    if total_cost > MAX_BUDGET_USD:
        sys.exit(f"ABORT: total_cost {total_cost} > MAX_BUDGET_USD {MAX_BUDGET_USD}")
    if size < market["min_size"]:
        sys.exit(f"ABORT: size {size} < market min {market['min_size']}")

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
    bids = sorted((Decimal(str(b.price)) for b in (getattr(ob, "bids", None) or [])), reverse=True)
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
