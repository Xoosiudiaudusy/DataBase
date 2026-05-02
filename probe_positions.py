"""Diagnostic: find out where the bought 12 YES shares actually live.

Probes Polymarket's data-api positions endpoint for both candidate addresses:
  - FUNDER_ADDRESS that we configured (the explicit funder)
  - The EOA derived from PRIVATE_KEY (sanity check)
Lists positions on the 18C-London-YES token.
Also fetches recent trades for the funder and checks the on-chain
receipt of any provided BUY_TXHASH.
"""
import os
import sys
import requests
from dotenv import load_dotenv
from eth_account import Account

load_dotenv()

DATA_API = "https://data-api.polymarket.com"
POLYGON_RPC = os.environ.get("POLYGON_RPC", "https://polygon-rpc.com")
TOKEN_ID = "78604544123992331734312725690662921776862929985583358454851645001147317024858"
BUY_TXHASH = os.environ.get("BUY_TXHASH", "0x9afd33a75ad4817fb1c78247fc05df668f6808bd73a16e5dd0333ff5cf6fcd87")

PK = os.environ["PRIVATE_KEY"]
FUNDER = os.environ["FUNDER_ADDRESS"]
EOA = Account.from_key(PK).address

print(f"EOA:    {EOA}")
print(f"FUNDER: {FUNDER}")
print()

for label, addr in [("FUNDER", FUNDER), ("EOA", EOA)]:
    print(f"=== positions for {label} {addr} ===")
    try:
        r = requests.get(f"{DATA_API}/positions",
                         params={"user": addr, "limit": 50}, timeout=30)
        print(f"  status={r.status_code}")
        if r.status_code == 200:
            data = r.json()
            positions = data if isinstance(data, list) else []
            print(f"  total positions: {len(positions)}")
            our_token = [p for p in positions if p.get("asset") == TOKEN_ID]
            if our_token:
                print(f"  >>> 18C London YES position FOUND: size={our_token[0].get('size')}")
            else:
                print(f"  >>> 18C London YES NOT in positions")
        else:
            print(f"  body[:300]: {r.text[:300]!r}")
    except Exception as e:
        print(f"  ERROR: {e!r}")
    print()

print(f"=== recent trades for FUNDER {FUNDER} ===")
try:
    r = requests.get(f"{DATA_API}/trades", params={"user": FUNDER, "limit": 20}, timeout=30)
    print(f"  status={r.status_code}")
    if r.status_code == 200:
        trades = r.json()
        if isinstance(trades, list):
            for t in trades[:20]:
                print(f"  {t.get('timestamp','?')} {t.get('side','?')} sz={t.get('size','?')} px={t.get('price','?')} asset={str(t.get('asset',''))[:25]}... title={t.get('title','')[:50]}")
        else:
            print(f"  unexpected: {trades}")
    else:
        print(f"  body[:300]: {r.text[:300]!r}")
except Exception as e:
    print(f"  ERROR: {e!r}")
print()

print(f"=== Polygon receipt for BUY_TXHASH {BUY_TXHASH} ===")
try:
    r = requests.post(POLYGON_RPC, json={
        "jsonrpc": "2.0", "id": 1,
        "method": "eth_getTransactionReceipt",
        "params": [BUY_TXHASH],
    }, timeout=30)
    res = r.json().get("result")
    if not res:
        print(f"  no receipt yet (tx not mined?). raw: {r.text[:300]}")
    else:
        print(f"  status: {res.get('status')} (1=success, 0=reverted)")
        print(f"  blockNumber: {int(res.get('blockNumber','0x0'),16)}")
        print(f"  from: {res.get('from')}  to: {res.get('to')}")
        print(f"  gasUsed: {int(res.get('gasUsed','0x0'),16)}")
        print(f"  logs: {len(res.get('logs',[]))}")
        # Try to find Transfer to FUNDER among ConditionalTokens TransferSingle logs
        # ERC1155 TransferSingle: keccak256("TransferSingle(address,address,address,uint256,uint256)")
        TRANSFER_SINGLE = "0xc3d58168c5ae7397731d063d5bbf3d657854427343f4c083240f7aacaa2d0f62"
        for i, log in enumerate(res.get("logs", [])):
            topic0 = (log.get("topics") or ["", ""])[0]
            if topic0 == TRANSFER_SINGLE:
                # topics[2] = from, topics[3] = to (padded)
                topics = log.get("topics", [])
                if len(topics) >= 4:
                    from_addr = "0x" + topics[2][-40:]
                    to_addr = "0x" + topics[3][-40:]
                    print(f"    log[{i}] TransferSingle from={from_addr} to={to_addr}")
except Exception as e:
    print(f"  ERROR: {e!r}")

