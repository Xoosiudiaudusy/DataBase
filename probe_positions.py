"""Diagnostic: find out where the bought 12 YES shares actually live.

Probes Polymarket's data-api positions endpoint for both candidate addresses:
  - FUNDER_ADDRESS that we configured (the explicit funder)
  - The EOA derived from PRIVATE_KEY (sanity check)
Lists positions on the 18C-London-YES token.
"""
import os
import requests
from dotenv import load_dotenv
from eth_account import Account

load_dotenv()

DATA_API = "https://data-api.polymarket.com"
TOKEN_ID = "78604544123992331734312725690662921776862929985583358454851645001147317024858"

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
            print(f"  total positions: {len(data) if isinstance(data, list) else 'n/a'}")
            for p in (data if isinstance(data, list) else []):
                if p.get("asset") == TOKEN_ID or str(p.get("conditionId","")).lower() in str(p):
                    print(f"  >>> match: {p}")
                else:
                    sz = p.get("size") or p.get("balance")
                    if sz:
                        print(f"      {p.get('asset','?')[:20]}... size={sz} title={p.get('title','')[:60]}")
        else:
            print(f"  body[:300]: {r.text[:300]!r}")
    except Exception as e:
        print(f"  ERROR: {e!r}")
    print()
