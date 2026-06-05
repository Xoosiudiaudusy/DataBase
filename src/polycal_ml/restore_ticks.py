"""Re-derive per-market tick files from HuggingFace quant.parquet.

The 181 MB weather_trades.parquet (and the 13K per-market tick files derived
from it) are NOT committed — they're re-derivable in ~5 min via DuckDB. This
script streams quant.parquet from HF, filters to our weather condition_ids,
and writes per-market tick parquets.

Run once after cloning:  python -m polycal_ml.restore_ticks
"""
from __future__ import annotations
import time
from pathlib import Path
import pandas as pd

HF_URL = "https://huggingface.co/datasets/SII-WANGZJ/Polymarket_data/resolve/main/quant.parquet"
WEATHER_LIST = Path("results/weather_markets_canonical.parquet")
TRADES_OUT = Path("cache/hf_polymarket/weather_trades.parquet")
TICKS_OUT = Path("cache/polymarket/ticks")
T_START = 1735689600  # 2025-01-01 UTC


def stream_filter():
    import duckdb
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute(f"CREATE TABLE w AS SELECT condition_id FROM '{WEATHER_LIST}'")
    TRADES_OUT.parent.mkdir(parents=True, exist_ok=True)
    print("Streaming + filtering quant.parquet from HF (~5 min)...")
    con.execute(f"""
        COPY (SELECT q.* FROM '{HF_URL}' q
              INNER JOIN w ON q.condition_id = w.condition_id
              WHERE q.timestamp >= {T_START})
        TO '{TRADES_OUT}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    print(f"  wrote {TRADES_OUT}")


def split_per_market():
    TICKS_OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(TRADES_OUT)
    markets = pd.read_parquet(WEATHER_LIST)
    cid_to_yes = dict(zip(markets["condition_id"], markets["token1"]))
    df["ts_utc"] = pd.to_datetime(df["timestamp"], unit="s")
    df["asset"] = df["condition_id"].map(cid_to_yes)
    df = df.dropna(subset=["asset"])
    n = 0
    for cid, grp in df.groupby("condition_id"):
        grp[["ts_utc", "price", "asset", "side"]].sort_values("ts_utc").to_parquet(
            TICKS_OUT / f"{cid}.parquet", index=False)
        n += 1
    print(f"  wrote {n} per-market tick files to {TICKS_OUT}")


if __name__ == "__main__":
    t0 = time.time()
    if not TRADES_OUT.exists():
        stream_filter()
    else:
        print(f"{TRADES_OUT} already present, skipping stream")
    split_per_market()
    print(f"Done in {time.time()-t0:.0f}s")
