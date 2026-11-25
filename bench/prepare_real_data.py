"""Prepare a contiguous LISTAUSDT replay starting at a real November 22 snapshot."""

import csv
import json
import os
import subprocess
from contextlib import ExitStack, closing
from heapq import merge
from itertools import dropwhile, islice
from operator import itemgetter
from pathlib import Path

import numpy as np
from hftbacktest.types import (
    BUY_EVENT, DEPTH_EVENT, DEPTH_SNAPSHOT_EVENT, EXCH_EVENT, LOCAL_EVENT,
    SELL_EVENT, TRADE_EVENT, event_dtype,
)

DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parent / "data"))
# The two earlier snapshot windows are shorter than a million events.
START_US = 1_795_356_381_112_000
LIMIT = 1_500_000
TICKS_PER_USDT = 100_000
TICK, LOT = 1 / TICKS_PER_USDT, 1.0
SIDE = {"buy": BUY_EVENT, "bid": BUY_EVENT, "sell": SELL_EVENT, "ask": SELL_EVENT}


def source_file(kind, symbol):
    root = os.environ.get("RAW_DATA_DIR")
    if not root:
        raise ValueError("Set RAW_DATA_DIR to the archive containing trades/ and incremental_book_L2/.")
    folder = Path(root) / kind / "2026/11/22"
    matches = sorted(folder.glob(f"{symbol}.csv.*"))
    if len(matches) != 1:
        raise ValueError(f"Expected one Zstandard-compressed {symbol}.csv.* in {folder}, found {len(matches)}")
    return matches[0]


def events(path, trades, start_us=START_US, *, from_snapshot=False):
    # The archive uses both .zst and .xz suffixes for Zstandard frames.
    with subprocess.Popen(["zstdcat", "-q", path], stdout=subprocess.PIPE, text=True) as proc:
        with proc.stdout as stream:
            rows = csv.reader(stream)
            marker = "id" if trades else "is_snapshot"
            header = next(rows)
            assert header == ["exchange", "symbol", "timestamp", "local_timestamp",
                              marker, "side", "price", "amount"]
            if from_snapshot:
                rows = dropwhile(lambda row: row[4] != "true" or int(row[2]) != start_us, rows)
            for _, _, ts, local, flag, side, px, qty in rows:
                ts = int(ts)
                if ts < start_us:
                    continue
                kind = TRADE_EVENT if trades else (DEPTH_SNAPSHOT_EVENT if flag == "true" else DEPTH_EVENT)
                yield (EXCH_EVENT | LOCAL_EVENT | SIDE[side] | kind, ts * 1000,
                       int(local) * 1000, float(px), float(qty), 0, 0, 0.0)
        if proc.wait():
            raise OSError(f"Could not decompress {path}")


def main():
    sources = {name: source_file(name, "LISTAUSDT") for name in ("trades", "incremental_book_L2")}
    with ExitStack() as stack:
        streams = [stack.enter_context(closing(events(path, name == "trades")))
                   for name, path in sources.items()]
        # At equal exchange times, process trades before the associated depth changes.
        rows = merge(*streams, key=itemgetter(1))
        data = np.fromiter(islice(rows, LIMIT + 1), dtype=event_dtype)

    if len(data) > LIMIT:
        # Keep the last exchange timestamp group whole.
        data = data[:np.searchsorted(data["exch_ts"], data["exch_ts"][LIMIT])]
    ex, local = data["exch_ts"], data["local_ts"]
    assert 1_000_000 <= len(data) <= LIMIT and data.itemsize == 64
    assert ex[0] == START_US * 1000 and np.all(ex[1:] >= ex[:-1])
    kinds = data["ev"] & 0xff
    assert set(np.unique(kinds)) == {DEPTH_SNAPSHOT_EVENT, DEPTH_EVENT, TRADE_EVENT}
    assert np.array_equal(kinds == DEPTH_SNAPSHOT_EVENT, ex == ex[0])
    # Both clocks must be ordered for a single shared exchange/local record per row.
    ordered_local = np.maximum.accumulate(np.maximum(ex, local))
    adjusted = int(np.count_nonzero(local != ordered_local))
    local[:] = ordered_local
    ticks = np.rint(data["px"] / TICK).astype(np.int64)
    assert np.allclose(data["px"], ticks * TICK, atol=TICK * 1e-8, rtol=0)
    assert np.all(data["qty"] % LOT == 0)
    data["px"] = ticks * TICK

    lo_tick = 9 * int(ticks.min()) // 10
    hi_tick = (11 * int(ticks.max()) + 9) // 10
    meta = dict(tick_size=TICK, lot_size=LOT, roi_lb=lo_tick / TICKS_PER_USDT,
                roi_ub=hi_tick / TICKS_PER_USDT, n_events=len(data),
                start_ts_ns=int(ex[0]), end_ts_ns=int(ex[-1]), end_local_ts_ns=int(local[-1]),
                adjusted_local_timestamps=adjusted,
                source_files={k: str(v.relative_to(os.environ["RAW_DATA_DIR"])) for k, v in sources.items()})
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(DATA_DIR / "lista_real.npz", data=data)
    data.tofile(DATA_DIR / "lista_real.bin")
    (DATA_DIR / "lista_real_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Prepared {len(data):,} events from {ex[0]} to {ex[-1]} ns")


if __name__ == "__main__":
    main()
