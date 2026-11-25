"""Run from the project root:

uv run --python 3.12 --no-project --with hftbacktest==2.4.4 --with numba --with numpy python bench/test_multi_asset_parity.py
"""

import json
import os
import shlex
import struct
import subprocess
from contextlib import closing
from heapq import merge
from itertools import islice
from operator import itemgetter
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter

import hftbacktest as hft
import numpy as np
from numba import njit

from prepare_real_data import DATA_DIR, events, source_file

BENCH = Path(__file__).resolve().parent
DATA = DATA_DIR / 'multi_asset_test'
SYMBOLS = ('LISTAUSDT', 'CYSUSDT')
COUNT, TICK, LOT = 200_000, 1e-5, 1.0
# The CYS snapshot nearest the start of lista_real.npz is 447 ms earlier.
CYS_START_US = 1_795_356_380_665_000
FIELDS = ('position', 'balance', 'fee', 'num_trades', 'trading_volume', 'trading_value', 'equity', 'final_mid')


def prepare(paths):
    with np.load(DATA_DIR / 'lista_real.npz') as archive:
        lista = archive['data'][:COUNT]
    trade_path = source_file('trades', 'CYSUSDT')
    depth_path = source_file('incremental_book_L2', 'CYSUSDT')
    with (
        closing(events(trade_path, True, CYS_START_US)) as trades,
        closing(events(depth_path, False, CYS_START_US, from_snapshot=True)) as depth,
    ):
        cys = np.fromiter(islice(merge(trades, depth, key=itemgetter(1)), COUNT), dtype=hft.event_dtype)
    assert cys[0]['exch_ts'] == CYS_START_US * 1000
    assert cys[0]['ev'] & 0xff == hft.DEPTH_SNAPSHOT_EVENT
    # Use the same monotonic capture-clock adjustment as the existing LISTA fixture.
    cys['local_ts'] = np.maximum.accumulate(np.maximum(cys['exch_ts'], cys['local_ts']))
    DATA.mkdir(parents=True, exist_ok=True)
    for path, rows in zip(paths, (lista, cys)):
        np.save(path, rows)


@njit
def quote(bt, asset, buy, target):
    oid = 1 if buy else 2
    o = bt.orders(asset).get(oid)
    status = 0
    if o is not None:
        if o.cancellable and o.price_tick != target:
            status = bt.cancel(asset, oid, False)
    elif buy:
        status = bt.submit_buy_order(asset, oid, target * TICK, 1., hft.GTX, hft.LIMIT, False)
    else:
        status = bt.submit_sell_order(asset, oid, target * TICK, 1., hft.GTX, hft.LIMIT, False)
    if status:
        raise RuntimeError('Official order request failed')


@njit
def replay(bt):
    previous = steps = 0
    first, second = bt.depth(0), bt.depth(1)
    while True:
        status = bt.elapse(100_000_000)
        if status == 1:
            break
        if status != 0:
            raise RuntimeError('Official replay failed')
        steps += 1
        b0, a0 = first.best_bid_tick, first.best_ask_tick
        b1, a1 = second.best_bid_tick, second.best_ask_tick
        if not (0 < b0 < a0 < np.iinfo(np.int64).max and 0 < b1 < a1 < np.iinfo(np.int64).max):
            continue
        mid2 = b0 + a0
        skew = 0 if previous == 0 else int(mid2 > previous) - int(mid2 < previous)
        previous = mid2
        for i, bid, ask in ((0, b0, a0), (1, b1 + skew, a1 + skew)):
            bt.clear_inactive_orders(i)
            quote(bt, i, True, bid)
            quote(bt, i, False, ask)
    return steps, bt.current_timestamp


def official(data):
    assets = [hft.BacktestAsset().data(rows).tick_size(TICK).lot_size(LOT).linear_asset(1.)
              .constant_order_latency(750_000, 750_000).trading_value_fee_model(-0.00003, 0.0005)
              .risk_adverse_queue_model().no_partial_fill_exchange() for rows in data]
    bt = hft.HashMapMarketDepthBacktest(assets)
    try:
        start = perf_counter()
        steps, now = replay(bt)
        seconds = perf_counter() - start
        states = []
        for i in range(len(assets)):
            depth, state = bt.depth(i), bt.state_values(i)
            mid = 0.5 * (depth.best_bid + depth.best_ask)
            equity = state.balance + state.position * mid - state.fee
            states.append(dict(zip(FIELDS, (state.position, state.balance, state.fee, state.num_trades,
                                            state.trading_volume, state.trading_value, equity, mid))))
        return dict(steps=steps, current_ts=now, assets=states, total_equity=sum(s['equity'] for s in states)), seconds
    finally:
        bt.close()


def compare(expected, actual):
    def exact(field, ref, found):
        if struct.pack('!d', ref) != struct.pack('!d', found):
            raise AssertionError(f'{field}: expected {ref!r}, found {found!r}')

    for field in ('steps', 'current_ts'):
        if actual[field] != expected[field]:
            raise AssertionError(f'{field}: expected {expected[field]}, found {actual[field]}')
    for symbol, ref, found in zip(SYMBOLS, expected['assets'], actual['assets'], strict=True):
        for field in FIELDS:
            exact(f'{symbol} {field}', ref[field], found[field])
    exact('total_equity', expected['total_equity'], actual['total_equity'])


def main():
    if hft.__version__ != '2.4.4':
        raise RuntimeError(f'Expected official hftbacktest 2.4.4, found {hft.__version__}')
    paths = [DATA / f'{symbol}.npy' for symbol in SYMBOLS]
    if not all(path.is_file() for path in paths):
        prepare(paths)
    data = [np.load(path, mmap_mode='r') for path in paths]
    args = []
    for path, rows in zip(paths, data):
        assert len(rows) == COUNT and rows.dtype == hft.event_dtype
        assert np.all(np.diff(rows['local_ts']) >= 0) and np.all(np.diff(rows['exch_ts']) >= 0)
        ticks = np.rint(rows['px'] / TICK).astype(np.int64)
        assert np.allclose(rows['px'], ticks * TICK, rtol=0, atol=TICK * 1e-8)
        assert np.all(rows['qty'] % LOT == 0)
        args.extend([path, str(TICK), str(LOT), str(ticks.min() - 1), str(ticks.max() + 1)])
    expected, _ = official(data)  # Compile Numba and warm the feed before timing.
    assert all(s['num_trades'] > 0 for s in expected['assets'])
    times = []
    for _ in range(3):
        measured, seconds = official(data)
        compare(expected, measured)
        times.append(seconds)
    rates = {'Rust': 2 * COUNT / median(times)}
    with TemporaryDirectory() as tmp:
        binary = Path(tmp) / 'test_multi_asset'
        flags = ['-std=c++20', '-O3', '-march=native', '-Wall', '-Wextra', '-Wpedantic', '-Werror',
                 '-ffp-contract=off', *shlex.split(os.environ.get('LTO', '-flto=auto'))]
        builds = {'C++': ['-DNDEBUG'],
                  'UBSan': ['-fsanitize=undefined', '-fsanitize-trap=undefined', '-D_GLIBCXX_ASSERTIONS']}
        for name, extra in builds.items():
            subprocess.run([*shlex.split(os.environ.get('CXX', 'c++')), *flags, *extra, '-I', BENCH.parent / 'include',
                            BENCH / 'test_multi_asset.cpp', '-o', binary], check=True)
            times = []
            for _ in range(3):
                measured = json.loads(subprocess.check_output([binary, *args], text=True))
                if measured['hot_allocations'] != 0:
                    raise AssertionError(f"{name}: {measured['hot_allocations']} hot allocations")
                compare(expected, measured)
                times.append(measured['seconds'])
            rates[name] = 2 * COUNT / median(times)
    for symbol, state in zip(SYMBOLS, expected['assets']):
        print(symbol, ', '.join(f'{k}={v!r}' for k, v in state.items()))
    print(f"{2 * COUNT:,} events, {expected['steps']:,} strategy steps, total equity {expected['total_equity']!r}; "
          'all fields bit-exact, 0 hot C++ allocations')
    print('Median throughput (M events/s): ' + ', '.join(f'{k} {v / 1e6:.3f}' for k, v in rates.items()))
    print(f"C++ / Rust median throughput: {rates['C++'] / rates['Rust']:.2f}x")


if __name__ == '__main__':
    main()
