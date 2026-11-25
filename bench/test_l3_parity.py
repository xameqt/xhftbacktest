"""Run with uv run --python 3.12 --no-project --with hftbacktest==2.4.4 --with numba --with numpy python bench/test_l3_parity.py."""

import json
import os
import shlex
import subprocess
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter

import hftbacktest as hft
import numpy as np
from hftbacktest.order import FOK, GTC, GTX, IOC
from numba import njit

TICK, LOT = 0.25, 0.5
ENTRY, RESPONSE = 4, 7
BID, ASK = hft.BUY_EVENT, hft.SELL_EVENT
ADD, CANCEL, MODIFY, FILL, CLEAR = (
    hft.ADD_ORDER_EVENT, hft.CANCEL_ORDER_EVENT, hft.MODIFY_ORDER_EVENT,
    hft.FILL_EVENT, hft.DEPTH_CLEAR_EVENT,
)
OBSERVE, BUY, SELL, CANCEL_REQ, MODIFY_REQ, PURGE = range(6)
FIELDS = ('position balance fee num_trades trading_volume trading_value equity best_bid_tick best_ask_tick '
          'best_bid best_ask bid_qty ask_qty order_count found status req price_tick qty leaves_qty '
          'exec_price_tick exec_qty exch_ts local_ts maker').split()


def fixture():
    rng = np.random.default_rng(244)
    actions, feed, expected_status = [], [], {}
    events = 0
    for episode in range(64):
        start = 1000 + 4000 * episode
        bid = (4094, 8190, 6024, 4058)[episode % 4] + int(rng.integers(-3, 4))
        ask = bid + 8

        def event(t, kind, side=0, oid=0, tick=0, lots=0):
            nonlocal events
            events += 1
            # Split some events into independent exchange/local rows, as in corrected feed data.
            flags = ((hft.EXCH_EVENT, hft.LOCAL_EVENT) if events % 11 == 0 else
                     (hft.EXCH_EVENT | hft.LOCAL_EVENT,))
            for flag in flags:
                feed.append((kind | side | flag, start+t, start+t+3, tick*TICK, lots*LOT, oid, 0, 0.))

        def action(t, op=OBSERVE, oid=0, tick=None, lots=0, tif=GTC, *, expect=None):
            if expect is not None:
                expected_status[len(actions)] = expect
            actions.append((start+t, op, oid, bid if tick is None else tick, lots, tif))

        event(0, CLEAR)
        event(1, ADD, BID, 10, bid, 8)
        event(2, ADD, ASK, 20, ask, 7)
        event(3, ADD, BID, 11, bid-257, 9)
        event(4, ADD, ASK, 21, ask+257, 8)
        action(7, PURGE)
        action(8, BUY, 10, bid, 3, GTX)  # Feed and strategy IDs can coincide.
        event(15, ADD, BID, 12, bid, 8)
        action(16, BUY, 21, bid, 2)
        event(30, FILL, BID, 10, bid, 1)
        action(38, oid=10, expect=hft.NEW)
        action(38, oid=21, expect=hft.NEW)
        action(41, CANCEL_REQ, 10)
        event(45, FILL, BID, 12, bid, 1)  # Same exchange timestamp as the cancel's arrival.
        action(54, oid=10, expect=hft.FILLED)
        action(54, oid=21, expect=hft.NEW)
        event(60, ADD, BID, 13, bid, 5)
        action(63, MODIFY_REQ, 21, bid, 1)
        event(80, FILL, BID, 13, bid, 2)
        action(88, oid=21, expect=hft.FILLED)  # Decreasing quantity retained priority.

        action(94, BUY, 30, bid, 2)
        event(102, ADD, BID, 14, bid, 3)
        action(108, MODIFY_REQ, 30, bid, 5)
        event(120, FILL, BID, 14, bid, 1)
        action(128, oid=30, expect=hft.NEW)  # Increasing quantity lost priority.
        event(129, MODIFY, BID, 14, bid, 6)
        event(130, FILL, BID, 14, bid, 1)
        action(138, oid=30, expect=hft.FILLED)

        action(145, SELL, 31, ask, 2)
        event(160, ADD, ASK, 22, ask, 5)
        action(172, MODIFY_REQ, 31, ask-2, 4)
        event(180, MODIFY, ASK, 22, ask-2, 2)
        event(190, FILL, ASK, 22, ask-2, 1)
        action(199, oid=31, tick=ask-2, expect=hft.FILLED)
        tif_cases = (
            (BUY, ask, GTX, hft.EXPIRED), (SELL, ask, IOC, hft.EXPIRED),
            (SELL, bid, FOK, hft.FILLED), (BUY, ask, IOC, hft.FILLED),
            (BUY, bid, FOK, hft.EXPIRED), (SELL, bid, GTC, hft.FILLED),
        )
        for i, (op, tick, tif, _) in enumerate(tif_cases):
            action(201+i, op, 40+i, tick, 2+i, tif)
        for i, (_, _, _, status) in enumerate(tif_cases):
            action(219, oid=40+i, expect=status)
        action(223, BUY, 32, bid+2, 1)
        event(240, ADD, ASK, 23, bid+2, 4)  # Resting bid crossed by the ask.
        action(248, oid=32, expect=hft.FILLED)
        action(251, SELL, 33, ask-1, 5)
        event(265, MODIFY, BID, 11, ask-1, 8)  # Resting ask crossed by repricing a bid.
        action(273, oid=33, expect=hft.FILLED)
        action(275, SELL, 34, ask+2, 3)
        event(290, CLEAR, ASK)
        action(298, oid=34, tick=ask+2, expect=hft.EXPIRED)
        action(300, BUY, 35, bid, 5)
        event(310, ADD, BID, 24, bid, 6)
        action(313, CANCEL_REQ, 35)
        event(318, FILL, BID, 24, bid, 2)  # Follows the cancel's exchange arrival by one nanosecond.
        action(326, oid=35, expect=hft.CANCELED)
        event(329, CLEAR, BID)
        action(333, PURGE)
        event(340, ADD, BID, 0, bid, 7)
        event(341, ADD, ASK, 1, ask, 6)

        # A live set spanning three bitmap groups, with reused IDs and dense per-price queues.
        # Keep fixed top-of-book anchors while churning both sides and exercising taking orders.
        live = {}
        for n in range(860):
            t = 350 + 4*n
            if n < 96 or not live or rng.integers(4) == 0:
                oid = n+100 if n % 2 else (1 << 63)+n
                side = BID if rng.integers(2) else ASK
                distance = int(rng.choice((1, 2, 3, 7, 64, 3200, 4000)))
                tick = bid-distance if side == BID else ask+distance
                lots = int(rng.integers(1, 17))
                live[oid] = (side, tick, lots)
                event(t, ADD, side, oid, tick, lots)
            else:
                oid = list(live)[int(rng.integers(len(live)))]
                side, tick, lots = live[oid]
                kind = int(rng.choice((MODIFY, MODIFY, FILL, CANCEL)))
                if kind == MODIFY:
                    tick += int(rng.choice((-1, 0, 0, 1)))
                    tick = min(tick, bid-1) if side == BID else max(tick, ask+1)
                    lots = int(rng.integers(1, 17))
                    live[oid] = (side, tick, lots)
                elif kind == CANCEL:
                    del live[oid]
                event(t, kind, side, oid, tick, lots if kind != FILL else 1)
            if n % 37 == 0:
                op = BUY if n % 2 else SELL
                action(t, op, 2000+n, ask if op == BUY else bid, n % 7 + 1, (FOK, IOC)[n % 2])
                action(t+3, oid=2000+n, tick=tick)
            elif n % 29 == 0:
                action(t+3, tick=tick)

        # Orders beyond the depth ROI still have IDs and FIFO priority.
        side, op, outside, inside = ((ASK, SELL, 12300, ask-1) if episode % 2 else
                                     (BID, BUY, -12, bid+1))
        action(3804, op, 70, outside, 3)
        event(3810, ADD, side, 99, outside, 5)
        event(3817, FILL, side, 99, outside, 1)
        action(3825, oid=70, tick=outside, expect=hft.FILLED)
        event(3827, MODIFY, side, 99, inside, 4)
        action(3830, tick=inside)
        event(3838, CANCEL, side, 99, inside)
        action(3841, tick=inside)

        # A fill through a better-priced strategy order, including two orders at the same tick.
        action(3880, BUY, 60, bid-1, 2)
        action(3881, BUY, 61, bid-1, 3)
        event(3885, ADD, BID, 98, bid-5, 8)
        event(3898, FILL, 0, 98, bid-5, 2)  # v2.4.4 ignores fills without a side.
        action(3908, oid=60, expect=hft.NEW)
        event(3912, FILL, BID, 98, bid-5, 2)
        action(3922, oid=60, expect=hft.FILLED)
        action(3922, oid=61, expect=hft.FILLED)
        action(3925, SELL, 62, ask+2, 6)
        event(3930, ADD, ASK, 97, ask+5, 6)
        event(3943, FILL, ASK, 97, ask+5, 1)
        action(3951, oid=62, expect=hft.FILLED)
        action(3956, BUY, 63, bid, 3)
        action(3957, SELL, 64, ask, 4)
        event(3970, CLEAR)
        action(3978, oid=63, expect=hft.EXPIRED)
        action(3978, oid=64, expect=hft.EXPIRED)
        event(3980, ADD, BID, 0, bid, 5)
        event(3981, ADD, ASK, 1, ask, 2)

    action(3999, PURGE)  # Keep the last feed row ahead of this strategy call.
    event(4000, MODIFY, BID, 0, bid, 10)
    action(4050, tick=bid)  # Observe the fully drained engine.
    assert events > 50_000
    return np.array(feed, dtype=hft.event_dtype), np.array(actions, dtype=np.int64), expected_status, events


@njit
def bits(value):
    return np.float64(value).view(np.uint64)


@njit
def checkpoint(bt, oid, tick, row):
    d, s, orders = bt.depth(0), bt.state_values(0), bt.orders(0)
    mid = 0.5 * (d.best_bid + d.best_ask)
    row[0], row[1], row[2] = bits(s.position), bits(s.balance), bits(s.fee)
    row[3], row[4], row[5] = s.num_trades, bits(s.trading_volume), bits(s.trading_value)
    row[6] = bits(s.balance + s.position*mid - s.fee)
    row[7], row[8] = d.best_bid_tick, d.best_ask_tick
    row[9], row[10] = bits(d.best_bid), bits(d.best_ask)
    row[11], row[12] = bits(d.bid_qty_at_tick(tick)), bits(d.ask_qty_at_tick(tick))
    row[13] = len(orders)
    o = orders.get(oid)
    if o is not None:
        row[14], row[15], row[16], row[17] = 1, o.status, o.req, o.price_tick
        row[18], row[19], row[20], row[21] = bits(o.qty), bits(o.leaves_qty), o.exec_price_tick, bits(o.exec_qty)
        row[22], row[23], row[24] = o.exch_timestamp, o.local_timestamp, o.arr[0].maker


@njit
def replay(bt, actions, trace):
    if bt.elapse(0) != 0:
        raise RuntimeError('Rust L3 initialization failed')
    for i in range(len(actions)):
        ts, op, oid, tick, lots, tif = actions[i]
        ret = bt.elapse(ts - bt.current_timestamp)
        if ret != int(i == len(actions)-1):
            raise RuntimeError('Unexpected Rust L3 elapse result', i, ret)
        if op == BUY:
            ret = bt.submit_buy_order(0, oid, tick*TICK, lots*LOT, tif, hft.LIMIT, False)
        elif op == SELL:
            ret = bt.submit_sell_order(0, oid, tick*TICK, lots*LOT, tif, hft.LIMIT, False)
        elif op == CANCEL_REQ:
            ret = bt.cancel(0, oid, False)
        elif op == MODIFY_REQ:
            ret = bt.modify(0, oid, tick*TICK, lots*LOT, False)
        elif op == PURGE:
            bt.clear_inactive_orders(0)
        if op != OBSERVE and ret:
            raise RuntimeError('Rust L3 strategy request failed', i, ret)
        if len(trace):
            checkpoint(bt, oid, tick, trace[i])


def official(feed, actions, trace, expected_final=None):
    asset = (hft.BacktestAsset().data(feed).tick_size(TICK).lot_size(LOT).roi_lb(0.).roi_ub(12287*TICK)
             .linear_asset(1.).constant_order_latency(ENTRY, RESPONSE).trading_value_fee_model(-0.00003, 0.0005)
             .l3_fifo_queue_model().no_partial_fill_exchange())
    bt = hft.ROIVectorMarketDepthBacktest([asset])
    try:
        start = perf_counter()
        replay(bt, actions, trace)
        seconds = perf_counter()-start
        if expected_final is not None:
            row = np.zeros(len(FIELDS), dtype=np.uint64)
            checkpoint(bt, actions[-1,2], actions[-1,3], row)
            np.testing.assert_array_equal(row, expected_final)
        return seconds
    finally:
        bt.close()


def compare(path, expected, actions):
    found = np.fromfile(path, np.uint64).reshape(expected.shape)
    different = np.argwhere(expected != found)
    if different.size:
        i, j = different[0]
        raise AssertionError(f'action {i} {actions[i].tolist()} field {FIELDS[j]}: '
                             f'Rust {expected[i,j]:#018x}, C++ {found[i,j]:#018x}')


def main():
    if hft.__version__ != '2.4.4':
        raise RuntimeError(f'Expected hftbacktest 2.4.4, got {hft.__version__}')
    feed, actions, statuses, count = fixture()
    trace = np.zeros((len(actions), len(FIELDS)), dtype=np.uint64)
    official(feed, actions, trace)  # Includes Numba warm-up.
    for i, status in statuses.items():
        if trace[i,15] != status:
            raise AssertionError(f'Fixture missed expected status {status} at {actions[i]}, got {trace[i,15]}')
    empty = trace[:0]
    rates = {'Rust': count / median(official(feed, actions, empty, trace[-1]) for _ in range(5))}
    bench = Path(__file__).resolve().parent
    with TemporaryDirectory() as tmp:
        binary, data, script, output = (Path(tmp)/s for s in ('runner', 'feed.npy', 'actions.bin', 'trace.bin'))
        np.save(data, feed)
        actions.tofile(script)
        flags = ['-std=c++20', '-O3', '-march=native', '-Wall', '-Wextra', '-Wpedantic', '-Werror', '-ffp-contract=off',
                 '-DXHFT_WRAP_MALLOC',
                 '-Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=aligned_alloc,--wrap=posix_memalign']
        for name, extra in [('C++', ['-DNDEBUG']),
                            ('UBSan', ['-fsanitize=undefined', '-fsanitize-trap=undefined', '-D_GLIBCXX_ASSERTIONS'])]:
            subprocess.run([*shlex.split(os.getenv('CXX', 'c++')), *flags, *extra, '-I', bench.parent/'include',
                            bench/'test_l3_parity.cpp', '-o', binary], check=True)
            times = []
            for dest in (output, '-', '-', '-', '-', '-'):
                result = json.loads(subprocess.check_output([binary, data, script, dest], text=True))
                if result['hot_allocations'] != 0:
                    raise AssertionError(f"{name} allocated {result['hot_allocations']} times during replay")
                if result['final'] != trace[-1].tolist():
                    raise AssertionError(f'{name} final checkpoint disagrees with Rust')
                if dest == output:
                    compare(output, trace, actions)
                else:
                    times.append(result['seconds'])
            rates[name] = count / median(times)
    print(f'{count:,} market events ({len(feed):,} feed rows), {len(actions):,} checkpoints, {trace[-1,3]:,} fills')
    print(f'Final BBO {trace[-1,9:11].view(np.float64).tolist()} at ticks {trace[-1,7:9].tolist()}')
    print('6 native state fields plus equity, BBO, sampled depth and order details bit-exact in both builds; '
          '0 hot allocations')
    print('Median million market events/s (5 runs): ' + ', '.join(f'{k} {v/1e6:.3f}' for k, v in rates.items()))
    print(f"C++ / Rust throughput {rates['C++']/rates['Rust']:.2f}x")


if __name__ == '__main__':
    main()
