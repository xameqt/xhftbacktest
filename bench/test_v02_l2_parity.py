"""uv run --python 3.12 --no-project --with hftbacktest==2.4.4 --with numba --with numpy python bench/test_v02_l2_parity.py"""

import os
import shlex
import struct
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

import hftbacktest as h
from hftbacktest.order import PARTIALLY_FILLED
import numpy as np

BENCH = Path(__file__).resolve().parent
BUY, SELL = h.BUY_EVENT, h.SELL_EVENT
FLAGS = h.EXCH_EVENT | h.LOCAL_EVENT
STATE = ('position', 'balance', 'fee', 'num_trades', 'trading_volume', 'trading_value')
ORDER = ('order_id', 'price_tick', 'qty', 'leaves_qty', 'exec_qty', 'exec_price_tick',
         'side', 'time_in_force', 'status', 'req', 'maker', 'exch_timestamp', 'local_timestamp')
FIELDS = ('timestamp', *STATE, 'equity', 'present', *ORDER)


def bits(x):
    return struct.unpack('=Q', struct.pack('=d', x))[0]


def events(rows):
    return np.array([(FLAGS | side | kind, ts, ts, px, qty, 0, 0, 0.)
                     for ts, side, kind, px, qty in rows], dtype=h.event_dtype)


@dataclass
class Case:
    name: str
    data: np.ndarray
    commands: list
    fill: str = 'partial'
    mode: str = 'linear'
    # Tick, lot, ROI ticks, contract size, maker/taker fees, entry/response ns.
    config: tuple = (1., 1., 1, 200, 3.7, -0.00003, 0.0005, 2, 3)

    # A multi-level taker produces one public Rust response for the whole sweep.
    sweep_legs: dict = field(default_factory=dict)

    def backtest(self, rows, *, ledger=False):
        tick, lot, lo, hi, contract, maker, taker, entry, response = self.config
        asset = (h.BacktestAsset().data(rows).tick_size(tick).lot_size(lot)
                 .roi_lb(lo * tick).roi_ub(hi * tick).risk_adverse_queue_model()
                 .constant_order_latency(0 if ledger else entry, 0 if ledger else response)
                 .trading_value_fee_model(maker, taker))
        (asset.linear_asset if self.mode == 'linear' else asset.inverse_asset)(contract)
        (asset.partial_fill_exchange if self.fill == 'partial' and not ledger
         else asset.no_partial_fill_exchange)()
        return h.ROIVectorMarketDepthBacktest([asset])


def snapshots(commands, mark=101.25):
    """Snapshot active ID after every request, clock advance, and settlement."""
    out, oid = [], 1
    for line in commands.splitlines():
        words = line.split()
        if not words:
            continue
        op, *args = words
        args = [float(x) if '.' in x else int(x) for x in args]
        if op in ('B', 'S', 'M', 'C'):
            oid = args[0]
        out.extend([(op, *args), ('P', oid, mark)])
    return out


def synthetic():
    depth, trade = h.DEPTH_EVENT, h.TRADE_EVENT
    book = [(100, BUY, depth, 99, 20), (100, BUY, depth, 100, 10),
            (100, SELL, depth, 103, 8), (100, SELL, depth, 104, 3)]
    end = [(10000, BUY, depth, 99, 20)]
    rows = events(book + [(115, SELL, trade, 100, 7), (125, SELL, trade, 100, 4),
                          (135, SELL, trade, 100, 2), (160, BUY, trade, 103, 10),
                          (170, BUY, trade, 103, 8)] + end)
    cmds = snapshots("""E 0
        B 1 100 12 1
        E 4
        E 1
        E 10
        M 1 100 8
        E 4
        E 1
        E 7
        E 1
        M 1 100 4
        E 5
        E 5
        C 1
        E 5
        X
        S 2 103 9 1
        E 5
        E 15
        E 10
        F 0.001 102.75
        F -0.002 107.25""")
    for mode in ('linear', 'inverse'):
        yield Case(f'maker partial, reduce, cancel, sell / {mode}', rows, cmds, mode=mode)

    rows = events(book + [(110, SELL, trade, 100, 7), (123, SELL, trade, 100, 4),
                          (130, SELL, trade, 99, 8), (140, SELL, trade, 99, 14)] + end)
    cmds = snapshots("""E 0
        B 1 100 12 1
        E 13
        M 1 100 16
        E 4
        E 1
        E 8
        M 1 99 8
        E 5
        E 2
        E 10
        S 2 103 3 1
        E 5
        M 2 100 3
        E 4
        E 1""")
    for fill in ('full', 'partial'):
        yield Case(f'increase, reprice, GTX crossing / {fill}', rows, cmds, fill=fill)

    cmds = snapshots("""E 0
        B 1 103 7 0
        E 5
        F 0.001 97.5
        S 2 100 9 0
        E 5
        F -0.002 110.5""")
    for mode in ('linear', 'inverse'):
        yield Case(f'taker and funding / {mode}', events(book + end), cmds, mode=mode)

    # FOK takes 8 at 103 then 2 at 104. IOC exhausts both bid levels.
    # GTC sells the residual at its limit even after consuming that level.
    cmds = snapshots("""E 0
        B 1 104 10 2
        E 5
        S 2 98 35 3
        E 5
        S 3 99 35 0
        E 5
        B 4 104 20 2
        E 5""")
    legs = {1: [(103, 8), (104, 2)], 2: [(100, 10), (99, 20)],
            3: [(100, 10), (99, 20), (99, 5)]}
    yield Case('FOK, IOC and GTC sweeps', events(book + end), cmds, mode='inverse', sweep_legs=legs)

    # Multiple fills on one order, stopping one nanosecond before and at each response.
    count = 129
    rows = events(book + [(110 + 5 * i, BUY, trade, 102, 1) for i in range(count)] + end)
    cmds = snapshots('E 0\nS 1 102 150 1\nE 8\n' + 'E 4\nE 1\n' * count + 'C 1\nE 5')
    yield Case('129 consecutive partials and cancel', rows, cmds)


def real_case():
    directory = Path(os.environ.get('DATA_DIR', BENCH / 'data'))
    npy, npz, raw = (directory / 'multi_asset_test/LISTAUSDT.npy', directory / 'lista_real.npz',
                     directory / 'lista_real.bin')
    if raw.is_file():
        data = np.memmap(raw, mode='r', dtype=h.event_dtype)
    elif npz.is_file():
        with np.load(npz) as archive:
            data = archive['data']
    elif npy.is_file():
        data = np.load(npy, mmap_mode='r')
    elif (directory / 'LISTAUSDT.npy').is_file():
        data = np.load(directory / 'LISTAUSDT.npy', mmap_mode='r')
    else:
        return None
    assert data.dtype == h.event_dtype
    tick = 1e-5
    # Pick an isolated aggressor that trades more than the displayed queue at its price.
    depth = {}
    for i, e in enumerate(data[:-1]):
        ev, ts, qty, px = int(e['ev']), int(e['exch_ts']), float(e['qty']), float(e['px'])
        if not ev & h.EXCH_EVENT:
            continue
        side, level = ev & (BUY | SELL), round(px / tick)
        if ev & 255 in (h.DEPTH_EVENT, h.DEPTH_SNAPSHOT_EVENT):
            depth[side, level] = qty
        if (ev & 255 == h.TRADE_EVENT and i and data[i-1]['exch_ts'] < ts - 10
                and data[i+1]['exch_ts'] > ts + 10 and qty > depth.get((side ^ (BUY | SELL), level), 0)):
            place = 'S' if side == BUY else 'B'
            start = int(min(data[0]['local_ts'], data[0]['exch_ts']))
            cmds = snapshots(f'E {ts - start - 8}\n{place} 1 {px:.12f} {int(qty + 10)} 1\n'
                             f'E 5\nE 6\nC 1\nE 5\nE {int(data[-1]["local_ts"]) - ts + 20}', px)
            return Case('LISTAUSDT partial and cancel', data, cmds,
                        config=(tick, 1., 1, 2 * level, 1., -0.00003, 0.0005, 2, 3))
    raise ValueError('No isolated trade exceeded the displayed queue in the LISTAUSDT data')


def state(bt):
    s = bt.state_values(0)
    return tuple(getattr(s, k) for k in STATE)


def account_executions(case, fills):
    """Obtain Rust account states for every execution, including its unbooked partials.

    The 2.4.4 local processor calls apply_fill only for FILLED. Replay each observed
    execution as one complete order in an independent Rust account, keeping side,
    tick, size, maker/taker classification, fees and execution order identical.
    """
    rows = []
    tick = case.config[0]
    for i, (side, level, qty, maker) in enumerate(fills):
        t = 100 + 10 * i
        px = tick * level
        rows.append((t, 0, h.DEPTH_CLEAR_EVENT, 0., 0.))
        if maker:
            rows.append((t+3, SELL if side == 1 else BUY, h.TRADE_EVENT, px, qty+1))
        else:
            rows.append((t, SELL if side == 1 else BUY, h.DEPTH_EVENT, px, qty))
    rows.append((100 + 10 * len(fills), BUY, h.DEPTH_EVENT, tick, 1.))
    data = events(rows)
    bt = case.backtest(data, ledger=True)
    states = []
    try:
        assert bt.elapse(0) == 0
        for i, (side, level, qty, maker) in enumerate(fills):
            fn = bt.submit_buy_order if side == 1 else bt.submit_sell_order
            assert fn(0, i+1, tick * level, qty, h.GTX if maker else h.GTC, h.LIMIT, False) == 0
            assert bt.elapse(3) == 0
            o = bt.orders(0).get(i+1)
            assert o.status == h.FILLED and o.exec_price_tick == level and o.exec_qty == qty
            assert bool(o.arr[0]['maker']) == maker
            states.append(state(bt))
            if i+1 < len(fills):
                assert bt.elapse(7) == 0
        return states
    finally:
        bt.close()


def oracle(case):
    bt = case.backtest(case.data)
    fills, observed, samples, funding = [], set(), [], []
    try:
        for op, *args in case.commands:
            if op in ('B', 'S'):
                oid, px, qty, tif = args
                fn = bt.submit_buy_order if op == 'B' else bt.submit_sell_order
                assert fn(0, oid, px, qty, tif, h.LIMIT, False) == 0
            elif op == 'M':
                assert bt.modify(0, *args, False) == 0
                # A reduce ACK can retain the previous execution and PARTIALLY_FILLED status.
                observed.add((args[0], bt.current_timestamp + case.config[-2]))
            elif op == 'C':
                assert bt.cancel(0, *args, False) == 0
            elif op == 'X':
                bt.clear_inactive_orders(0)
            elif op == 'E':
                assert bt.elapse(*args) in (0, 1)
            elif op == 'F':
                funding.append(tuple(args))
            elif op == 'P':
                oid, mark = args
                o = bt.orders(0).get(oid)
                fields = []
                if o is not None:
                    row = o.arr[0]
                    fields = [bits(row[k]) if row.dtype[k].kind == 'f' else int(row[k]) for k in ORDER]
                    key = (oid, o.exch_timestamp)
                    if oid in case.sweep_legs and o.status in (h.EXPIRED, h.FILLED) and key not in observed:
                        legs = case.sweep_legs[oid]
                        assert (o.exec_price_tick, o.exec_qty) == legs[-1]
                        assert o.qty - o.leaves_qty == sum(q for _, q in legs)
                        fills.extend((o.side, tick, qty, False) for tick, qty in legs)
                        observed.add(key)
                    elif o.status in (PARTIALLY_FILLED, h.FILLED) and o.req == 0 and key not in observed:
                        fills.append((o.side, o.exec_price_tick, o.exec_qty, bool(row['maker'])))
                        observed.add(key)
                samples.append((bt.current_timestamp, mark, state(bt), len(fills), funding, fields))
                funding = []
    finally:
        bt.close()
    corrected = any(s[3] != s[2][3] for s in samples)
    states = account_executions(case, fills) if corrected else []
    changes = len(fills) - samples[-1][2][3]
    if 'partial' in case.name and 'GTX crossing' not in case.name:
        assert changes > 0, f'{case.name} did not exercise a partial execution'
    expected = []
    settled_balance, previous = None, 0
    for ts, mark, raw, count, settlements, order in samples:
        ledger = states[count-1] if corrected and count else raw
        if ledger[3] == raw[3]:
            assert [bits(v) for v in ledger] == [bits(v) for v in raw]
        pos, balance, fee, trades, volume, value = ledger
        contract = case.config[4]
        if settled_balance is not None:
            # After settlement the raw Rust account has a different balance.
            # Book subsequent executions on the settled balance in their original order.
            balance = settled_balance
            for side, level, qty, _ in fills[previous:count]:
                px = level * case.config[0]
                amount = contract * qty / px if case.mode == 'inverse' else contract * px * qty
                balance -= side * amount
        for rate, price in settlements:
            # Inverse raw balance is negated in Rust equity. Positive rate debits longs.
            if case.mode == 'inverse':
                balance += (contract * pos / price) * rate
            else:
                balance -= (contract * price * pos) * rate
        if settlements or settled_balance is not None:
            settled_balance = balance
        previous = count
        equity = (-balance - contract * pos / mark - fee if case.mode == 'inverse' else
                  balance + contract * pos * mark - fee)
        expected.append([ts, bits(pos), bits(balance), bits(fee), trades, bits(volume),
                         bits(value), bits(equity), bool(order), *order])
    return expected, changes


def main():
    if h.__version__ != '2.4.4':
        raise RuntimeError(f'Expected hftbacktest 2.4.4, found {h.__version__}')
    cases = list(synthetic())
    real = real_case()
    if real:
        cases.append(real)
    else:
        print('LISTAUSDT unavailable in DATA_DIR, using synthetic stress data.')
    with TemporaryDirectory() as tmp:
        fixtures = []
        for i, case in enumerate(cases):
            path = Path(tmp) / f'{i}.npy'
            np.save(path, case.data)
            ref, omitted = oracle(case)
            script = '\n'.join(' '.join(map(str, row)) for row in (case.config, *case.commands)) + '\n'
            fixtures.append((case, path, script, ref))
            print(f'{case.name}: {len(case.data):,} events, {len(ref)} checkpoints, '
                  f'{omitted} executions omitted by Rust local state.')
        for build, flags in [('release', ['-DNDEBUG']), ('UBSan', ['-fsanitize=undefined',
                                  '-fsanitize-undefined-trap-on-error', '-D_GLIBCXX_ASSERTIONS'])]:
            exe = Path(tmp) / build
            subprocess.run([*shlex.split(os.environ.get('CXX', 'c++')), '-std=c++20', '-O3', '-march=native',
                            '-Wall', '-Wextra', '-Wpedantic', '-Werror', '-ffp-contract=off', *flags,
                            '-DXHFT_TEST_WRAP_ALLOC',
                            '-Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=aligned_alloc,--wrap=posix_memalign',
                            '-I', BENCH.parent / 'include', BENCH / 'test_v02_l2_parity.cpp', '-o', exe], check=True)
            for case, path, script, ref in fixtures:
                out = subprocess.check_output([exe, case.fill, case.mode, path], input=script, text=True)
                for i, (expected, row) in enumerate(zip(ref, out.splitlines(), strict=True)):
                    for k, (a, b) in enumerate(zip(expected, map(int, row.split()), strict=True)):
                        if a != b:
                            raise AssertionError(f'{build} {case.name} snapshot {i} {FIELDS[k]}: {b} != {a}')
            print(f'{build}: {sum(len(f[3]) for f in fixtures)} bit-exact checkpoints.', flush=True)
            subprocess.run([exe, '--self-test'], check=True)


if __name__ == '__main__':
    main()
