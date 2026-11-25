"""Run the nanobind unit tests and any prepared real-data parity fixtures."""

import argparse
import csv
import json
import math
import os
import sys
import weakref
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest import SkipTest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "build"), str(ROOT / "bench")]
import xhftbacktest as xh

DATA = Path(os.environ.get("DATA_DIR", ROOT / "bench" / "data"))
LATENCY = 750_000
GTX = xh.TimeInForce.GTX
EVENT_DTYPE = np.dtype([("ev", "u8"), ("exch_ts", "i8"), ("local_ts", "i8"), ("px", "f8"),
                        ("qty", "f8"), ("order_id", "u8"), ("ival", "i8"), ("fval", "f8")], align=True)
EXCH, LOCAL, BUY, SELL = 1 << 31, 1 << 30, 1 << 29, 1 << 28
DEPTH, TRADE, ADD, CANCEL, FILL = 1, 2, 10, 11, 13


def event(ts, flags, price, qty, order_id=0):
    return EXCH | LOCAL | flags, ts, ts, price, qty, order_id, 0, 0.


def quotes():
    return np.array([event(10, DEPTH | BUY, 8, 1), event(10, DEPTH | SELL, 16, 2),
                     event(50, DEPTH | BUY, 8, 3)], dtype=EVENT_DTYPE)


def test_borrowed_array():
    config = xh.AssetConfig(1, 1, 0, 20)
    for multi_asset in (False, True):
        rows = quotes()
        if multi_asset:
            bt = xh.MultiAssetPowerProb()
            bt.add_asset_array(config, rows, 0, 0)
        else:
            bt = xh.BacktestRiskAdverse(config, rows, 0, 0)
        # Editing a future event distinguishes borrowing from copying the payload.
        rows[0]["px"], rows[0]["qty"] = 11, 7
        ref = weakref.ref(rows)
        del rows
        assert ref() is not None
        if multi_asset:
            assert bt.wait_next_feed(False, 0) == 2 and bt.best_bid(0) == 11
        else:
            assert bt.elapse(0) and bt.bid_qty_at_tick(11) == 7
        del bt
        assert ref() is None


def test_array_layout():
    config = xh.AssetConfig(1, 1, 0, 20)
    rows = np.empty(4, dtype=EVENT_DTYPE)
    unaligned = np.ndarray(2, dtype=rows.dtype, buffer=bytearray(129), offset=1)
    for invalid in (rows[::2], rows.reshape(2, 2), rows.view("V64"),
                    rows.view(rows.dtype.newbyteorder("S")), unaligned):
        with np.testing.assert_raises((TypeError, ValueError)):
            xh.BacktestRiskAdverse(config, invalid, 0, 0)


def test_partial_fills():
    rows = np.array([event(10, DEPTH | BUY, 8, 1), event(10, DEPTH | SELL, 16, 2),
                     event(20, TRADE | SELL, 8, 3), event(30, TRADE | SELL, 8, 2)], dtype=EVENT_DTYPE)
    for cls in (xh.BacktestPartialRiskAdverse, xh.BacktestPartialPowerProb):
        bt = cls(xh.AssetConfig(1, 1, 0, 20), rows, 0, 0)
        assert bt.elapse(0) and bt.submit_buy_order(1, 8, 4, GTX)
        # One lot in front of our order absorbs the first lot of the trade at 20.
        assert bt.elapse(10) and bt.position == 2
        order, = bt.active_orders
        assert (order.status, order.exec_qty, order.leaves_qty) == (xh.OrderStatus.PARTIALLY_FILLED, 2, 2)
        assert not bt.elapse(10)
        assert bt.active_orders[0].status == xh.OrderStatus.FILLED
        assert (bt.position, bt.state_values.balance, bt.state_values.num_trades) == (4, -32, 2)


def test_inverse_modify_and_funding():
    cfg = xh.AssetConfig(1, 1, 0, 20, contract_size=8, taker_fee=0.125)
    bt = xh.BacktestInverseRiskAdverse(cfg, quotes(), 1, 1)
    assert bt.elapse(0) and bt.submit_buy_order(7, 8, 4, xh.TimeInForce.GTC)
    assert not bt.modify(7, 16, 2)  # New order awaits its acknowledgment.
    assert bt.elapse(2) and bt.active_orders[0].status == xh.OrderStatus.NEW
    assert bt.modify(7, 8, 2) and bt.elapse(2)
    assert bt.active_orders[0].leaves_qty == 2
    assert bt.modify(order_id=7, price=16, qty=2)
    assert bt.elapse(1) and bt.position == 0
    assert bt.elapse(1) and bt.position == 2
    state = bt.state_values
    assert (state.balance, state.trading_value, state.fee) == (-1, 1, 0.125)
    bt.apply_funding(funding_rate=0.25, mark_price=8)
    assert (bt.state_values.balance, bt.state_values.num_trades) == (-0.5, 1)
    recorder = xh.Recorder(1)
    assert recorder.record(bt.current_timestamp, 8, bt.state_values, contract_size=8)
    assert recorder.summary().final_equity == -1.625


def test_multi_modify_and_funding():
    cfg = xh.AssetConfig(1, 1, 0, 20, contract_size=2)
    for cls in (xh.MultiAssetRiskAdverse, xh.MultiAssetPowerProb):
        bt = cls()
        for i in range(2):
            assert bt.add_asset_array(cfg, quotes(), 0, 0) == i
        assert bt.elapse(0) and bt.submit_buy_order(1, 7, 8, 2, xh.TimeInForce.GTC)
        assert bt.elapse(0) and bt.modify(1, 7, 16, 2) and bt.elapse(0)
        assert bt.position(1) == 2 and bt.state_values(1).balance == -64
        bt.apply_funding(1, 0.125, 12)
        assert bt.state_values(1).balance == -70
        assert bt.position(0) == bt.state_values(0).balance == 0
        assert bt.total_equity() == -22


def test_l3_fifo():
    cfg = xh.AssetConfig(1, 1, 0, 20)
    rows = np.array([event(10, ADD | BUY, 8, 5, 100), event(10, ADD | SELL, 16, 4, 200),
                     event(20, ADD | BUY, 8, 1, 101), event(30, FILL | BUY, 8, 1, 100),
                     event(40, FILL | BUY, 8, 1, 101), event(50, CANCEL | BUY, 8, 0, 100)],
                    dtype=EVENT_DTYPE)
    with TemporaryDirectory() as temp:
        path = Path(temp) / "orders.npy"
        np.save(path, rows)
        for source in (path, rows):
            bt = xh.L3Backtest(cfg, source, 0, 0)
            if source is rows:
                rows[0]["qty"] = 6  # The engine borrows the array.
            assert bt.elapse(0) and bt.current_timestamp == 10
            assert bt.best_bid == bt.best_bid_tick == 8 and bt.best_ask == bt.best_ask_tick == 16
            assert bt.bid_qty_at_tick(8) == (6 if source is rows else 5) and bt.ask_qty_at_tick(16) == 4
            assert bt.submit_buy_order(1, 8, 3, GTX) and bt.submit_sell_order(2, 16, 1, GTX)
            assert bt.elapse(0) and bt.modify(1, 8, 2) and bt.cancel(2)
            assert bt.elapse(20) and bt.position == 0  # Fill of the older feed order.
            bt.clear_inactive_orders()
            order, = bt.active_orders
            assert (order.order_id, order.qty, order.status) == (1, 2, xh.OrderStatus.NEW)
            assert bt.elapse(10) and bt.position == 2  # Fill of the feed order behind us.
            assert bt.active_orders[0].maker and bt.state_values.balance == -16
            assert not bt.elapse(10) and bt.bid_qty_at_tick(8) == 1
            bt.clear_inactive_orders()
            assert bt.active_orders == []
    invalid = np.array([event(10, CANCEL | BUY, 8, 0, 999)], dtype=EVENT_DTYPE)
    with np.testing.assert_raises(ValueError):
        xh.L3Backtest(cfg, invalid, 0, 0).elapse(0)


def test_recorder():
    bt = xh.BacktestRiskAdverse(xh.AssetConfig(1, 1, 0, 20, contract_size=4), quotes(), 0, 0)
    rec = xh.Recorder(capacity=2)
    assert rec.size() == 0 and math.isnan(rec.summary().final_equity)
    assert bt.elapse(0) and rec.record(10, 12, bt.state_values)
    assert bt.submit_buy_order(1, 16, 2, xh.TimeInForce.IOC) and bt.elapse(1)
    assert rec.record(11, 12, bt.state_values, contract_size=4)
    assert not rec.record(12, 16, bt.state_values) and rec.size() == 2
    stats = rec.summary()
    assert isinstance(stats, xh.SummaryStats)
    assert (stats.total_return, stats.max_drawdown, stats.final_equity) == (-32, 32, -32)
    assert (stats.num_trades, stats.total_volume, stats.total_value, stats.total_fee) == (1, 2, 128, 0)
    with TemporaryDirectory() as temp:
        path = Path(temp) / "samples.csv"
        assert rec.write_csv(path)
        with path.open(newline="") as stream:
            first, last = csv.DictReader(stream)
        assert (int(first["timestamp"]), float(first["equity"])) == (10, 0)
        assert (int(last["timestamp"]), float(last["equity"]), float(last["balance"])) == (11, -32, -128)


def quote(bt, asset_args, orders, order_id, target, qty, tick):
    for order in orders:
        if order.order_id == order_id:
            if order.req == 0 and order.status == xh.OrderStatus.NEW and order.price_tick != target:
                if not bt.cancel(*asset_args, order_id):
                    raise AssertionError(f"cancel failed for {asset_args} order {order_id}")
            return
    if target is not None:
        submit = bt.submit_buy_order if order_id == 1 else bt.submit_sell_order
        if not submit(*asset_args, order_id, target * tick, qty, GTX):
            raise AssertionError(f"submit failed for {asset_args} order {order_id}")


def replay_single(bt, tick):
    import run_official_hftbt as single

    while bt.elapse(single.INTERVAL_NS):
        bid, ask = bt.best_bid, bt.best_ask
        if not 0 < bid < ask:
            continue
        pos = bt.position
        mid_ticks = (0.5 * (bid + ask)) / tick
        half_spread, skew = mid_ticks * 0.0015, pos * single.SKEW_TICKS
        buy = math.floor(mid_ticks - half_spread - skew) if pos + single.QTY <= single.CAP else None
        sell = math.ceil(mid_ticks + half_spread - skew) if pos - single.QTY >= -single.CAP else None
        bt.clear_inactive_orders()
        orders = bt.active_orders
        quote(bt, (), orders, 1, buy, single.QTY, tick)
        quote(bt, (), orders, 2, sell, single.QTY, tick)
    state = bt.state_values
    mid = 0.5 * (bt.best_bid + bt.best_ask)
    equity = state.balance + state.position * mid - state.fee
    return dict(zip(single.STATE_FIELDS, (state.position, state.balance, state.fee, state.num_trades,
                                          state.trading_volume, equity, mid), strict=True))


def test_single_asset():
    if not all((DATA / name).is_file() for name in ("lista_real.bin", "lista_real.npz", "lista_real_meta.json")):
        raise SkipTest("single-asset fixtures are missing from DATA_DIR")
    import hftbacktest as hft
    import run_official_hftbt as single

    baseline = json.loads((single.BENCH / "official_baseline.json").read_text())
    meta = json.loads((DATA / "lista_real_meta.json").read_text())
    with np.load(DATA / "lista_real.npz") as archive:
        rows = archive["data"]
    assert len(rows) == baseline["n_events"] == meta["n_events"] > 0
    assert rows.dtype == hft.event_dtype and baseline["hftbacktest_version"] == "2.4.4"
    tick = meta["tick_size"]
    cfg = xh.AssetConfig(tick, meta["lot_size"], round(meta["roi_lb"] / tick),
                         round(meta["roi_ub"] / tick), 1.0, -0.00003, 0.0005)
    for model, cls in (("risk_adverse", xh.BacktestRiskAdverse), ("power_prob", xh.BacktestPowerProb)):
        expected = baseline["models"][model]["final_state"]
        for label, source in (("mmap", DATA / "lista_real.bin"), ("array", rows)):
            bt = cls(cfg, source, LATENCY, LATENCY)
            start = perf_counter()
            actual = replay_single(bt, tick)
            seconds = perf_counter() - start
            assert actual.keys() == expected.keys()
            for field, ref in expected.items():
                value = actual[field]
                assert float(value).hex() == float(ref).hex(), (model, label, field, value, ref)
            print(f"LISTAUSDT {model} {label}  {len(rows) / seconds / 1e6:.3f} M events/s, "
                  f"{actual['num_trades']} trades, 7/7 fields bit-exact")


def test_multi_asset():
    paths = [DATA / "multi_asset_test" / f"{symbol}.npy" for symbol in ("LISTAUSDT", "CYSUSDT")]
    prepared = all(path.is_file() for path in paths)
    if not prepared and not (os.environ.get("RAW_DATA_DIR") and (DATA / "lista_real.npz").is_file()):
        raise SkipTest("two-asset fixtures are missing from DATA_DIR")
    import hftbacktest as hft
    import run_official_hftbt as single
    import test_multi_asset_parity as multi

    assert hft.__version__ == "2.4.4", f"Expected hftbacktest 2.4.4, found {hft.__version__}"
    if not prepared:
        multi.prepare(paths)
    data = [np.load(path, mmap_mode="r") for path in paths]
    assert all(len(rows) == multi.COUNT and rows.dtype == hft.event_dtype for rows in data)
    expected, _ = multi.official(data)
    bt = xh.MultiAssetRiskAdverse()
    for i, (path, rows) in enumerate(zip(paths, data, strict=True)):
        ticks = np.rint(rows["px"] / multi.TICK).astype(np.int64)
        cfg = xh.AssetConfig(multi.TICK, multi.LOT, int(ticks.min()) - 1, int(ticks.max()) + 1,
                             1.0, -0.00003, 0.0005)
        index = (bt.add_asset_file(cfg, path, LATENCY, LATENCY) if i == 0 else
                 bt.add_asset_array(cfg, rows, LATENCY, LATENCY))
        assert index == i
    assert bt.asset_count == len(paths)
    previous = steps = 0
    max_tick = np.iinfo(np.int64).max
    start = perf_counter()
    while bt.elapse(single.INTERVAL_NS):
        steps += 1
        b0, a0 = bt.best_bid_tick(0), bt.best_ask_tick(0)
        b1, a1 = bt.best_bid_tick(1), bt.best_ask_tick(1)
        if not (0 < b0 < a0 < max_tick and 0 < b1 < a1 < max_tick):
            continue
        mid2 = b0 + a0
        skew = (mid2 > previous) - (mid2 < previous) if previous else 0
        previous = mid2
        for i, bid, ask in ((0, b0, a0), (1, b1 + skew, a1 + skew)):
            bt.clear_inactive_orders(i)
            orders = bt.active_orders(i)
            quote(bt, (i,), orders, 1, bid, 1.0, multi.TICK)
            quote(bt, (i,), orders, 2, ask, 1.0, multi.TICK)
    seconds = perf_counter() - start
    assets = []
    for i in range(bt.asset_count):
        state = bt.state_values(i)
        mid = 0.5 * (bt.best_bid(i) + bt.best_ask(i))
        fields = dict(position=bt.position(i), balance=state.balance, fee=state.fee,
                      num_trades=state.num_trades, trading_volume=state.trading_volume,
                      trading_value=state.trading_value, final_mid=mid)
        fields["equity"] = state.balance + state.position * mid - state.fee
        assets.append(fields)
    multi.compare(expected, dict(steps=steps, current_ts=bt.current_timestamp,
                                 assets=assets, total_equity=bt.total_equity()))
    print(f"{' + '.join(multi.SYMBOLS)}  {sum(map(len, data)) / seconds / 1e6:.3f} M events/s, "
          f"{steps:,} steps, total equity {bt.total_equity()!r}, all fields bit-exact vs Rust 2.4.4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-contained", action="store_true", help="run only tests that use synthetic events")
    args = parser.parse_args()
    unit_tests = (test_array_layout, test_borrowed_array, test_partial_fills, test_inverse_modify_and_funding,
                  test_multi_modify_and_funding, test_l3_fifo, test_recorder)
    for test in unit_tests:
        test()
    print(f"{len(unit_tests)} self-contained Python tests passed")
    if not args.self_contained:
        for test in (test_single_asset, test_multi_asset):
            try:
                test()
            except SkipTest as reason:
                print(f"Skipped {test.__name__}: {reason}")
