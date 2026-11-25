"""Run the nanobind parity suites with `make test-python`."""

import json
import math
import os
import sys
import weakref
from pathlib import Path
from time import perf_counter

import hftbacktest as hft
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "build"), str(ROOT / "bench")]
import xhftbacktest as xh
import run_official_hftbt as single
import test_multi_asset_parity as multi

LATENCY = 750_000
GTX = xh.TimeInForce.GTX


def test_borrowed_array():
    config = xh.AssetConfig(1, 1, 0, 20)
    for multi_asset in (False, True):
        rows = np.zeros(2, dtype=hft.event_dtype)
        rows["ev"] = hft.EXCH_EVENT | hft.LOCAL_EVENT | hft.BUY_EVENT | hft.DEPTH_EVENT
        rows["exch_ts"] = rows["local_ts"] = [10, 20]
        rows["px"], rows["qty"] = 5, 1
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
    rows = np.empty(4, dtype=hft.event_dtype)
    unaligned = np.ndarray(2, dtype=rows.dtype, buffer=bytearray(129), offset=1)
    for invalid in (rows[::2], rows.reshape(2, 2), rows.view("V64"),
                    rows.view(rows.dtype.newbyteorder("S")), unaligned):
        with np.testing.assert_raises((TypeError, ValueError)):
            xh.BacktestRiskAdverse(config, invalid, 0, 0)


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
    baseline = json.loads((single.BENCH / "official_baseline.json").read_text())
    data = Path(os.environ.get("DATA_DIR", ROOT / "bench" / "data"))
    meta = json.loads((data / "lista_real_meta.json").read_text())
    with np.load(data / "lista_real.npz") as archive:
        rows = archive["data"]
    assert len(rows) == baseline["n_events"] == meta["n_events"] > 0
    assert rows.dtype == hft.event_dtype and baseline["hftbacktest_version"] == "2.4.4"
    tick = meta["tick_size"]
    cfg = xh.AssetConfig(tick, meta["lot_size"], round(meta["roi_lb"] / tick),
                         round(meta["roi_ub"] / tick), 1.0, -0.00003, 0.0005)
    for model, cls in (("risk_adverse", xh.BacktestRiskAdverse), ("power_prob", xh.BacktestPowerProb)):
        expected = baseline["models"][model]["final_state"]
        for label, source in (("mmap", data / "lista_real.bin"), ("array", rows)):
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
    assert hft.__version__ == "2.4.4", f"Expected hftbacktest 2.4.4, found {hft.__version__}"
    paths = [multi.DATA / f"{symbol}.npy" for symbol in multi.SYMBOLS]
    if not all(path.is_file() for path in paths):
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
    test_array_layout()
    test_borrowed_array()
    test_single_asset()
    test_multi_asset()
