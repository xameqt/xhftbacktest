"""Time the official 2.4.4 ROI-vector replay on the prepared LISTAUSDT data."""

import argparse
import json
import math
import os
from pathlib import Path
from statistics import median
from time import perf_counter

import hftbacktest as hft
import numpy as np
from hftbacktest import GTX, LIMIT
from numba import njit

BENCH = Path(__file__).resolve().parent
DATA = Path(os.environ.get("DATA_DIR", BENCH / "data"))
INTERVAL_NS = 100_000_000
QTY, CAP = 100.0, 500.0
SKEW_TICKS = 0.01  # One tick per 100 LISTA held.
STATE_FIELDS = ("position", "balance", "fee", "num_trades", "trading_volume",
                "equity_at_last_mid", "final_mid")


@njit
def quote(bt, buy, target_tick, allowed, tick):
    order_id = 1 if buy else 2
    order = bt.orders(0).get(order_id)
    # Reuse each side's ID only after its cancel or fill has been acknowledged.
    status = 0
    if order is not None:
        if order.cancellable and (not allowed or order.price_tick != target_tick):
            status = bt.cancel(0, order_id, False)
    elif allowed:
        px = target_tick * tick
        if buy:
            status = bt.submit_buy_order(0, order_id, px, QTY, GTX, LIMIT, False)
        else:
            status = bt.submit_sell_order(0, order_id, px, QTY, GTX, LIMIT, False)
    if status != 0:
        raise RuntimeError("Official order request failed")


@njit
def replay(bt, tick):
    depth = bt.depth(0)
    while True:
        status = bt.elapse(INTERVAL_NS)
        if status == 1:
            break
        if status != 0:
            raise RuntimeError("Official replay failed")
        bid, ask = depth.best_bid, depth.best_ask
        if not (0 < bid < ask):
            continue
        pos = bt.position(0)
        mid_ticks = (0.5 * (bid + ask)) / tick
        half_spread = mid_ticks * 0.0015
        skew = pos * SKEW_TICKS
        bt.clear_inactive_orders(0)
        quote(bt, True, math.floor(mid_ticks - half_spread - skew), pos + QTY <= CAP, tick)
        quote(bt, False, math.ceil(mid_ticks + half_spread - skew), pos - QTY >= -CAP, tick)

    mid = 0.5 * (depth.best_bid + depth.best_ask)
    s = bt.state_values(0)
    equity = s.balance + s.position * mid - s.fee
    return s.position, s.balance, s.fee, s.num_trades, s.trading_volume, equity, mid


def run(data, meta, model):
    asset = (hft.BacktestAsset().data(data).linear_asset(1.0)
             .constant_order_latency(750_000, 750_000).trading_value_fee_model(-0.00003, 0.0005)
             .tick_size(meta["tick_size"]).lot_size(meta["lot_size"])
             .roi_lb(meta["roi_lb"]).roi_ub(meta["roi_ub"]).no_partial_fill_exchange())
    if model == "risk_adverse":
        asset.risk_adverse_queue_model()
    else:
        asset.power_prob_queue_model(3.0)
    bt = hft.ROIVectorMarketDepthBacktest([asset])
    try:
        start = perf_counter()
        state = replay(bt, meta["tick_size"])
        elapsed = perf_counter() - start
    finally:
        bt.close()
    return elapsed, dict(zip(STATE_FIELDS, state))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BENCH.parent / "build/official_baseline.json")
    output = parser.parse_args().output
    if hft.__version__ != "2.4.4":
        raise RuntimeError(f"Expected hftbacktest 2.4.4, found {hft.__version__}")
    meta = json.loads((DATA / "lista_real_meta.json").read_text())
    with np.load(DATA / "lista_real.npz") as archive:
        data = archive["data"]
    assert len(data) == meta["n_events"] and data.dtype == hft.event_dtype

    report = dict(hftbacktest_version=hft.__version__, n_events=len(data),
                  skew_ticks_per_unit=SKEW_TICKS, models={})
    run(data, meta, "risk_adverse")  # One shared Numba specialization, outside the samples.
    for model in ("risk_adverse", "power_prob"):
        samples = [run(data, meta, model) for _ in range(3)]
        times = [seconds for seconds, _ in samples]
        state = samples[0][1]
        assert all(s == state for _, s in samples) and state["num_trades"] > 0
        result = dict(elapsed_s=times, final_state=state)
        for label, seconds in (("best", min(times)), ("median", median(times))):
            result[label] = dict(elapsed_s=seconds, events_per_sec=len(data) / seconds,
                                 ns_per_event=seconds * 1e9 / len(data))
        report["models"][model] = result
        print(f"{model}: {state['num_trades']} trades, {result['median']['events_per_sec']:,.0f} events/s")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
