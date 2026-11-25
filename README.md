# xhftbacktest

Version **0.2.0** is a header-only C++20 engine for testing limit-order strategies on L2 depth feeds and L3 order-by-order feeds, with nanobind bindings for Python 3.12. A strategy sees its local book and receives fills after the configured response latency.

## Benchmark results

The reference for replay comparisons is Rust **hftbacktest v2.4.4**, driven by a Numba-compiled strategy. The single-asset comparisons use its ROI-vector book. The two-asset comparison uses its hash-map book and quotes both instruments on one clock.

| Recorded workload | Result | Measurement |
| --- | ---: | --- |
| LISTAUSDT, RiskAdverse queue | 2.58× Rust | 44.98 M/s C++ and 17.46 M/s Rust |
| LISTAUSDT, PowerProb queue (power 3) | 2.54× Rust | 43.84 M/s C++ and 17.23 M/s Rust |
| LISTAUSDT + CYSUSDT, shared clock | 3.57× Rust | Median throughput over 400,000 events |
| 20 C++ workers sharing one mmap | 251.33 M/s | Aggregate across 20 parameter configurations |
| Python strategy via nanobind | 12.3 M/s | Input events processed per second in a Python-driven replay |
| L3 FIFO replay | 12.64 M market events/s | 2.04× Rust v2.4.4, median of five runs. [L3 comparison](bench/test_l3_parity.py) |
| L2 PartialFill, delayed maker fills | 52.6 M fills/s | 16,384 fills per run, median of seven runs. [Fill benchmark](bench/test_v02_l2_parity.cpp) |
| Recorder, 100,000 calls to `record()` | 0 heap allocations | Measured after preallocating the recorder. [Allocation test](tests/test_asset_and_recorder.cpp) |

The LISTAUSDT queue-model rows are medians of three replays after warmup. Across 1,499,984 events, RiskAdverse produced 751 trades and PowerProb produced 757. For both models, all seven reported final values match Rust bit for bit, including fees and equity at the last mid. The two-asset test also checks each asset’s trading value, the strategy step count, final timestamp, and combined equity. Parity applies to these fixtures and fields. Timings depend on the host and workload. [Rust samples](bench/official_baseline.json) · [C++ samples and sweep](bench/cpp_results.json) · [Two-asset comparison](bench/test_multi_asset_parity.py) · [Python replay](python/test_python_parity.py)

## Architecture

L2 queue, latency, fill, and asset models are C++ template parameters. The engines allocate book storage and order mailboxes during setup. The measured native replay loops perform zero heap allocations. Each sweep worker owns its engine and borrows the same read-only event span. Python snapshots allocate outside the native loop.

| File | Lines | Responsibility |
| --- | ---: | --- |
| `event.hpp` | 54 | Native 64-byte event layout, flags, and timestamp fields |
| `book.hpp` | 264 | Tick-indexed quantities and occupancy bitmaps over a configured price range |
| `l3_book.hpp` | 418 | Feed order IDs, FIFO priority at each price, and aggregated depth |
| `mmap_loader.hpp` | 180 | Read-only POSIX mapping of raw event records and validated structured NumPy files |
| `order.hpp` | 116 | Order state, time in force, fill policies, and fixed-capacity order-ID lookup |
| `queue_models.hpp` | 84 | RiskAdverse, PowerProb, and LogProb queue advancement |
| `latency_models.hpp` | 54 | Constant latency and integer interpolation of timestamped latency samples |
| `asset_type.hpp` | 38 | Linear and inverse formulas for execution value, equity, and funding |
| `engine.hpp` | 548 | Single-asset L2 scheduling, matching, amendments, and settlement |
| `l3_engine.hpp` | 319 | Single-asset FIFO replay, strategy order priority, fills, and acknowledgments |
| `multi_asset.hpp` | 145 | Shared L2 clock, per-asset execution and funding, and portfolio equity |
| `recorder.hpp` | 136 | Preallocated state samples, equity statistics, and CSV output |
| `sweep.hpp` | 54 | Parallel replay of independent configurations, with results in input order |
| `python/xhftbacktest_py.cpp` | 262 | nanobind L2, L3, and Recorder APIs for Python |

## Quickstart

The native build requires GNU Make and a C++20 compiler on a POSIX system with mmap and 128-bit integer support. The Makefile is tested with GCC 16 on Linux. Set `LTO=-flto` for toolchains that use that spelling. The smoke test needs only the compiler and standard library.

```sh
make test       # Release and UBSan-trap, including allocation checks
```

Save this as `example.cpp`. The IOC buy crosses the ask at 101. With two nanoseconds of entry latency and three of response latency, the strategy sees its fill at time 5.

```cpp
#include <xhftbacktest/engine.hpp>
#include <iostream>

int main() {
    using namespace xhftbacktest;
    constexpr auto depth = EXCH_EVENT | LOCAL_EVENT | DEPTH_EVENT;
    constexpr Event tape[]{
        {depth | BUY_EVENT,   0,  0,  99, 2, 0, 0, 0},
        {depth | SELL_EVENT,  0,  0, 101, 2, 0, 0, 0},
        {depth | BUY_EVENT,  20, 20, 100, 1, 0, 0, 0},
    };
    AssetConfig asset{1.0, 1.0, 90, 110};  // tick size, lot size, ROI bounds in ticks
    Backtest<RiskAdverseQueueModel, ConstantLatency> bt(tape, asset, {2, 3});
    if (bt.elapse(0) != 0 || !bt.submit_buy_order(1, 101, 1, IOC)) return 1;
    bt.elapse(5);
    std::cout << bt.position() << '\n';  // 1
}
```

```sh
c++ -std=c++20 -O3 -ffp-contract=off -pthread -Iinclude example.cpp -o example
./example
```

For a file-backed replay, construct `MmapLoader mapping(path)` and pass `mapping.events()` in place of `tape`. Keep the mapping alive for the full replay. Single-asset C++ `elapse` returns 0 while work remains and 1 at EOF. L3 also returns 2 for an invalid feed order ID or an exhausted order pool. The multi-asset and Python APIs return a boolean that is true while work remains. Python L3 raises `ValueError` on a feed error.

### Python

`make python` builds `build/xhftbacktest<EXT_SUFFIX>` for the selected interpreter. Make first tries `PYTHON` (default `python3`) with nanobind and NumPy 2 or newer. If dependencies are missing and `uv` is available, Make uses a Python 3.12 environment.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install nanobind 'numpy>=2'
make python PYTHON=python
python -B python/test_python_parity.py --self-contained
```

### Parity and timing

`make test-l3` compares a generated FIFO tape against Rust. `make test-v02` checks L2 order amendments, partial fills, inverse accounting, and funding on synthetic cases and includes a LISTAUSDT case when available. Both targets exercise Release and UBSan-trap and use the `hftbacktest==2.4.4` reference package with Numba.

Set `DATA_DIR` to a prepared fixture directory (default `bench/data`). The single-asset targets use `lista_real.bin`, `lista_real.npz`, and `lista_real_meta.json`. The two-asset targets use `multi_asset_test/{LISTAUSDT,CYSUSDT}.npy`. Supply the recordings separately.

```sh
export DATA_DIR=/path/to/prepared-fixtures
make bench          # Release parity, timing, and 20-worker sweep
make check          # Same native checks under UBSan-trap
make test-multi     # Two-asset Release and UBSan-trap vs the Rust replay
make test-python    # Self-contained suite plus each available real-data parity suite
```

The single-asset harness checks that each C++ median beats the selected Rust median. The default baseline contains the published timings. To run that speed check on your hardware, time Rust locally and select its report. New timing reports go under `build/`.

```sh
uv run --no-project --python 3.12 --with hftbacktest==2.4.4 python bench/run_official_hftbt.py
make bench BASELINE=build/official_baseline.json
```

To prepare the fixtures from raw feeds, point `RAW_DATA_DIR` at an archive with `trades/2026/11/22/` and `incremental_book_L2/2026/11/22/`. Each directory supplies a Zstandard-compressed `<symbol>.csv.*` for LISTAUSDT and CYSUSDT. Preparation requires `zstdcat`. It starts from a snapshot, merges trades and depth by exchange timestamp, and advances out-of-order local timestamps to keep both timelines monotonic. The multi-asset harness prepares its two slices on first use.

```sh
export RAW_DATA_DIR=/path/to/raw-feeds
uv run --no-project --python 3.12 --with hftbacktest==2.4.4 python bench/prepare_real_data.py
make test-multi
```

## Scope and limitations

**L2** handles depth snapshots, updates, clears, and trades. `NoPartialFill` fills a crossing taker at the best opposite price for its full size. `PartialFill` can report several executions for one order as the maker queue advances or the taker walks displayed levels. Both replay historical market depth without consuming it. Limit orders support GTX, GTC, IOC, FOK, cancellation, and ModifyOrder. `apply_funding(rate, mark_price)` explicitly settles funding on the current position. A positive rate debits long equity and credits short equity. L2 supports linear and inverse contracts. Portfolio equity assumes a common settlement currency.

**L3** reconstructs a single instrument from order adds, modifications, cancellations, fills, and book clears. Strategy orders share FIFO priority with historical orders at each price. Published depth reflects the feed. L3 uses full fills and linear accounting. Set price bounds and order capacity before replay.

**Python 3.12** exposes the RiskAdverse and cubic PowerProb L2 queues for single- and multi-asset replay with constant latency. Single-asset variants include `BacktestPartialRiskAdverse`, `BacktestPartialPowerProb`, and `BacktestInverseRiskAdverse`. `L3Backtest` takes the same constructor arguments, including a file path or a contiguous NumPy `event_dtype` array borrowed for the engine’s lifetime. Treat the partial-fill tape as read-only after construction because the engine sizes its response mailbox from the supplied events. LogProb and interpolated latency are available in C++.

**Recorder** samples a timestamp, midprice, and engine state into preallocated storage. Python’s `record(ts, mid, state_values, contract_size=1.0)` uses the engine state’s linear or inverse valuation. Pass the instrument’s contract size when it differs from 1. A full recorder returns `False` and retains the earlier rows. `summary()` reports equity change and maximum drawdown in settlement units, annualized Sharpe and Sortino ratios, and cumulative execution totals. `write_csv(path)` exports the samples.

Released under the [MIT License](LICENSE).
