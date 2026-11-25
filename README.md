# xhftbacktest

A header-only C++20 engine for replaying L2 order books and testing limit-order strategies. Exchange events, local feed updates, order requests, and acknowledgments run on their own timelines. A strategy sees its local book and receives fills after the configured response latency.

## Benchmark results

The reference is the official Rust **hftbacktest v2.4.4**, driven by a Numba-compiled strategy. The single-asset comparison uses its ROI-vector book. The two-asset comparison uses its hash-map book and a strategy that quotes both instruments on one clock. Rates are in millions of input events per second.

| Recorded workload | Result | Measurement |
| --- | ---: | --- |
| LISTAUSDT, RiskAdverse queue | 2.58× Rust | 44.98 M/s C++ and 17.46 M/s Rust |
| LISTAUSDT, PowerProb queue (power 3) | 2.54× Rust | 43.84 M/s C++ and 17.23 M/s Rust |
| LISTAUSDT + CYSUSDT, shared clock | 3.57× Rust | Median throughput over 400,000 events |
| 20 C++ workers sharing one mmap | 251.33 M/s | Aggregate across 20 parameter configurations |
| Python strategy via nanobind | 12.3 M/s | Input events processed per second in a Python-driven replay |

The single-asset rows are medians of three replays after warmup. Across 1,499,984 events, RiskAdverse produced 751 trades and PowerProb produced 757. For both models, all seven reported final values match Rust bit for bit, including fees and equity at the last mid. The two-asset test also checks each asset’s trading value, the strategy step count, final timestamp, and combined equity. Parity applies to these fixtures and fields. Timings depend on the host and workload. [Rust samples](bench/official_baseline.json) · [C++ samples and sweep](bench/cpp_results.json) · [Two-asset comparison](bench/test_multi_asset_parity.py) · [Python replay](python/test_python_parity.py)

## Architecture

Queue and latency models are template parameters with zero virtual dispatch. Book storage is allocated at initialization, and orders and messages use fixed-capacity tables. After setup, the native replay loop performs zero heap allocations on the measured paths. Each sweep worker owns its engine and borrows the same read-only event span. Python order snapshots and other Python objects allocate outside the native loop.

| File | Lines | Responsibility |
| --- | ---: | --- |
| `event.hpp` | 54 | Native 64-byte event layout, flags, and timestamp fields |
| `book.hpp` | 264 | Tick-indexed quantities and occupancy bitmaps over a configured price range |
| `mmap_loader.hpp` | 180 | Read-only POSIX mapping of raw event records and validated structured NumPy files |
| `order.hpp` | 112 | Order state, time in force, and fixed-capacity order-ID lookup |
| `queue_models.hpp` | 84 | RiskAdverse, PowerProb, and LogProb queue advancement |
| `latency_models.hpp` | 54 | Constant latency and integer interpolation of timestamped latency samples |
| `engine.hpp` | 346 | Single-asset scheduling, matching, acknowledgments, and accounting |
| `multi_asset.hpp` | 133 | Shared strategy clock, per-asset execution, and portfolio equity |
| `sweep.hpp` | 54 | Parallel replay of independent configurations, with results in input order |
| `python/xhftbacktest_py.cpp` | 202 | nanobind single- and multi-asset APIs for mapped files and borrowed NumPy arrays |

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

For a file-backed replay, construct `MmapLoader mapping(path)` and pass `mapping.events()` in place of `tape`. Keep the mapping alive for the full replay. The single-asset C++ `elapse` returns 0 while work remains and 1 at end of replay. The multi-asset and Python APIs return a boolean that is true while work remains.

### Python

`make python` builds `build/xhftbacktest<EXT_SUFFIX>` for the selected interpreter. Make first tries `PYTHON` (default `python3`) with nanobind and NumPy 2 or newer. If dependencies are missing and `uv` is available, Make uses a Python 3.12 environment. The parity targets also require the official `hftbacktest==2.4.4` package with Numba.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install nanobind 'numpy>=2'
make python PYTHON=python
PYTHONPATH=build python -c 'import xhftbacktest as x; print(x.AssetConfig.__name__)'
```

### Real-data parity and timing

Set `DATA_DIR` to a prepared fixture directory (default `bench/data`). The single-asset targets use `lista_real.bin`, `lista_real.npz`, and `lista_real_meta.json`. The two-asset targets use `multi_asset_test/{LISTAUSDT,CYSUSDT}.npy`. Supply the recordings separately.

```sh
export DATA_DIR=/path/to/prepared-fixtures
make bench          # Release parity, timing, and 20-worker sweep
make check          # Same native checks under UBSan-trap
make test-multi     # Two-asset Release and UBSan-trap vs the Rust replay
make test-python    # Borrowed-array checks, then single- and two-asset parity
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

## Scope

The engine supports L2 depth snapshots, updates, clears, and trades. It models linear contracts (`LinearAsset`) and full fills (`NoPartialFillExchange`) with GTX, GTC, IOC, and FOK limit orders. A crossing taker order fills its full quantity at the best opposite price, even when displayed quantity is smaller. Price bounds and order capacity are set before replay. Multi-asset equity assumes a common quote currency.

ModifyOrder, funding payments, partial fills, and L3 order-by-order reconstruction are outside the current implementation. Python exposes RiskAdverse and cubic PowerProb with constant latency. LogProb and interpolated order latency are available in C++.

Released under the [MIT License](LICENSE).
