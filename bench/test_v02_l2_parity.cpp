#include "xhftbacktest/multi_asset.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <new>
#include <string_view>
#include <vector>

namespace {
std::size_t allocations = 0;

void* allocate(std::size_t bytes, std::size_t alignment = 0) {
    ++allocations;
    bytes = std::max(bytes, std::size_t{1});
    void* ptr;
    if (alignment) {
        if (::posix_memalign(&ptr, alignment, bytes) == 0) return ptr;
    } else if ((ptr = std::malloc(bytes))) {
        return ptr;
    }
    throw std::bad_alloc{};
}
}

#ifdef XHFT_TEST_WRAP_ALLOC
extern "C" {
void* __real_malloc(std::size_t);
void* __real_calloc(std::size_t, std::size_t);
void* __real_realloc(void*, std::size_t);
void* __real_aligned_alloc(std::size_t, std::size_t);
int __real_posix_memalign(void**, std::size_t, std::size_t);
void* __wrap_malloc(std::size_t n) { ++allocations; return __real_malloc(n); }
void* __wrap_calloc(std::size_t n, std::size_t s) { ++allocations; return __real_calloc(n, s); }
void* __wrap_realloc(void* p, std::size_t n) { ++allocations; return __real_realloc(p, n); }
void* __wrap_aligned_alloc(std::size_t a, std::size_t n) {
    ++allocations;
    return __real_aligned_alloc(a, n);
}
int __wrap_posix_memalign(void** p, std::size_t a, std::size_t n) {
    ++allocations;
    return __real_posix_memalign(p, a, n);
}
}
#endif

void* operator new(std::size_t n) { return allocate(n); }
void* operator new[](std::size_t n) { return allocate(n); }
void* operator new(std::size_t n, std::align_val_t a) { return allocate(n, std::size_t(a)); }
void* operator new[](std::size_t n, std::align_val_t a) { return allocate(n, std::size_t(a)); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }

namespace {
using namespace xhftbacktest;

void require(bool ok, const char* message) {
    if (!ok) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(EXIT_FAILURE);
    }
}

void check_multi() {
    constexpr auto feed = EXCH_EVENT | LOCAL_EVENT;
    constexpr Event tape[] = {
        {feed | BUY_EVENT | DEPTH_EVENT, 10, 10, 127, 2, 0, 0, 0},
        {feed | SELL_EVENT | DEPTH_EVENT, 10, 10, 129, 2, 0, 0, 0},
        {EXCH_EVENT | SELL_EVENT | TRADE_EVENT, 20, 20, 128, 1, 0, 0, 0},
        {feed | BUY_EVENT | DEPTH_EVENT, 99, 99, 127, 2, 0, 0, 0},
    };
    MultiAssetBacktest<RiskAdverseQueueModel, ConstantLatency, 2, PartialFill, InverseAssetMode> bt;
    for (int i = 0; i < 2; ++i)
        bt.add_asset({1, 1, 120, 140}, tape, std::size(tape), {}, {1, 2});
    const auto before = allocations;
    require(bt.elapse(0) && bt.submit_buy_order(1, 7, 128, 4, GTX), "multi submit");
    require(bt.elapse(3) && bt.modify(1, 7, 128, 3), "multi modify");
    require(bt.wait_next_feed(true, 3) == 3 && bt.current_timestamp() == 16, "modify latency");
    require(bt.wait_next_feed(true, 30) == 3 && bt.current_timestamp() == 22, "partial fill latency");
    const auto* o = bt.orders(1).find(7);
    require(o && o->leaves_qty == 2 && o->status == status::PARTIALLY_FILLED &&
            o->q_ahead == 0 && bt.position(1) == 1, "multi partial fill");
    bt.apply_funding(1, 0.125, 128);
    require(bt.total_equity() == -1. / 1024 && bt.state_values(0).balance == 0, "multi funding");
    require(bt.cancel(1, 7) && bt.elapse(3) && o->status == status::CANCELED, "multi cancel");
    bt.clear_inactive_orders(1);
    require(bt.orders(1).size() == 0 && allocations == before, "multi cleanup or allocation");
}

void check_taker_sweeps() {
    constexpr auto feed = EXCH_EVENT | LOCAL_EVENT | DEPTH_EVENT;
    constexpr Event tape[] = {
        {feed | BUY_EVENT, 1, 1, 99, 1, 0, 0, 0}, {feed | BUY_EVENT, 1, 1, 98, 2, 0, 0, 0},
        {feed | SELL_EVENT, 1, 1, 101, 1, 0, 0, 0}, {feed | SELL_EVENT, 1, 1, 102, 2, 0, 0, 0},
        {feed | BUY_EVENT, 9, 9, 99, 1, 0, 0, 0},
    };
    for (const int side : {-1, 1}) for (const int qty : {2, 5}) for (auto tif : {GTC, GTX, FOK, IOC}) {
        Backtest<RiskAdverseQueueModel, ConstantLatency, 2, PartialFill> bt(
            tape, {1, 1, 90, 110, 1, 0, 0.125}, {0, 1});
        const int limit = 100 + 2 * side;
        const auto before = allocations;
        require(bt.elapse(0) == 0 && (side > 0 ? bt.submit_buy_order(1, limit, qty, tif)
                                               : bt.submit_sell_order(1, limit, qty, tif)), "taker submit");
        require(bt.elapse(1) == 0, "taker response");
        const int executed = tif == GTX || (tif == FOK && qty > 3) ? 0 :
                             tif == IOC ? std::min(qty, 3) : qty;
        // Rust's sell GTC visits the limit level, then fills residual size at the limit.
        const int trades = executed == 0 ? 0 : (side < 0 && tif == GTC && executed > 3 ? 3 : 2);
        const double value = executed ? 100 + side + (executed - 1) * limit : 0;
        const auto& s = bt.state_values();
        const auto* o = bt.orders().find(1);
        require(o && o->leaves_qty == qty - executed && o->status == (qty == executed ? status::FILLED : status::EXPIRED)
                && s.num_trades == trades && s.position == side * executed && s.trading_volume == executed
                && s.trading_value == value && s.balance == -side * value && s.fee == value * 0.125
                && allocations == before, "two-level taker accounting");
    }
}

void stress() {
    constexpr int fills = 16384, delay = 512;
    constexpr auto depth = EXCH_EVENT | LOCAL_EVENT | DEPTH_EVENT;
    std::vector<Event> tape{{depth | BUY_EVENT, 1, 1, 99, 1, 0, 0, 0},
                            {depth | SELL_EVENT, 1, 1, 101, 1, 0, 0, 0}};
    for (int i = 1; i <= fills; ++i)
        tape.push_back({EXCH_EVENT | SELL_EVENT | TRADE_EVENT, 1 + i, 1 + i, 100, 1, 0, 0, 0});
    tape.push_back({depth | BUY_EVENT, fills + delay + 2, fills + delay + 2, 99, 1, 0, 0, 0});
    using Clock = std::chrono::steady_clock;
    std::array<double, 7> seconds;
    for (auto& sample : seconds) {
        Backtest<RiskAdverseQueueModel, ConstantLatency, 2, PartialFill> bt(
            tape, {1, 1, 90, 110}, {0, delay});
        const auto before = allocations;
        const auto start = Clock::now();
        require(bt.elapse(0) == 0 && bt.submit_buy_order(1, 100, fills, GTX), "stress submit");
        // Hundreds of exchange fills arrive before the first local execution report.
        require(bt.elapse(delay) == 0 && bt.position() == 0, "premature execution");
        for (int count = 64; count <= fills; count += 64) {
            const auto result = bt.elapse(64);
            const auto* order = bt.orders().find(1);
            require(result == 0 && order && bt.position() == count && order->leaves_qty == fills - count &&
                    order->status == (count == fills ? status::FILLED : status::PARTIALLY_FILLED),
                    "delayed partial execution missing or early");
        }
        sample = std::chrono::duration<double>(Clock::now() - start).count();
        require(bt.state_values().num_trades == fills && bt.state_values().balance == -100. * fills &&
                allocations == before, "stress accounting or allocation");
    }
    std::sort(seconds.begin(), seconds.end());
    std::printf("%d delayed maker fills, 0 hot allocations, %.3f M fills/s (median of %zu)\n",
                fills, fills / seconds[seconds.size() / 2] / 1e6, seconds.size());
}

auto bits(double v) { return std::bit_cast<std::uint64_t>(v); }

void snapshot(const auto& bt, std::uint64_t id, double mark) {
    const auto& s = bt.state_values();
    const auto* o = bt.orders().find(id);
    std::cout << bt.current_timestamp() << ' ' << bits(s.position) << ' ' << bits(s.balance)
              << ' ' << bits(s.fee) << ' ' << s.num_trades << ' ' << bits(s.trading_volume)
              << ' ' << bits(s.trading_value) << ' ' << bits(bt.equity(mark)) << ' ' << (o != nullptr);
    if (o) std::cout << ' ' << o->order_id << ' ' << o->price_tick << ' ' << bits(o->qty)
                     << ' ' << bits(o->leaves_qty) << ' ' << bits(o->exec_qty) << ' ' << o->exec_price_tick
                     << ' ' << int(o->side) << ' ' << int(o->tif) << ' ' << int(o->status) << ' ' << int(o->req)
                     << ' ' << o->maker << ' ' << o->exch_ts << ' ' << o->local_ts;
    std::cout << '\n';
}

template<class... Fields>
void read(Fields&... fields) {
    require(static_cast<bool>((std::cin >> ... >> fields)), "malformed parity fixture");
}

template<class Fill, class Mode>
void replay(std::span<const Event> events, AssetConfig config, ConstantLatency latency) {
    Backtest<RiskAdverseQueueModel, ConstantLatency, 8, Fill, Mode> bt(events, config, latency);
    for (char op; std::cin >> op;) {
        auto before = allocations;
        auto args = [&](auto&... fields) { read(fields...); before = allocations; };
        std::uint64_t id;
        std::int64_t duration;
        double x, y;
        unsigned tif;
        switch (op) {
        case 'E': args(duration); bt.elapse(duration); break;
        case 'B': case 'S':
            args(id, x, y, tif);
            require(op == 'B' ? bt.submit_buy_order(id, x, y, TimeInForce(tif))
                              : bt.submit_sell_order(id, x, y, TimeInForce(tif)), "submit failed");
            break;
        case 'M': args(id, x, y); require(bt.modify(id, x, y), "modify failed"); break;
        case 'C': args(id); require(bt.cancel(id), "cancel failed"); break;
        case 'X': bt.clear_inactive_orders(); break;
        case 'F': args(x, y); bt.apply_funding(x, y); break;
        case 'P': read(id, x); snapshot(bt, id, x); continue;
        default: require(false, "unknown parity fixture action");
        }
        require(allocations == before, "heap allocation during replay");
    }
}
}

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--self-test") {
        check_multi();
        check_taker_sweeps();
        stress();
        return 0;
    }
    require(argc == 4, "expected FILL_MODEL ASSET_MODE EVENTS");
    const std::string_view fill = argv[1], mode = argv[2];
    require((fill == "partial" || fill == "full") && (mode == "inverse" || mode == "linear"),
            "unknown parity policy");
    const MmapLoader file(argv[3]);
    AssetConfig config{};
    ConstantLatency latency{};
    read(config.tick_size, config.lot_size, config.roi_lb_tick, config.roi_ub_tick, config.contract_size,
         config.maker_fee, config.taker_fee, latency.entry_ns, latency.resp_ns);
    if (fill == "partial" && mode == "linear") replay<PartialFill, LinearAssetMode>(file.events(), config, latency);
    if (fill == "partial" && mode == "inverse") replay<PartialFill, InverseAssetMode>(file.events(), config, latency);
    if (fill == "full" && mode == "linear") replay<NoPartialFill, LinearAssetMode>(file.events(), config, latency);
    if (fill == "full" && mode == "inverse") replay<NoPartialFill, InverseAssetMode>(file.events(), config, latency);
}
