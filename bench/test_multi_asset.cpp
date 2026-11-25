#include "xhftbacktest/multi_asset.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <new>
#include <string>

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
using Replay = MultiAssetBacktest<RiskAdverseQueueModel, ConstantLatency>;

void require(bool ok, const char* message) {
    if (!ok) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(EXIT_FAILURE);
    }
}

void check_shared_clock() {
    constexpr auto depth = EXCH_EVENT | LOCAL_EVENT | DEPTH_EVENT;
    constexpr Event late[] = {
        {depth | BUY_EVENT, 150, 151, 10, 1, 0, 0, 0},
        {depth | SELL_EVENT, 150, 151, 12, 1, 0, 0, 0},
        {depth | BUY_EVENT, 250, 250, 11, 1, 0, 0, 0},
    };
    constexpr Event early[] = {
        {depth | BUY_EVENT, 100, 100, 19, 1, 0, 0, 0},
        {depth | SELL_EVENT, 100, 100, 21, 1, 0, 0, 0},
    };
    MultiAssetBacktest<RiskAdverseQueueModel, ConstantLatency, 8> bt;
    const AssetConfig cfg{1, 1, 0, 30, 2};
    require(bt.add_asset(cfg, late, std::size(late), {}, {2, 3}) == 0, "first asset number");
    require(bt.add_asset(cfg, early, std::size(early), {}, {2, 3}) == 1, "second asset number");
    require(bt.asset_count() == 2 && bt.elapse(0) && bt.current_timestamp() == 100,
            "earliest feed wins regardless of registration order");
    require(bt.depth(0).best_bid_tick() == FlatBook::no_bid && bt.depth(1).best_bid() == 19,
            "lookahead into the later asset");
    require(bt.elapse(40) && bt.submit_buy_order(1, 1, 21, 1, IOC), "submit after asset EOF");
    require(bt.elapse(4) && bt.position(1) == 0, "early response on exhausted asset");
    require(bt.elapse(1) && bt.position(1) == 1, "missing response on exhausted asset");
    require(bt.elapse(6) && bt.depth(0).best_ask() == 12, "late asset local feed");
    require(!bt.elapse(100) && bt.depth(0).best_bid() == 11, "did not drain last event");
    require(bt.total_equity() == -2, "contract-size-weighted total equity");
}

void check_wait(std::size_t buyer, bool include_responses, bool instant) {
    constexpr auto bid = EXCH_EVENT | LOCAL_EVENT | BUY_EVENT | DEPTH_EVENT;
    constexpr Event trade{EXCH_EVENT | SELL_EVENT | TRADE_EVENT, 33, 33, 98, 4, 0, 0, 0};
    constexpr Event a[] = {
        {bid, 10, 30, 100, 1, 0, 0, 0}, trade, {bid, 45, 60, 101, 1, 0, 0, 0},
        {bid, 100, 110, 102, 1, 0, 0, 0},
    };
    constexpr Event b[] = {
        {bid, 17, 27, 100, 1, 0, 0, 0}, {bid, 30, 33, 101, 1, 0, 0, 0},
        trade, {bid, 60, 61, 102, 1, 0, 0, 0},
    };
    MultiAssetBacktest<RiskAdverseQueueModel, ConstantLatency, 8> bt;
    const auto latency = instant ? ConstantLatency{0, 0} : ConstantLatency{1, 2};
    for (const auto& tape : {std::span(a), std::span(b)})
        bt.add_asset({1, 1, 0, 120}, tape.data(), tape.size(), {}, latency);
    const auto before = allocations;
    require(bt.current_timestamp() == std::numeric_limits<std::int64_t>::max(), "initial clock");
    require(bt.elapse(0) && bt.current_timestamp() == 10, "start at first exchange event");
    auto wait = [&](int timeout, int status, int ts) {
        require(bt.wait_next_feed(include_responses, timeout) == status && bt.current_timestamp() == ts,
                "wait status or timestamp differs from 2.4.4");
    };
    wait(3, 0, 13);
    wait(50, 2, 27);
    wait(50, 2, 30);
    require(bt.submit_buy_order(buyer, 1, 99, 1, GTX), "could not submit resting order");
    if (instant && include_responses) wait(50, 3, 30);
    // The later asset's local feed wins against the earlier asset's response at 33.
    // Within one asset the response wins against its own local feed at the same time.
    wait(50, include_responses && buyer == 1 ? 3 : 2, 33);
    const auto* order = bt.orders(buyer).find(1);
    require(bt.depth(1).best_bid_tick() == 101 && order &&
            order->status == (instant ? status::FILLED : status::NEW),
            "simultaneous local updates");
    if (!instant && include_responses) wait(50, 3, 35);
    wait(50, 2, 60);
    require(bt.position(buyer) == 1, "missing exchange-generated response");
    wait(50, 2, 61);
    wait(50, 1, 61);
    require(bt.depth(0).best_bid_tick() == 102, "EOF applies the last feed");
    require(allocations == before, "allocation while waiting for feed or order response");
}

void quote(Replay& bt, std::size_t i, bool buy, std::int64_t target) {
    const unsigned id = buy ? 1 : 2;
    if (const Order* o = bt.orders(i).find(id)) {
        if (o->status == status::NEW && o->req == req::NONE && o->price_tick != target)
            require(bt.cancel(i, id), "cancel failed");
    } else {
        const double price = bt.depth(i).tick_to_price(target);
        require(buy ? bt.submit_buy_order(i, id, price, 1, GTX)
                    : bt.submit_sell_order(i, id, price, 1, GTX), "submit failed");
    }
}
}

int main(int argc, char** argv) try {
    require(argc == 11, "expected FILE TICK LOT LOW_TICK HIGH_TICK for each of two assets");
    check_shared_clock();
    for (std::size_t buyer : {0u, 1u})
        for (bool responses : {false, true})
            for (bool instant : {false, true}) check_wait(buyer, responses, instant);
    std::array<MmapLoader, 2> mappings;
    Replay bt;
    for (std::size_t i = 0; i < mappings.size(); ++i) {
        const auto args = argv + 1 + 5 * i;
        mappings[i] = MmapLoader(args[0]);
        const auto events = mappings[i].events();
        AssetConfig cfg{std::stod(args[1]), std::stod(args[2]), std::stoll(args[3]),
                        std::stoll(args[4]), 1.0, -0.00003, 0.0005};
        require(bt.add_asset(cfg, events.data(), events.size(), {}, {750'000, 750'000}) == i,
                "unexpected asset number");
    }

    using Clock = std::chrono::steady_clock;
    std::size_t steps = 0;
    std::int64_t previous_mid_sum = 0;
    const auto before = allocations;
    const auto start = Clock::now();
    while (bt.elapse(100'000'000)) {
        ++steps;
        const auto b0 = bt.depth(0).best_bid_tick(), a0 = bt.depth(0).best_ask_tick();
        const auto b1 = bt.depth(1).best_bid_tick(), a1 = bt.depth(1).best_ask_tick();
        if (!(0 < b0 && b0 < a0 && a0 < FlatBook::no_ask &&
              0 < b1 && b1 < a1 && a1 < FlatBook::no_ask)) continue;
        const auto mid_sum = b0 + a0;
        const int signal = previous_mid_sum == 0 ? 0 :
            (mid_sum > previous_mid_sum) - (mid_sum < previous_mid_sum);
        previous_mid_sum = mid_sum;
        for (std::size_t i = 0; i < bt.asset_count(); ++i) {
            const int skew = i == 1 ? signal : 0;
            bt.clear_inactive_orders(i);
            quote(bt, i, true, bt.depth(i).best_bid_tick() + skew);
            quote(bt, i, false, bt.depth(i).best_ask_tick() + skew);
        }
    }
    const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
    const auto hot_allocations = allocations - before;
    require(hot_allocations == 0, "allocation during replay");
    std::cout << std::setprecision(17) << "{\"assets\":[";
    for (std::size_t i = 0; i < bt.asset_count(); ++i) {
        const auto& s = bt.state_values(i);
        const auto& d = bt.depth(i);
        const double mid = 0.5 * (d.best_bid() + d.best_ask());
        if (i) std::cout << ',';
        std::cout << "{\"position\":" << s.position << ",\"balance\":" << s.balance
                  << ",\"fee\":" << s.fee << ",\"num_trades\":" << s.num_trades
                  << ",\"trading_volume\":" << s.trading_volume << ",\"trading_value\":" << s.trading_value
                  << ",\"equity\":" << s.balance + s.position * mid - s.fee << ",\"final_mid\":" << mid << '}';
    }
    std::cout << "],\"steps\":" << steps << ",\"current_ts\":" << bt.current_timestamp()
              << ",\"total_equity\":" << bt.total_equity() << ",\"seconds\":" << seconds
              << ",\"hot_allocations\":" << hot_allocations << "}\n";
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
