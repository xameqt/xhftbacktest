#include "xhftbacktest/multi_asset.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <new>

namespace {
std::size_t allocations = 0;

void* allocate(std::size_t bytes, std::size_t alignment = 0) {
    ++allocations;
    bytes = std::max(bytes, std::size_t{1});
    void* p{};
    if (alignment) {
        if (::posix_memalign(&p, alignment, bytes) == 0) return p;
    } else if ((p = std::malloc(bytes))) {
        return p;
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

void check(bool ok, const char* expression, int line) {
    if (!ok) {
        std::fprintf(stderr, "smoke failed on line %d: %s\n", line, expression);
        std::exit(EXIT_FAILURE);
    }
}
#define CHECK(expr) check((expr), #expr, __LINE__)

constexpr auto bid = DEPTH_EVENT | BUY_EVENT;
constexpr auto ask = DEPTH_EVENT | SELL_EVENT;
constexpr auto sale = TRADE_EVENT | SELL_EVENT;
constexpr Event event(std::uint64_t flags, std::int64_t t, double px, double qty) {
    return {flags | LOCAL_EVENT | EXCH_EVENT, t, t, px, qty, 0, 0, 0};
}

void test_sparse_book() {
    FlatBook book;
    book.init(0.25, 1, 0, 12'000);
    const auto before = allocations;
    for (auto tick : {1, 65, 10'000}) book.update_bid(tick, 2);
    for (auto tick : {11'000, 10'100}) book.update_ask(tick, 3);
    CHECK(book.best_bid() == 2'500 && book.best_ask() == 2'525);
    book.update_bid(10'000, 0);
    CHECK(book.best_bid_tick() == 65 && book.bid_qty_at_tick(10'000) == 0);
    book.clear_bid(60);
    book.clear_ask(10'100);
    CHECK(book.best_bid_tick() == 1 && book.best_ask_tick() == 11'000);
    book.clear_all();
    CHECK(book.best_bid_tick() == FlatBook::no_bid && book.best_ask_tick() == FlatBook::no_ask);
    CHECK(allocations == before);
}

void test_orders() {
    constexpr Event rows[]{event(bid, 100, 10, 4), event(ask, 100, 12, 1), event(sale, 110, 10, 5)};
    Backtest<RiskAdverseQueueModel, ConstantLatency, 8> bt(rows, {1, 1, 1, 20, 2, -0.125, 0.25}, {2, 2});
    const auto before = allocations;
    CHECK(bt.elapse(0) == 0);
    CHECK(bt.submit_buy_order(1, 12, 1, GTX));
    CHECK(bt.submit_buy_order(2, 12, 1, GTC));
    CHECK(bt.submit_buy_order(3, 12, 3, IOC));
    CHECK(bt.submit_sell_order(4, 10, 2, FOK));
    CHECK(bt.submit_buy_order(5, 10, 1, IOC));
    CHECK(bt.submit_sell_order(6, 12, 1, FOK));
    CHECK(bt.submit_buy_order(7, 10, 1, GTX));
    CHECK(!bt.submit_buy_order(7, 10, 1, GTX));
    CHECK(bt.submit_sell_order(8, 13, 1, GTC));
    CHECK(bt.elapse(3) == 0 && bt.position() == 0);
    CHECK(bt.elapse(1) == 0 && bt.position() == 2);
    auto has_status = [&](unsigned id, status::Code code) {
        const auto* o = bt.orders().find(id);
        return o && o->status == code;
    };
    for (auto id : {1, 5, 6}) CHECK(has_status(id, status::EXPIRED));
    for (auto id : {2, 3, 4}) CHECK(has_status(id, status::FILLED));
    CHECK(has_status(7, status::NEW) && has_status(8, status::NEW));
    CHECK(bt.cancel(8) && !bt.cancel(8));
    CHECK(bt.elapse(4) == 0 && has_status(8, status::CANCELED));
    CHECK(bt.elapse(4) == 1 && has_status(7, status::FILLED) && bt.orders().find(7)->maker);
    const auto& s = bt.state_values();
    CHECK(s.position == 3 && s.num_trades == 4 && s.balance == -76 && s.fee == 31.5);
    CHECK(s.trading_value == 156 && s.trading_volume == 7);
    bt.clear_inactive_orders();
    CHECK(bt.orders().size() == 0 && allocations == before);
}

template<class Model>
void test_queue(int fill_time) {
    constexpr Event rows[]{event(bid, 0, 10, 4), event(ask, 0, 12, 2), event(bid, 2, 10, 10),
                           event(bid, 3, 10, 5), event(sale, 4, 10, 2.5), event(sale, 5, 10, 1),
                           event(sale, 6, 10, 2)};
    Backtest<Model, ConstantLatency, 4> bt(rows, {1, 1, 0, 20}, {0, 0});
    const auto before = allocations;
    CHECK(bt.elapse(0) == 0 && bt.submit_buy_order(1, 10, 1, GTX));
    // The cancellation at time 3 advances each model's queue position differently.
    for (int t = 1; t <= 6; ++t) {
        CHECK(bt.elapse(1) == (t == 6));
        CHECK(bt.position() == (t >= fill_time));
    }
    CHECK(allocations == before);
}

void test_latency() {
    constexpr std::int64_t base = 1LL << 60;
    constexpr std::int64_t times[]{base, base + 10, base + 20};
    constexpr std::int64_t entries[]{1, 6, 16}, responses[]{2, 8, 18};
    const IntpOrderLatency latency(times, entries, responses);
    CHECK(latency.entry(base - 1) == 1 && latency.entry(base + 5) == 3);
    CHECK(latency.response(base + 5) == 5 && latency.entry(base + 15) == 11);
    CHECK(latency.response(base + 21) == 18);

    constexpr Event rows[]{event(ask, base + 5, 12, 1), event(ask, base + 30, 13, 1)};
    Backtest<RiskAdverseQueueModel, IntpOrderLatency, 4> bt(rows, {1, 1, 0, 20}, {times, entries, responses});
    const auto before = allocations;
    CHECK(bt.elapse(0) == 0 && bt.submit_buy_order(1, 12, 1, IOC));
    // Entry at base+8, followed by six nanoseconds of response latency.
    CHECK(bt.elapse(8) == 0 && bt.position() == 0);
    CHECK(bt.elapse(1) == 0 && bt.position() == 1);
    CHECK(allocations == before);
}

void test_multi() {
    constexpr Event late[]{event(bid, 110, 6, 2), event(ask, 110, 8, 2), event(bid, 140, 6, 1)};
    constexpr Event early[]{event(bid, 100, 20, 1), event(ask, 100, 24, 3)};
    MultiAssetBacktest<RiskAdverseQueueModel, ConstantLatency, 4> bt;
    const AssetConfig config{1, 1, 0, 30};
    CHECK(bt.add_asset(config, late, std::size(late), {}, {1, 2}) == 0);
    CHECK(bt.add_asset(config, early, std::size(early), {}, {1, 2}) == 1);
    const auto before = allocations;
    CHECK(bt.elapse(0) && bt.current_timestamp() == 100);
    CHECK(bt.depth(0).best_ask_tick() == FlatBook::no_ask && bt.depth(1).best_ask() == 24);
    CHECK(bt.wait_next_feed(true, 20) == 2 && bt.current_timestamp() == 110);
    CHECK(bt.submit_buy_order(0, 1, 8, 1, IOC) && bt.submit_sell_order(1, 1, 20, 1, IOC));
    CHECK(bt.wait_next_feed(true, 2) == 0 && bt.position(0) == 0 && bt.position(1) == 0);
    CHECK(bt.wait_next_feed(true, 10) == 3 && bt.current_timestamp() == 113);
    CHECK(bt.position(0) == 1 && bt.position(1) == -1 && bt.total_equity() == -3);
    CHECK(bt.wait_next_feed(true, 100) == 1 && bt.depth(0).bid_qty_at_tick(6) == 1);
    CHECK(allocations == before);
}
}

int main() {
    test_sparse_book();
    test_orders();
    test_queue<RiskAdverseQueueModel>(6);
    test_queue<PowerProbQueueModel<3>>(5);
    test_queue<LogProbQueueModel>(4);
    test_latency();
    test_multi();
    std::puts("smoke passed (zero hot-loop allocations)");
}
