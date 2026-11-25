#include "xhftbacktest/l3_engine.hpp"

#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <new>
#include <stdexcept>
#include <string_view>
#include <vector>

// The benchmark links --wrap for the C allocation entry points. Route allocating
// C++ new overloads through them too, including calls from template instantiations.
static std::atomic<std::size_t> allocations;
static void count_allocation() noexcept { allocations.fetch_add(1, std::memory_order_relaxed); }
#ifdef XHFT_WRAP_MALLOC
extern "C" {
void* __real_malloc(std::size_t);
void* __real_calloc(std::size_t, std::size_t);
void* __real_realloc(void*, std::size_t);
void* __real_aligned_alloc(std::size_t, std::size_t);
int __real_posix_memalign(void**, std::size_t, std::size_t);
void* __wrap_malloc(std::size_t n) { count_allocation(); return __real_malloc(n); }
void* __wrap_calloc(std::size_t n, std::size_t s) { count_allocation(); return __real_calloc(n, s); }
void* __wrap_realloc(void* p, std::size_t n) { count_allocation(); return __real_realloc(p, n); }
void* __wrap_aligned_alloc(std::size_t a, std::size_t n) {
    count_allocation();
    return __real_aligned_alloc(a, n);
}
int __wrap_posix_memalign(void** p, std::size_t a, std::size_t n) {
    count_allocation();
    return __real_posix_memalign(p, a, n);
}
}
#endif
void* operator new(std::size_t n) {
#ifndef XHFT_WRAP_MALLOC
    count_allocation();
#endif
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc{};
}
void* operator new(std::size_t n, std::align_val_t a) {
#ifndef XHFT_WRAP_MALLOC
    count_allocation();
#endif
    void* p;
    if (::posix_memalign(&p, std::size_t(a), n ? n : 1) == 0) return p;
    throw std::bad_alloc{};
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void* operator new[](std::size_t n, std::align_val_t a) { return ::operator new(n, a); }
[[gnu::noinline]] static void release(void* p) noexcept { std::free(p); }
void operator delete(void* p) noexcept { release(p); }
void operator delete[](void* p) noexcept { release(p); }
void operator delete(void* p, std::size_t) noexcept { release(p); }
void operator delete[](void* p, std::size_t) noexcept { release(p); }
void operator delete(void* p, std::align_val_t) noexcept { release(p); }
void operator delete[](void* p, std::align_val_t) noexcept { release(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { release(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { release(p); }

namespace {
using namespace xhftbacktest;
using Replay = L3Backtest<ConstantLatency>;
using Word = std::uint64_t;
using Snapshot = std::array<Word, 25>;
constexpr AssetConfig config{0.25, 0.5, 0, 12287, 1., -0.00003, 0.0005};

struct Action { std::int64_t ts, op, id, tick, lots, tif; };
static_assert(sizeof(Action) == 48);

void check_pool() {
    L3Book<7> book;
    book.init(1., 1., 0, 262145);
    constexpr std::array ticks{0, 63, 64, 4095, 4096, 262143, 262144};
    const auto before = allocations.load(std::memory_order_relaxed);
    // These seven IDs all start in the last bucket of the 16-slot index.
    constexpr std::array<Word, 7> cluster{8, 21, 42, 55, 76, 97, 110};
    for (Word round = 0; round < 256; ++round) {
        const Word base = UINT64_MAX - 3 + 7 * round;
        const auto id = [&](Word i) { return round == 1 ? cluster[i] : base + i; };
        // Delete oldest first to force backward shifts through the wraparound.
        for (auto i = ticks.size(); i-- > 0; )
            if (!book.add_buy_order(id(i), ticks[i], 2.) ||
                book.add_sell_order(id(i), ticks[i], 1.))
                throw std::runtime_error("duplicate order or lost pool slot");
        if (book.add_buy_order(base + 7, 42., 1.) ||
            !book.modify_order(id(0), 63., 3.) || book.bid_qty_at_tick(63) != 5.)
            throw std::runtime_error("capacity or aggregate quantity mismatch");
        for (Word i = ticks.size() - 1; i > 0; --i) {
            if (book.best_bid_tick() != ticks[i] || !book.delete_order(id(i)) ||
                book.delete_order(id(i)))
                throw std::runtime_error("bitmap predecessor or hash deletion mismatch");
        }
        if (round == 255) book.clear_orders();
        else if (!book.delete_order(id(0))) throw std::runtime_error("lost last order");
        if (book.best_bid_tick() != FlatBook::no_bid || book.find_order(id(0)))
            throw std::runtime_error("nonempty book after last removal");
    }
    if (allocations.load(std::memory_order_relaxed) != before) throw std::runtime_error("allocation during book updates");
}

std::vector<Action> read_actions(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    const std::streamoff bytes = file.tellg();
    if (bytes <= 0 || bytes % sizeof(Action)) throw std::runtime_error("invalid action tape length");
    std::vector<Action> actions(static_cast<std::size_t>(bytes) / sizeof(Action));
    if (!file.seekg(0).read(reinterpret_cast<char*>(actions.data()), bytes))
        throw std::runtime_error("could not read action tape");
    return actions;
}

Word bits(double x) {
    return std::bit_cast<Word>(x);
}

Snapshot snapshot(const Replay& bt, const Action& a) {
    const auto& s = bt.state_values();
    const auto& d = bt.depth();
    const double mid = 0.5 * (d.best_bid() + d.best_ask());
    const auto* o = bt.orders().find(a.id);
    return {bits(s.position), bits(s.balance), bits(s.fee), Word(s.num_trades),
            bits(s.trading_volume), bits(s.trading_value), bits(s.balance + s.position * mid - s.fee),
            Word(d.best_bid_tick()), Word(d.best_ask_tick()), bits(d.best_bid()), bits(d.best_ask()),
            bits(d.bid_qty_at_tick(a.tick)), bits(d.ask_qty_at_tick(a.tick)), bt.orders().size(), o != nullptr,
            Word(o ? o->status : 0), Word(o ? o->req : 0), Word(o ? o->price_tick : 0),
            bits(o ? o->qty : 0.), bits(o ? o->leaves_qty : 0.), Word(o ? o->exec_price_tick : 0),
            bits(o ? o->exec_qty : 0.), Word(o ? o->exch_ts : 0), Word(o ? o->local_ts : 0), o && o->maker};
}

bool act(Replay& bt, const Action& a) {
    const double px = a.tick * config.tick_size, qty = a.lots * config.lot_size;
    const auto tif = static_cast<TimeInForce>(a.tif);
    switch (a.op) {
    case 0: return true;
    case 1: return bt.submit_buy_order(a.id, px, qty, tif);
    case 2: return bt.submit_sell_order(a.id, px, qty, tif);
    case 3: return bt.cancel(a.id);
    case 4: return bt.modify(a.id, px, qty);
    case 5: bt.clear_inactive_orders(); return true;
    default: return false;
    }
}
} // namespace

int main(int argc, char** argv) try {
    if (argc != 4) throw std::invalid_argument("usage: test_l3_parity FEED ACTIONS TRACE_OR_DASH");
    check_pool();
    const MmapLoader mapping(argv[1]);
    const auto actions = read_actions(argv[2]);
    const bool capture = std::string_view(argv[3]) != "-";
    std::vector<Snapshot> trace(capture ? actions.size() : 0);
    auto bt = std::make_unique<Replay>(mapping.events(), config, ConstantLatency{4, 7});
    const auto before = allocations.load(std::memory_order_relaxed);
    const auto start = std::chrono::steady_clock::now();
    if (bt->elapse(0) != 0) throw std::runtime_error("could not start replay");
    for (std::size_t i = 0; i < actions.size(); ++i) {
        const Action& a = actions[i];
        if (a.ts < bt->current_timestamp() ||
            bt->elapse(a.ts - bt->current_timestamp()) != int(i + 1 == actions.size()) || !act(*bt, a)) {
            std::fprintf(stderr, "replay failed on action %zu, op %lld\n", i, static_cast<long long>(a.op));
            return 1;
        }
        if (capture) trace[i] = snapshot(*bt, a);
    }
    const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const auto hot = allocations.load(std::memory_order_relaxed) - before;
    if (capture) {
        std::ofstream out(argv[3], std::ios::binary);
        out.write(reinterpret_cast<const char*>(trace.data()), trace.size() * sizeof(Snapshot));
        out.close();
        if (!out) throw std::runtime_error("could not write parity trace");
    }
    std::printf("{\"seconds\":%.12g,\"hot_allocations\":%zu,\"final\":[", seconds, hot);
    const auto final = snapshot(*bt, actions.back());
    for (std::size_t i = 0; i < final.size(); ++i)
        std::printf("%s%" PRIu64, i ? "," : "", final[i]);
    std::puts("]}");
    return hot != 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
}
