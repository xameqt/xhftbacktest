#include "xhftbacktest/engine.hpp"
#include "xhftbacktest/sweep.hpp"

#include <bit>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <new>
#include <sstream>

using namespace xhftbacktest;
using Clock = std::chrono::steady_clock;

namespace {
thread_local std::size_t allocations = 0;

void* allocate(std::size_t bytes, std::size_t alignment = 0) {
    ++allocations;
    void* p;
    if (alignment) {
        if (::posix_memalign(&p, alignment, std::max(bytes, std::size_t{1})) == 0) return p;
    } else if ((p = std::malloc(std::max(bytes, std::size_t{1})))) {
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
struct Reference {
    double position, balance, fee;
    std::int64_t trades;
    double volume, equity, mid, events_per_sec;
};
#include "baseline.inc"

struct QuoteConfig { double skew = 0.01, half_spread = 0.0015; };
struct Sample {
    StateValues state;
    double mid, equity, seconds;
    std::size_t hot_allocations;
};

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

// The LISTA strategy submits only GTX. Exercise the other limit order types and
// a cancel/fill race against the same exchange-before-request tie as v2.4.4.
void check_order_requests() {
    constexpr auto both = LOCAL_EVENT | EXCH_EVENT;
    constexpr Event tape[]{
        {both | BUY_EVENT | DEPTH_EVENT, 100, 100, 99, 2, 0, 0, 0},
        {both | SELL_EVENT | DEPTH_EVENT, 100, 100, 103, 2, 0, 0, 0},
        {EXCH_EVENT | SELL_EVENT | DEPTH_EVENT, 105, 0, 101, 2, 0, 0, 0},
        {LOCAL_EVENT | SELL_EVENT | DEPTH_EVENT, 0, 114, 101, 2, 0, 0, 0},
        {both | SELL_EVENT | TRADE_EVENT, 115, 118, 99, 2.5, 0, 0, 0},
    };
    Backtest<RiskAdverseQueueModel, ConstantLatency> bt(tape, {1, 1, 90, 110, 2, -0.01, 0.1}, {5, 5});
    require(bt.elapse(0) == 0, "first feed timestamp");
    require(bt.submit_buy_order(1, 101, 4, GTX) && bt.submit_buy_order(2, 101, 4, GTC)
         && bt.submit_sell_order(3, 99, 4, IOC) && bt.submit_buy_order(4, 101, 4, FOK)
         && bt.submit_sell_order(5, 102, 4, FOK) && bt.submit_buy_order(6, 99, 4, IOC)
         && bt.submit_buy_order(7, 99, 4, GTX), "time-in-force requests");
    require(bt.elapse(5) == 0 && bt.position() == 0 && bt.depth().best_ask() == 103, "local latency");
    require(bt.elapse(5) == 0 && bt.position() == 4, "taker fills");
    for (int id : {1, 5, 6}) require(bt.orders().find(id)->status == status::EXPIRED, "expiration");
    for (int id : {2, 3, 4}) require(bt.orders().find(id)->exec_qty == 4, "full taker quantity");
    require(bt.cancel(7) && !bt.cancel(7), "pending cancellation");
    require(bt.elapse(5) == 0 && bt.position() == 4 && bt.depth().best_ask() == 101, "in-flight fill");
    require(bt.elapse(5) == 1 && bt.current_timestamp() == 115, "end after final response");
    const auto& s = bt.state_values();
    require(bt.orders().find(7)->maker && s.position == 8 && s.num_trades == 4, "maker beat cancel");
    require(s.balance == -1616 && std::abs(s.fee - 232.88) < 1e-10, "linear execution value and fees");
    bt.clear_inactive_orders();
    require(bt.orders().size() == 0, "terminal order cleanup");
}

void quote(auto& bt, bool buy, std::int64_t target, bool allowed) {
    const auto id = buy ? 1u : 2u;
    if (const Order* o = bt.orders().find(id)) {
        if (o->status == status::NEW && o->req == req::NONE && (!allowed || o->price_tick != target))
            require(bt.cancel(id), "cancel failed");
    } else if (allowed) {
        const double px = target * tick;
        require(buy ? bt.submit_buy_order(id, px, 100., GTX)
                    : bt.submit_sell_order(id, px, 100., GTX), "submit failed");
    }
}

template<class Model>
Sample replay(std::span<const Event> events, const QuoteConfig& c = {}) {
    Backtest<Model, ConstantLatency> bt(events, asset, {750'000, 750'000});
    const auto before = allocations;
    const auto start = Clock::now();
    const auto& d = bt.depth();
    while (bt.elapse(100'000'000) == 0) {
        const double bid = d.best_bid(), ask = d.best_ask();
        if (!(0 < bid && bid < ask)) continue;
        const double pos = bt.position();
        const double mid_ticks = (0.5 * (bid + ask)) / tick;
        const double half = mid_ticks * c.half_spread, skew = pos * c.skew;
        bt.clear_inactive_orders();
        quote(bt, true, static_cast<std::int64_t>(std::floor(mid_ticks - half - skew)), pos + 100 <= 500);
        quote(bt, false, static_cast<std::int64_t>(std::ceil(mid_ticks + half - skew)), pos - 100 >= -500);
    }
    const double mid = 0.5 * (d.best_bid() + d.best_ask());
    const auto s = bt.state_values();
    const double equity = s.balance + s.position * mid - s.fee;
    const auto seconds = std::chrono::duration<double>(Clock::now() - start).count();
    Sample out{s, mid, equity, seconds, allocations - before};
    require(out.hot_allocations == 0, "allocation during replay");
    return out;
}

void check(const Sample& sample, const Reference& r) {
    const auto& s = sample.state;
    const auto same = [](double a, double b) {
        return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
    };
    require(s.num_trades == r.trades, "num_trades mismatch");
    require(same(s.position, r.position), "position mismatch");
    require(same(s.trading_volume, r.volume), "trading_volume mismatch");
    require(same(s.balance, r.balance), "balance mismatch");
    require(same(s.fee, r.fee), "fee mismatch");
    require(same(sample.equity, r.equity), "equity mismatch");
    require(same(sample.mid, r.mid), "final_mid mismatch");
}

void rate(std::ostream& json, double seconds) {
    json << "{\"elapsed_s\":" << seconds << ",\"events_per_sec\":" << event_count / seconds
         << ",\"ns_per_event\":" << seconds * 1e9 / event_count << "}";
}

template<class Model>
void measure(std::ostream& json, std::span<const Event> events, const char* name, const Reference& r) {
    std::array<double, 3> times{};
    Sample sample{};
    for (double& t : times) {
        sample = replay<Model>(events);
        check(sample, r);
        t = sample.seconds;
    }
    const auto& s = sample.state;
    json << "    \"" << name << "\": {\n      \"elapsed_s\":[" << times[0] << ',' << times[1] << ',' << times[2]
         << "],\n      \"final_state\":{\"position\":" << s.position << ",\"balance\":" << s.balance
         << ",\"fee\":" << s.fee << ",\"num_trades\":" << s.num_trades << ",\"trading_volume\":" << s.trading_volume
         << ",\"equity_at_last_mid\":" << sample.equity << ",\"final_mid\":" << sample.mid
         << "},\n      \"hot_allocations\":" << sample.hot_allocations << ",\n      \"best\":";
    std::sort(times.begin(), times.end());
    rate(json, times[0]);
    json << ",\n      \"median\":";
    rate(json, times[1]);
    const double speed = events.size() / times[1];
    json << ",\n      \"speedup_vs_rust_median\":" << speed / r.events_per_sec << "\n    }";
    std::cout << name << "  " << s.num_trades << " trades, " << speed / 1e6 << " M events/s, "
              << 1e9 / speed << " ns/event, " << speed / r.events_per_sec << "x Rust, "
              << sample.hot_allocations << " hot allocations\n";
    require(speed > r.events_per_sec, "median below official Rust baseline");
}
} // namespace

int main(int argc, char** argv) try {
    check_order_requests();
    require(argc <= 2, "usage: bench_release [RESULTS_JSON]");
    const char* data_dir = std::getenv("DATA_DIR");
    MmapLoader mapping(std::filesystem::path(data_dir ? data_dir : "bench/data") / "lista_real.bin");
    const auto events = mapping.events();
    require(events.size() == event_count, "dataset length mismatch");
    check(replay<RiskAdverseQueueModel>(events), refs[0]); // Same shared warmup as Python.
    std::ostringstream json;
    json << std::setprecision(17) << "{\n  \"n_events\":" << events.size() << ",\n  \"models\":{\n";
    measure<RiskAdverseQueueModel>(json, events, "risk_adverse", refs[0]);
    json << ",\n";
    measure<PowerProbQueueModel<3>>(json, events, "power_prob", refs[1]);

    std::array<QuoteConfig, 20> configs;
    for (std::size_t i = 0; i < configs.size(); ++i)
        configs[i] = {0.005 * (1 + i % 4), 0.001 + 0.00025 * (i / 4)};
    const auto start = Clock::now();
    const auto samples = sweep(events, std::span<const QuoteConfig>(configs), 20, replay<RiskAdverseQueueModel>);
    const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
    const double speed = events.size() * samples.size() / seconds;
    check(samples[9], refs[0]); // Reference configuration is also in the sweep.
    std::cout << "20-thread shared mmap sweep  " << speed / 1e6 << " M aggregate events/s\n";
    json << "\n  },\n  \"sweep\":{\"workers\":20,\"configs\":" << samples.size()
         << ",\"queue_model\":\"risk_adverse\",\"elapsed_s\":" << seconds
         << ",\"events_per_sec\":" << speed << ",\"hot_allocations\":0}\n}\n";
    std::ofstream out(argc > 1 ? argv[1] : "build/cpp_results.json");
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out << json.str();
    out.close();
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
