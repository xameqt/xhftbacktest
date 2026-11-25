#include "xhftbacktest/asset_type.hpp"
#include "xhftbacktest/engine.hpp"
#include "xhftbacktest/recorder.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <new>
#include <sstream>
#include <string>
#include <unistd.h>

namespace {
std::size_t allocations{};

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

void check(bool ok, const char* expr, int line) {
    if (!ok) {
        std::fprintf(stderr, "line %d failed: %s\n", line, expr);
        std::exit(EXIT_FAILURE);
    }
}
#define CHECK(expr) check((expr), #expr, __LINE__)

void test_asset_types() {
    // Binary64 fixtures distinguish the v2.4.4 multiplication orders for amount and equity.
    const LinearAsset linear{0.1};
    const InverseAsset inverse{0.1};
    CHECK(linear.amount(27.3, 0.7) == 0x1.e9374bc6a7efbp+0);
    CHECK(inverse.amount(27.3, 0.7) == 0x1.5015015015014p-9);
    CHECK(linear.equity(27.3, 0.125, 0.7, 0.0625) == 0x1.f9374bc6a7ef8p+0);
    CHECK(inverse.equity(27.3, 0.125, 0.7, 0.0625) == -0x1.8540540540540p-3);
    CHECK(linear.funding_cashflow(27.3, 0.7, 0.05) == -0x1.875f6fd21ff30p-4);
    CHECK(inverse.funding_cashflow(27.3, 0.7, 0.05) == -0x1.0cdd9aa677343p-13);
    CHECK(linear.funding_cashflow(27.3, -0.7, 0.05) == 0x1.875f6fd21ff30p-4);
    CHECK(inverse.funding_cashflow(27.3, 0.7, -0.05) == 0x1.0cdd9aa677343p-13);

    // hftbacktest==2.4.4: buy two size-10 contracts at 4, mark at 8, pay 0.25 in fees.
    const LinearAsset linear_fill{10};
    const InverseAsset inverse_fill{10};
    CHECK(linear_fill.amount(4, 2) == 80 && linear_fill.equity(8, -80, 2, 0.25) == 79.75);
    CHECK(inverse_fill.amount(4, 2) == 5 && inverse_fill.equity(8, -5, 2, 0.25) == 2.25);
    CHECK(inverse_fill.equity(4, -5, 2, 0) == 0);
    CHECK(inverse_fill.equity(8, 5, -2, 0.25) == -2.75);
}

void test_record_and_csv() {
    const std::int64_t ts = (std::int64_t{1} << 60) + 1;
    const InverseAsset asset{2};
    StateValues state{2, -9, 0.25, 7, std::nextafter(2.0, 3.0), 22.5};
    Recorder rec(1);
    CHECK(rec.record(ts, 4, state, asset));
    CHECK(!rec.record(ts + 1, 8, state, asset));
    CHECK(rec.size() == 1);
    const auto check_row = [&](const RecordRow& row) {
        CHECK(row.timestamp == ts && row.mid == 4 && row.equity == 7.75);
        CHECK(row.balance == state.balance && row.position == state.position && row.fee == state.fee);
        CHECK(row.num_trades == state.num_trades && row.trading_volume == state.trading_volume);
        CHECK(row.trading_value == state.trading_value);
    };
    check_row(rec[0]);

    auto path = (std::filesystem::temp_directory_path() / "xhft-recorder-XXXXXX").string();
    const int fd = ::mkstemp(path.data());
    CHECK(fd >= 0 && ::close(fd) == 0);
    CHECK(rec.write_csv(path));
    std::ifstream csv(path);
    CHECK(std::filesystem::remove(path));
    std::string line;
    CHECK(std::getline(csv, line) && line ==
          "timestamp,mid,balance,position,fee,num_trades,trading_volume,trading_value,equity");
    CHECK(std::getline(csv, line) && std::count(line.begin(), line.end(), ',') == 8);
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream fields(line);
    RecordRow row{};
    fields >> row.timestamp >> row.mid >> row.balance >> row.position >> row.fee
           >> row.num_trades >> row.trading_volume >> row.trading_value >> row.equity;
    CHECK(!fields.fail());
    check_row(row);
    CHECK(!std::getline(csv, line));
}

bool near(double actual, double expected) {
    return std::abs(actual - expected) <= 1e-12 * std::max(1.0, std::abs(expected));
}

void test_summary() {
    constexpr std::int64_t day = 86'400'000'000'000;
    constexpr std::array times{0 * day, day, day + day / 2, 2 * day, 4 * day};
    constexpr std::array equities{0., 25., 0., -25., 12.5};
    Recorder rec(times.size());
    const LinearAsset asset{1};
    StateValues state{};
    for (std::size_t i = 0; i < times.size(); ++i) {
        state.fee = 0.5 * i;
        state.balance = equities[i] + state.fee;
        state.num_trades = 2 * i;
        state.trading_volume = 1.5 * i;
        state.trading_value = 10 * i;
        CHECK(rec.record(times[i], 8, state, asset));
    }
    // Settlement-unit P&Ls are +25, -25, -25, +37.5. Equity falls from +25 to -25.
    const auto s = rec.summary();
    CHECK(s.total_return == 12.5 && s.max_drawdown == 50 && s.return_over_mdd == 0.25);
    CHECK(near(s.annualized_sharpe, 3.125 * std::sqrt(365.0 * 3 / 3242.1875)));
    CHECK(near(s.annualized_sortino, 3.125 * std::sqrt(365.0 * 4 / 1250)));
    CHECK(s.num_trades == 8 && s.total_volume == 6 && s.total_value == 40);
    CHECK(s.total_fee == 2 && s.final_equity == 12.5);

    Recorder flat(2);
    CHECK(flat.record(0, 8, state, asset));
    CHECK(flat.record(day, 8, state, asset));
    const auto f = flat.summary();
    CHECK(f.total_return == 0 && f.max_drawdown == 0 && std::isnan(f.return_over_mdd));
    CHECK(std::isnan(f.annualized_sharpe) && std::isnan(f.annualized_sortino));
}

void test_zero_allocations() {
    constexpr std::size_t count = 100'000;
    Recorder rec(count);
    const LinearAsset asset{1};
    const StateValues state{2, 40, 1, 8, 6, 10};
    static_assert(noexcept(rec.record(0, 1, state, asset)));
    const auto before = allocations;
    for (std::size_t i = 0; i < count; ++i) {
        CHECK(rec.record(i, i + 1, state, asset));
    }
    CHECK(allocations == before && rec.size() == count);
    CHECK(!rec.record(count, 1, state, asset) && allocations == before);
    for (std::size_t i = 0; i < count; ++i) {
        const auto row = rec[i];
        CHECK(row.timestamp == static_cast<std::int64_t>(i));
        CHECK(row.mid == i + 1 && row.equity == 41 + 2 * i);
    }
    Recorder empty(0);
    CHECK(!empty.record(0, 1, state, asset) && empty.size() == 0);
}

void test_sampling_edges() {
    constexpr auto end = std::numeric_limits<std::int64_t>::max();
    Recorder rec(4);
    const LinearAsset asset{1};
    StateValues state{};
    const auto empty = rec.summary();
    CHECK(std::isnan(empty.total_return) && std::isnan(empty.final_equity) && !empty.num_trades);
    CHECK(rec.record(end - 2, 1, state, asset));
    const auto single = rec.summary();
    CHECK(single.total_return == 0 && single.max_drawdown == 0 && std::isnan(single.annualized_sharpe));

    state.balance = -1;
    CHECK(rec.record(end - 1, 1, state, asset));
    state.balance = 2;
    CHECK(rec.record(end, 1, state, asset));
    // Adjacent nanoseconds near INT64_MAX round to the same double timestamp.
    const auto fine = rec.summary();
    constexpr double ns_per_year = 365.0 * 86'400'000'000'000.0;
    CHECK(near(fine.annualized_sharpe, std::sqrt(ns_per_year / 8)));
    CHECK(near(fine.annualized_sortino, std::sqrt(ns_per_year * 2)));
    CHECK(rec.record(end, 1, state, asset));
    const auto repeat = rec.summary();
    CHECK(std::isnan(repeat.annualized_sharpe) && std::isnan(repeat.annualized_sortino));
    CHECK(repeat.total_return == 2 && repeat.max_drawdown == 1);
}
}

int main() {
    test_asset_types();
    test_record_and_csv();
    test_summary();
    test_sampling_edges();
    test_zero_allocations();
    std::puts("Asset formulas, recorder statistics and CSV passed. 100,000 record() calls, 0 allocations.");
}
