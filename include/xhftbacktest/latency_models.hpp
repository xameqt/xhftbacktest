#ifndef XHFTBACKTEST_LATENCY_MODELS_HPP
#define XHFTBACKTEST_LATENCY_MODELS_HPP

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>

namespace xhftbacktest {

struct ConstantLatency {
    std::int64_t entry_ns;
    std::int64_t resp_ns;

    constexpr std::int64_t entry(std::int64_t) const noexcept { return entry_ns; }
    constexpr std::int64_t response(std::int64_t) const noexcept { return resp_ns; }
};

// Samples must outlive the model. Entry and response share one chronological query stream.
struct IntpOrderLatency {
    std::span<const std::int64_t> ts, entry_ns, resp_ns;
    mutable std::size_t cursor = 0;

    IntpOrderLatency(std::span<const std::int64_t> timestamps,
                     std::span<const std::int64_t> entries,
                     std::span<const std::int64_t> responses) noexcept
        : ts(timestamps), entry_ns(entries), resp_ns(responses) {
        assert(!ts.empty() && ts.size() == entries.size() && ts.size() == responses.size());
        assert(std::adjacent_find(ts.begin(), ts.end(), std::greater_equal<>{}) == ts.end());
    }

    std::int64_t entry(std::int64_t t) const noexcept { return interpolate(t, entry_ns); }
    std::int64_t response(std::int64_t t) const noexcept { return interpolate(t, resp_ns); }

private:
    std::int64_t interpolate(std::int64_t t, std::span<const std::int64_t> values) const noexcept {
        while (cursor + 1 < ts.size() && t >= ts[cursor + 1]) ++cursor;
        if (cursor + 1 == ts.size() || t <= ts[cursor]) return values[cursor];

        // Subtract in the integer domain to preserve nanoseconds at epoch-scale timestamps.
        // Unsigned differences also cover intervals spanning the entire signed time range.
        const auto left = static_cast<std::uint64_t>(ts[cursor + 1]) - static_cast<std::uint64_t>(t);
        const auto right = static_cast<std::uint64_t>(t) - static_cast<std::uint64_t>(ts[cursor]);
        // GCC's wide integer keeps the weighted sum exact until truncation to nanoseconds.
        const auto weighted = static_cast<__int128_t>(left) * values[cursor]
                            + static_cast<__int128_t>(right) * values[cursor + 1];
        return static_cast<std::int64_t>(weighted / (left + right));
    }
};

}  // namespace xhftbacktest
#endif
