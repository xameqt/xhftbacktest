#ifndef XHFTBACKTEST_BOOK_HPP
#define XHFTBACKTEST_BOOK_HPP

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace xhftbacktest {

class FlatBook {
    using Tick = std::int64_t;
    using Word = std::uint64_t;
    using Index = std::size_t;
    static constexpr Index none = std::numeric_limits<Index>::max();

    // Each group covers 64 leaf words (4096 ticks). Nonempty groups are linked
    // in tick order so removing the best never scans across an empty price gap.
    struct Group {
        Word words = 0;
        Index prev = none, next = none;
    };

    struct Side {
        // calloc keeps untouched pages of a wide ROI shared with the OS zero page.
        Index width;
        std::unique_ptr<double[], decltype(&std::free)> qty;
        std::vector<Word> leaves;
        std::vector<Group> groups;
        Index first = none, last = none, best = none, farthest = none;

        explicit Side(Index n = 0)
            : width(n), qty(static_cast<double*>(std::calloc(n, sizeof(double))), &std::free),
              leaves((n + 63) / 64), groups((leaves.size() + 63) / 64) {
            if (n && !qty) throw std::bad_alloc{};
        }

        Side(const Side& s) : Side(s.width) {
            std::copy_n(s.qty.get(), width, qty.get());
            leaves = s.leaves;
            groups = s.groups;
            first = s.first; last = s.last; best = s.best; farthest = s.farthest;
        }
        Side(Side&&) noexcept = default;
        Side& operator=(Side&&) noexcept = default;
        Side& operator=(const Side& s) { return *this = Side(s); }

        void link(Index g) noexcept {
            Index next = none;
            if (last != none && g < last) {
                next = first;
                if (g > first) {
                    next = g + 1;
                    while (!groups[next].words) ++next;
                }
            }
            const Index prev = next == none ? last : groups[next].prev;
            groups[g].prev = prev;
            groups[g].next = next;
            if (prev == none) first = g; else groups[prev].next = g;
            if (next == none) last = g; else groups[next].prev = g;
        }

        void unlink(Index g) noexcept {
            const auto [words, prev, next] = groups[g];
            if (prev == none) first = next; else groups[prev].next = next;
            if (next == none) last = prev; else groups[next].prev = prev;
        }

        void store(Index i, double value) noexcept {
            const double previous = std::exchange(qty[i], value);
            if ((previous > 0) == (value > 0)) return;
            const Index w = i / 64, g = w / 64;
            const Word bit = Word{1} << (i % 64);
            const Word word_bit = Word{1} << (w % 64);
            if (value > 0) {
                if (!leaves[w]) {
                    if (!groups[g].words) link(g);
                    groups[g].words |= word_bit;
                }
                leaves[w] |= bit;
            } else {
                leaves[w] &= ~bit;
                if (!leaves[w]) {
                    groups[g].words &= ~word_bit;
                    if (!groups[g].words) unlink(g);
                }
            }
        }

        template<bool Bid>
        static Index edge(Word bits) noexcept {
            return Bid ? 63 - std::countl_zero(bits) : std::countr_zero(bits);
        }

        template<bool Bid>
        static Word beyond(Index i) noexcept {
            const Word bit = Word{1} << (i % 64);
            return Bid ? bit - 1 : ~(bit | (bit - 1));
        }

        // Strict predecessor (bid) or successor (ask). For a depleted best,
        // the starting group is still linked when this is called.
        template<bool Bid>
        Index next_level(Index i) const noexcept {
            Index w = i / 64, g = w / 64;
            if (Word bits = leaves[w] & beyond<Bid>(i))
                return w * 64 + edge<Bid>(bits);
            Word words = groups[g].words & beyond<Bid>(w);
            if (!words) {
                if (groups[g].words) {
                    g = Bid ? groups[g].prev : groups[g].next;
                } else {
                    // A crossing update can start between occupied groups.
                    g = Bid ? last : first;
                    while (g != none && (Bid ? g >= i / 4096 : g <= i / 4096))
                        g = Bid ? groups[g].prev : groups[g].next;
                }
                if (g == none) return none;
                words = groups[g].words;
            }
            w = g * 64 + edge<Bid>(words);
            return w * 64 + edge<Bid>(leaves[w]);
        }

        // Ordinary best-price searches stop at the farthest admitted whole-lot level.
        template<bool Bid>
        Index next_best(Index i) const noexcept {
            const Index candidate = next_level<Bid>(i);
            return farthest != none && (Bid ? candidate < farthest : candidate > farthest)
                ? none : candidate;
        }

        void reset() noexcept {
            std::fill_n(qty.get(), width, 0.0);
            std::fill(leaves.begin(), leaves.end(), 0);
            std::fill(groups.begin(), groups.end(), Group{});
            first = last = best = farthest = none;
        }
    };

    Side bids_, asks_;
    Tick base_ = 0;
    double tick_size_ = 0, half_lot_ = 0;

    // Unsigned subtraction also handles extreme out-of-ROI ticks.
    Word index(Tick t) const noexcept { return Word(t) - Word(base_); }

    template<bool Bid>
    void update(Side& side, Side& opposite, Tick tick, double qty) noexcept {
        const auto i = index(tick);
        if (i >= side.width) return;
        if (qty < half_lot_) {
            if (side.best == i) {
                side.best = side.next_best<Bid>(i);
                if (side.best == none) side.farthest = none;
            }
        } else {
            if (side.farthest == none || (Bid ? i < side.farthest : i > side.farthest))
                side.farthest = i;
            if (side.best == none || (Bid ? i > side.best : i < side.best)) {
                side.best = i;
                if (opposite.best != none && (Bid ? i >= opposite.best : i <= opposite.best))
                    opposite.best = opposite.next_best<!Bid>(i);
            }
        }
        side.store(i, qty);
    }

    // 2.4.4's ROI clear searches strictly beyond the tick adjacent to the cutoff.
    template<bool Bid>
    Index after_clear(const Side& side, Tick tick) const noexcept {
        if (side.first == none) return none;
        const auto start = static_cast<__int128_t>(tick) - base_ + (Bid ? -1 : 1);
        const auto bounded = std::clamp<__int128_t>(start, 0, side.width - 1);
        return side.next_best<Bid>(static_cast<Index>(bounded));
    }

    template<bool Bid>
    void clear(Side& side, Tick tick) noexcept {
        const auto limit = index(tick);
        if (limit >= side.width) return;
        while (side.best != none && (Bid ? side.best >= limit : side.best <= limit)) {
            const Index i = side.best;
            side.best = side.next_level<Bid>(i);
            side.store(i, 0.0);
        }
    }

public:
    static constexpr Tick no_bid = std::numeric_limits<Tick>::min();
    static constexpr Tick no_ask = std::numeric_limits<Tick>::max();

    void init(double tick_size, double lot_size, Tick roi_lb_tick, Tick roi_ub_tick) {
        if (!(tick_size > 0 && lot_size > 0 && std::isfinite(tick_size) && std::isfinite(lot_size))
            || roi_lb_tick > roi_ub_tick || roi_lb_tick == no_bid || roi_ub_tick == no_ask)
            throw std::invalid_argument("invalid FlatBook configuration");
        const Word width = Word(roi_ub_tick) - Word(roi_lb_tick) + 1;
        if (width > std::numeric_limits<std::ptrdiff_t>::max() / sizeof(double)) throw std::length_error("FlatBook ROI too wide");
        Side bids(width), asks(width);
        bids_ = std::move(bids);
        asks_ = std::move(asks);
        base_ = roi_lb_tick;
        tick_size_ = tick_size;
        half_lot_ = lot_size * 0.5;
    }

    void update_bid(Tick t, double qty) noexcept { update<true>(bids_, asks_, t, qty); }
    void update_ask(Tick t, double qty) noexcept { update<false>(asks_, bids_, t, qty); }

    // Clear from the current best toward the cutoff tick, inclusive.
    void clear_bid(Tick upto) noexcept { clear<true>(bids_, upto); }
    void clear_ask(Tick from) noexcept { clear<false>(asks_, from); }
    void clear_all() noexcept { bids_.reset(); asks_.reset(); }

    // Preserve the asymmetric clear offsets of the 2.4.4 ROI-vector implementation.
    void replay_clear_bid(Tick tick) noexcept {
        clear_bid(std::max(tick, base_));
        bids_.best = after_clear<true>(bids_, tick);
        if (bids_.best == none) bids_.farthest = none;
    }
    void replay_clear_ask(Tick tick) noexcept {
        const Tick upper = base_ + static_cast<Tick>(asks_.width - 1);
        if (tick >= upper) {
            const Index end = std::min(Word(tick) - Word(upper), asks_.width - 1);
            clear_ask(base_ + static_cast<Tick>(end));
        }
        asks_.best = after_clear<false>(asks_, tick);
        if (asks_.best == none) asks_.farthest = none;
    }

    Tick best_bid_tick() const noexcept {
        return bids_.best == none ? no_bid : base_ + static_cast<Tick>(bids_.best);
    }
    Tick best_ask_tick() const noexcept {
        return asks_.best == none ? no_ask : base_ + static_cast<Tick>(asks_.best);
    }
    double best_bid() const noexcept {
        return bids_.best == none ? std::numeric_limits<double>::quiet_NaN() : tick_to_price(best_bid_tick());
    }
    double best_ask() const noexcept {
        return asks_.best == none ? std::numeric_limits<double>::quiet_NaN() : tick_to_price(best_ask_tick());
    }
    double bid_qty_at_tick(Tick t) const noexcept {
        const auto i = index(t);
        return i < bids_.width ? bids_.qty[i] : std::numeric_limits<double>::quiet_NaN();
    }
    double ask_qty_at_tick(Tick t) const noexcept {
        const auto i = index(t);
        return i < asks_.width ? asks_.qty[i] : std::numeric_limits<double>::quiet_NaN();
    }
    Tick price_to_tick(double px) const noexcept { return static_cast<Tick>(std::round(px / tick_size_)); }
    double tick_to_price(Tick t) const noexcept { return static_cast<double>(t) * tick_size_; }
};

} // namespace xhftbacktest
#endif
