#ifndef XHFTBACKTEST_L3_BOOK_HPP
#define XHFTBACKTEST_L3_BOOK_HPP

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace xhftbacktest {

struct L3Order {
    std::uint64_t order_id{}, priority{};
    std::int64_t price_tick{}, timestamp{};
    double qty{};
    std::uint32_t prev{}, next{}, strategy_prev{}, strategy_next{};
    std::int8_t side{};
    bool strategy{};
    std::uint32_t outside_slot{};
};

namespace detail {

// One word covers 64 ticks, its parent covers 4096, and further summaries
// cover wider ROIs. A successor/predecessor visits at most two words per tier.
class L3Bitmap {
    using Word = std::uint64_t;
    std::vector<Word> words_;
    std::array<std::size_t, 12> offset_{};
    unsigned height_{};

public:
    static constexpr std::size_t missing = std::numeric_limits<std::size_t>::max();
    explicit L3Bitmap(std::size_t n = 0) {
        std::size_t total = 0;
        while (n) {
            n = (n - 1) / 64 + 1;
            offset_[height_++] = total;
            total += n;
            if (n == 1) break;
        }
        words_.resize(total);
    }

    void mark(std::size_t i, bool present) noexcept {
        for (unsigned h = 0; h < height_; ++h, i /= 64) {
            Word& w = words_[offset_[h] + i / 64];
            const Word bit = Word{1} << (i % 64);
            const bool empty = !w;
            w = present ? w | bit : w & ~bit;
            if (empty == !w) break;
        }
    }
    bool any() const noexcept { return !words_.empty() && words_.back(); }
    void clear() noexcept { std::fill(words_.begin(), words_.end(), 0); }

    template<bool Up>
    std::size_t seek(std::size_t i) const noexcept {
        auto edge = [](Word w) { return Up ? std::countr_zero(w) : 63 - std::countl_zero(w); };
        for (unsigned h = 0; h < height_; ++h) {
            const Word bit = Word{1} << (i % 64);
            // Include the starting tick, then skip the exhausted child on ascent.
            const Word mask = Up ? (~(bit - 1) & ~(h ? bit : 0)) : ((bit - 1) | (h ? 0 : bit));
            Word bits = words_[offset_[h] + i / 64] & mask;
            i /= 64;
            if (bits) {
                for (;;) {
                    i = i * 64 + edge(bits);
                    if (!h) return i;
                    bits = words_[offset_[--h] + i];
                }
            }
        }
        return missing;
    }
};

} // namespace detail

// Construct with tick size, lot size and inclusive ROI bounds in ticks, or call init
// before use. Feed add/modify take prices; strategy orders and quantity lookups
// take ticks. MaxOrders bounds live feed and strategy nodes together. Insertion
// returns false on a duplicate in its ID namespace or a full pool. Modification
// and deletion return false for an unknown ID. Replay operations never allocate.
template<std::size_t MaxOrders = 65536>
class L3Book {
    using Tick = std::int64_t;
    using Id = std::uint64_t;
    using Index = std::uint32_t;  // zero terminates all intrusive lists
    static_assert(MaxOrders > 0 && MaxOrders < UINT32_MAX / 2);
    static constexpr auto slots = std::bit_ceil(2 * MaxOrders);

public:
    static constexpr Tick no_bid = std::numeric_limits<Tick>::min();
    static constexpr Tick no_ask = std::numeric_limits<Tick>::max();

private:
    struct Level {
        double qty{};
        Index head{}, tail{}, strategy_head{}, strategy_tail{};
        explicit operator bool() const noexcept { return head != 0; }
    };
    struct Side {
        std::vector<Level> levels;
        detail::L3Bitmap occupied, strategies;
        Tick best, farthest;
        bool negative_qty{};
        Side(std::size_t width, bool buy)
            : levels(width), occupied(width), strategies(width),
              best(buy ? no_bid : no_ask), farthest(buy ? no_ask : no_bid) {}
    };
    Tick lb_{}, ub_{};
    double tick_size_{}, lot_size_{};
    Side bids_{0, true}, asks_{0, false};
    std::vector<L3Order> nodes_;
    std::vector<Index> ids_;
    // Out-of-ROI orders still participate in FIFO and can later move into the ROI.
    std::vector<Level> outside_;
    std::vector<Index> outside_strategies_, outside_fills_;
    Index outside_count_{};
    Index unused_ = 1, free_{};
    Id priority_{};

    static std::size_t home(Id key, bool tag) noexcept {
        return ((key ^ (tag ? 0xd1b54a32d192ed03ULL : 0ULL)) * 0x9e3779b97f4a7c15ULL)
               >> (64 - std::bit_width(slots - 1));
    }
    static auto next_slot(std::size_t i) noexcept { return (i + 1) & (slots - 1); }
    template<class T, class Hash>
    static void backshift(std::vector<T>& table, std::size_t hole, Hash hash) noexcept {
        for (auto i = next_slot(hole); table[i]; i = next_slot(i)) {
            const auto start = hash(table[i]);
            if (((hole - start) & (slots - 1)) < ((i - start) & (slots - 1))) {
                table[hole] = table[i];
                hole = i;
            }
        }
        table[hole] = {};
    }
    auto id_slot(Id id, bool strategy) const noexcept {
        auto i = home(id, strategy);
        while (Index n = ids_[i]) {
            if (nodes_[n].order_id == id && nodes_[n].strategy == strategy) break;
            i = next_slot(i);
        }
        return i;
    }
    auto offset(Tick tick) const noexcept { return Id(tick) - Id(lb_); }
    Side& side(int s) noexcept { return s > 0 ? bids_ : asks_; }
    bool inside(Tick tick) const noexcept { return offset(tick) < bids_.levels.size(); }
    Level& level(int s, Tick tick) noexcept {
        if (inside(tick)) return side(s).levels[offset(tick)];
        auto i = home(Id(tick), s > 0);
        while (Index n = outside_[i].head) {
            if (nodes_[n].price_tick == tick && nodes_[n].side == s) break;
            i = next_slot(i);
        }
        return outside_[i];
    }
    // Strategy-only links skip all market orders when dispatching FIFO fills.
    template<bool Attach>
    void link(Index i) noexcept {
        auto& o = nodes_[i];
        auto& l = level(o.side, o.price_tick);
        auto update = [&](Index& head, Index& tail, Index L3Order::*prev, Index L3Order::*next) {
            if constexpr (Attach) {
                o.*prev = tail;
                o.*next = 0;
                if (tail) nodes_[tail].*next = i; else head = i;
                tail = i;
            } else {
                if (o.*prev) nodes_[o.*prev].*next = o.*next; else head = o.*next;
                if (o.*next) nodes_[o.*next].*prev = o.*prev; else tail = o.*prev;
            }
        };
        if constexpr (Attach) o.priority = ++priority_;
        update(l.head, l.tail, &L3Order::prev, &L3Order::next);
        if (o.strategy) {
            const bool had = l.strategy_head != 0;
            update(l.strategy_head, l.strategy_tail, &L3Order::strategy_prev, &L3Order::strategy_next);
            if (inside(o.price_tick)) {
                if (had != (l.strategy_head != 0)) side(o.side).strategies.mark(offset(o.price_tick), !had);
            } else if constexpr (Attach) {
                o.outside_slot = outside_count_;
                outside_strategies_[outside_count_++] = i;
            } else {
                const Index last = outside_strategies_[--outside_count_];
                outside_strategies_[o.outside_slot] = last;
                nodes_[last].outside_slot = o.outside_slot;
            }
        }
        if constexpr (!Attach) {
            if (!l.head && !inside(o.price_tick)) {
                backshift(outside_, static_cast<std::size_t>(&l - outside_.data()), [&](const Level& v) {
                    const auto& first = nodes_[v.head];
                    return home(Id(first.price_tick), first.side > 0);
                });
            }
        }
    }

    // The reference searches strictly beyond the clamped start tick, stopping at farthest.
    Tick beyond(int s, Tick start) noexcept {
        auto& d = side(s);
        const auto i = offset(std::clamp(start, lb_, ub_));
        auto found = detail::L3Bitmap::missing;
        if (s > 0 && i) found = d.occupied.template seek<false>(i - 1);
        if (s < 0 && i + 1 < d.levels.size()) found = d.occupied.template seek<true>(i + 1);
        if (found != detail::L3Bitmap::missing) {
            const Tick t = lb_ + static_cast<Tick>(found);
            if (s > 0 ? t >= d.farthest : t <= d.farthest) return t;
        }
        return s > 0 ? no_bid : no_ask;
    }
    void improve(int s, Tick tick) noexcept {
        auto& d = side(s);
        d.farthest = s > 0 ? std::min(d.farthest, tick) : std::max(d.farthest, tick);
        if (s > 0 ? tick > d.best : tick < d.best) {
            d.best = tick;
            if (bids_.best >= asks_.best) side(-s).best = beyond(-s, tick);
        }
    }
    void change_depth(const L3Order& o, double delta, bool remove) noexcept {
        if (!inside(o.price_tick)) return;
        auto& d = side(o.side);
        const auto i = offset(o.price_tick);
        double& qty = d.levels[i].qty;
        const bool was_positive = qty > 0;
        qty += delta;
        if (remove && !(std::abs(qty / lot_size_) >= 0.5)) {
            qty = 0;
            if (o.price_tick == d.best) {
                d.best = beyond(o.side, o.price_tick);
                if (d.best == (o.side > 0 ? no_bid : no_ask))
                    d.farthest = o.side > 0 ? no_ask : no_bid;
            }
        }
        if (qty < 0) d.negative_qty = true;
        if (was_positive != (qty > 0)) d.occupied.mark(i, qty > 0);
    }
    bool add(Id id, std::int8_t s, Tick tick, double qty, Tick ts, bool strategy) noexcept {
        const auto slot = id_slot(id, strategy);
        if (ids_[slot] || (!free_ && unused_ > MaxOrders)) return false;
        const Index i = free_ ? free_ : unused_++;
        if (free_) free_ = nodes_[i].next;
        ids_[slot] = i;
        nodes_[i] = {id, 0, tick, ts, qty, 0, 0, 0, 0, s, strategy};
        link<true>(i);
        if (!strategy) {
            change_depth(nodes_[i], qty, false);
            improve(s, tick);
        }
        return true;
    }
    bool erase(Id id, bool strategy) noexcept {
        const auto slot = id_slot(id, strategy);
        const Index i = ids_[slot];
        if (!i) return false;
        auto& o = nodes_[i];
        if (!strategy) change_depth(o, -o.qty, true);
        link<false>(i);
        backshift(ids_, slot, [&](Index n) { return home(nodes_[n].order_id, nodes_[n].strategy); });
        o.side = 0;
        o.next = free_;
        free_ = i;
        return true;
    }
    bool modify(Id id, Tick tick, double qty, Tick ts, bool strategy) noexcept {
        const Index i = ids_[id_slot(id, strategy)];
        if (!i) return false;
        auto& o = nodes_[i];
        const bool moved = tick != o.price_tick;
        const bool loses_priority = moved || qty > o.qty;
        if (loses_priority) link<false>(i);
        if (!strategy) change_depth(o, moved ? -o.qty : qty - o.qty, moved);
        o.price_tick = tick;
        o.qty = qty;
        if (moved) o.timestamp = ts;
        if (loses_priority) link<true>(i);
        if (moved && !strategy && inside(tick)) {
            change_depth(o, qty, false);
            improve(o.side, tick);
        }
        return true;
    }
    template<class F>
    void visit(const Level& l, Id through, F& fn) noexcept {
        for (Index i = l.strategy_head; i && nodes_[i].priority <= through; ) {
            const auto& o = nodes_[i];
            i = o.strategy_next;  // callback may unlink the current order
            fn(o.order_id);
        }
    }
    template<class F>
    void visit_outside(int s, Tick low, Tick high, F& fn) noexcept {
        Index count = 0;
        for (Index j = 0; j < outside_count_; ++j) {
            const Index i = outside_strategies_[j];
            const auto& o = nodes_[i];
            if (o.side == s && o.price_tick >= low && o.price_tick <= high)
                outside_fills_[count++] = i;
        }
        std::sort(outside_fills_.begin(), outside_fills_.begin() + count, [&](Index a, Index b) {
            const auto& x = nodes_[a];
            const auto& y = nodes_[b];
            return x.price_tick != y.price_tick ? x.price_tick < y.price_tick : x.priority < y.priority;
        });
        for (Index j = 0; j < count; ++j) fn(nodes_[outside_fills_[j]].order_id);
    }

public:
    L3Book() = default;
    L3Book(double tick_size, double lot_size, Tick lb, Tick ub) {
        if (!(tick_size > 0 && lot_size > 0 && std::isfinite(tick_size) && std::isfinite(lot_size))
            || lb > ub || lb == no_bid || ub == no_ask)
            throw std::invalid_argument("invalid L3Book configuration");
        const auto width = Id(ub) - Id(lb) + 1;
        bids_ = Side(width, true);
        asks_ = Side(width, false);
        nodes_.resize(MaxOrders + 1);
        ids_.assign(slots, 0);
        outside_.assign(slots, {});
        outside_strategies_.resize(MaxOrders);
        outside_fills_.resize(MaxOrders);
        lb_ = lb; ub_ = ub; tick_size_ = tick_size; lot_size_ = lot_size;
    }
    void init(double tick_size, double lot_size, Tick lb, Tick ub) {
        *this = L3Book(tick_size, lot_size, lb, ub);
    }
    Tick price_to_tick(double px) const noexcept {
        const double t = std::round(px / tick_size_);
        if (std::abs(t) < 0x1p63) return static_cast<Tick>(t);
        return t > 0 ? no_ask : (t < 0 ? no_bid : 0);  // Rust saturating f64 -> i64
    }
    double tick_to_price(Tick t) const noexcept { return static_cast<double>(t) * tick_size_; }
    Tick best_bid_tick() const noexcept { return bids_.best; }
    Tick best_ask_tick() const noexcept { return asks_.best; }
    double best_bid() const noexcept {
        return bids_.best == no_bid ? std::numeric_limits<double>::quiet_NaN() : tick_to_price(bids_.best);
    }
    double best_ask() const noexcept {
        return asks_.best == no_ask ? std::numeric_limits<double>::quiet_NaN() : tick_to_price(asks_.best);
    }
    double bid_qty_at_tick(Tick t) const noexcept {
        return inside(t) ? bids_.levels[offset(t)].qty : std::numeric_limits<double>::quiet_NaN();
    }
    double ask_qty_at_tick(Tick t) const noexcept {
        return inside(t) ? asks_.levels[offset(t)].qty : std::numeric_limits<double>::quiet_NaN();
    }
    const L3Order* find_order(Id id) const noexcept {
        const Index i = ids_[id_slot(id, false)];
        return i ? &nodes_[i] : nullptr;
    }
    bool add_buy_order(Id id, double px, double qty, Tick ts = 0) noexcept {
        return add(id, 1, price_to_tick(px), qty, ts, false);
    }
    bool add_sell_order(Id id, double px, double qty, Tick ts = 0) noexcept {
        return add(id, -1, price_to_tick(px), qty, ts, false);
    }
    bool delete_order(Id id, Tick = 0) noexcept { return erase(id, false); }
    bool modify_order(Id id, double px, double qty, Tick ts = 0) noexcept {
        return modify(id, price_to_tick(px), qty, ts, false);
    }
    bool add_strategy(Id id, std::int8_t s, Tick tick, double qty) noexcept {
        return add(id, s, tick, qty, 0, true);
    }
    bool delete_strategy(Id id) noexcept { return erase(id, true); }
    bool modify_strategy(Id id, Tick tick, double qty) noexcept { return modify(id, tick, qty, 0, true); }
    template<class F>
    void for_each_strategy_before(const L3Order& market, F&& fn) noexcept {
        visit(level(market.side, market.price_tick), market.priority, fn);
    }
    template<class F>
    void for_each_strategy_before(Id market_id, F&& fn) noexcept {
        if (const auto* o = find_order(market_id)) for_each_strategy_before(*o, fn);
    }
    template<class F>
    void for_each_strategy_at(int s, Tick tick, F&& fn) noexcept { visit(level(s, tick), UINT64_MAX, fn); }
    template<class F>
    void for_each_strategy_in_range(int s, Tick low, Tick high, F&& fn) noexcept {
        if (low < lb_) visit_outside(s, low, std::min(high, lb_ - 1), fn);
        auto& d = side(s);
        const Tick last = std::min(high, ub_);
        if (std::max(low, lb_) <= last) {
            for (auto i = offset(std::max(low, lb_)); i <= offset(last); ++i) {
                i = d.strategies.template seek<true>(i);
                if (i > offset(last)) break;
                visit(d.levels[i], UINT64_MAX, fn);
            }
        }
        if (high > ub_) visit_outside(s, std::max(low, ub_ + 1), high, fn);
    }
    void clear_orders(int s = 0) noexcept {
        for (Index i = 1; i < unused_; ++i) {
            auto& o = nodes_[i];
            if (o.side && (!s || s == o.side)) erase(o.order_id, o.strategy);
        }
        if (!s) { unused_ = 1; free_ = 0; }
        for (int k : {1, -1}) {
            if (s && s != k) continue;
            auto& d = side(k);
            // Sparse removal normally zeros each level. Extreme floating-point
            // cancellation can leave a residual even after its last order.
            if (d.negative_qty || d.occupied.any()) {
                std::fill(d.levels.begin(), d.levels.end(), Level{});
                d.occupied.clear();
                d.negative_qty = false;
            }
            d.best = k > 0 ? no_bid : no_ask; d.farthest = k > 0 ? no_ask : no_bid;
        }
    }
};

} // namespace xhftbacktest
#endif
