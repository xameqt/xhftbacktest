#ifndef XHFTBACKTEST_ENGINE_HPP
#define XHFTBACKTEST_ENGINE_HPP

#include "book.hpp"
#include "latency_models.hpp"
#include "mmap_loader.hpp"
#include "queue_models.hpp"

namespace xhftbacktest {

struct AssetConfig {
    double tick_size, lot_size;
    std::int64_t roi_lb_tick, roi_ub_tick;
    double contract_size = 1.0, maker_fee = 0.0, taker_fee = 0.0;
};

struct StateValues {
    double position{}, balance{}, fee{};
    std::int64_t num_trades{};
    double trading_volume{}, trading_value{};
};

namespace detail {

inline constexpr auto never = std::numeric_limits<std::int64_t>::max();

// Each timeline is ordered on its own timestamp. A row may belong to either or both.
template<std::uint64_t Flag>
struct Feed {
    std::span<const Event> rows;

    void skip_unseen() noexcept {
        while (!rows.empty() && !(rows.front().ev & Flag)) rows = rows.subspan(1);
    }
    void advance() noexcept {
        rows = rows.subspan(1);
        skip_unseen();
    }
    std::int64_t next_time() const noexcept {
        if (rows.empty()) return never;
        if constexpr (Flag == EXCH_EVENT) return rows.front().exch_ts;
        else return rows.front().local_ts;
    }
};

template<std::size_t N>
class OrderBus {
    static_assert(std::has_single_bit(N));
    struct Message { Order order; std::int64_t time; };
    std::array<Message, N> slots_;
    std::size_t head_ = 0, tail_ = 0;
public:
    std::size_t size() const noexcept { return tail_ - head_; }
    std::int64_t next_time() const noexcept {
        return size() ? slots_[head_ % N].time : never;
    }
    void push(const Order& o, std::int64_t time) noexcept {
        assert(size() < N);
        if (size()) time = std::max(time, slots_[(tail_ - 1) % N].time);
        slots_[tail_++ % N] = {o, time};
    }
    Order pop() noexcept { return slots_[head_++ % N].order; }
};

} // namespace detail

// Single-asset L2 limit-order replay. The event mapping outlives this object.
// Cap bounds resident local orders. Request methods return false on invalid ID,
// a pending request, or a full order/mailbox table.
template<class QueueModel, class LatencyModel, std::size_t Cap = 64>
class Backtest {
    template<class, class, std::size_t> friend class MultiAssetBacktest;
    using Time = std::int64_t;
    static constexpr Time never = detail::never;
    AssetConfig config_;
    LatencyModel latency_;
    FlatBook local_book_, exch_book_;
    FlatOrderMap<Cap> local_orders_, resting_;
    detail::OrderBus<Cap> requests_;
    detail::OrderBus<4 * Cap> responses_;
    detail::Feed<EXCH_EVENT> exchange_;
    detail::Feed<LOCAL_EVENT> local_;
    StateValues state_;
    Time now_ = never;

    static Time later(Time t, Time delay) noexcept {
        assert(delay >= 0);
        return t > never - 1 - delay ? never - 1 : t + delay;
    }

    static void fill(Order& o, bool maker, Time tick, Time t) noexcept {
        o.maker = maker;
        o.exec_price_tick = tick;
        o.exec_qty = o.leaves_qty;
        o.leaves_qty = 0;
        o.status = status::FILLED;
        o.exch_ts = t;
    }

    void respond(const Order& o) noexcept {
        responses_.push(o, later(o.exch_ts, latency_.response(o.exch_ts)));
    }

    void maker_fill(Order& o, Time t) noexcept {
        fill(o, true, o.price_tick, t);
        respond(o);
        resting_.erase(o.order_id);
    }

    // Rust scans the crossed tick interval for short moves, and all orders for
    // long moves. These sets can differ after a depth clear uncovers old levels.
    template<bool Up>
    static bool scans_levels(Time from, Time to, std::size_t count) noexcept {
        if (from == (Up ? FlatBook::no_bid : FlatBook::no_ask)) return false;
        const auto delta = static_cast<__int128_t>(to) - from;
        return (Up ? delta : -delta) <= count;
    }

    template<bool Bid, bool Exchange>
    void depth_event(FlatBook& book, Time tick, const Event& e) noexcept {
        const auto best = [&] {
            return Bid ? book.best_bid_tick() : book.best_ask_tick();
        };
        const Time before = best();
        const double prev = Bid ? book.bid_qty_at_tick(tick) : book.ask_qty_at_tick(tick);
        if constexpr (Bid) book.update_bid(tick, e.qty);
        else book.update_ask(tick, e.qty);
        if constexpr (Exchange) {
            const Time after = best();
            const bool improved = Bid ? after > before : after < before;
            // Erase swaps in the last order. Walk backwards to visit each exactly once.
            const auto count = resting_.size();
            for (auto i = count; i != 0; ) {
                Order& o = resting_.begin()[--i];
                if (o.side == (Bid ? 1 : -1)) {
                    if (o.price_tick == tick) QueueModel::on_depth(o, prev, e.qty);
                } else if (improved && (Bid ? o.price_tick <= after : o.price_tick >= after)) {
                    if (!scans_levels<Bid>(before, after, count) || (Bid ? o.price_tick > before : o.price_tick < before))
                        maker_fill(o, e.exch_ts);
                }
            }
        }
    }

    template<bool BuyTrade>
    void trade_event(Time tick, const Event& e) noexcept {
        const Time from = BuyTrade ? exch_book_.best_bid_tick() : exch_book_.best_ask_tick();
        const bool levels = scans_levels<BuyTrade>(from, tick, resting_.size());
        for (auto i = resting_.size(); i != 0; ) {
            Order& o = resting_.begin()[--i];
            if (levels && (BuyTrade ? o.price_tick <= from : o.price_tick >= from)) continue;
            if (o.side != (BuyTrade ? -1 : 1)) continue;
            if (tick == o.price_tick) {
                QueueModel::trade(o, e.qty);
                if (!QueueModel::filled_qty(o, config_.lot_size)) continue;
            } else if (BuyTrade ? tick < o.price_tick : tick > o.price_tick) {
                continue;
            }
            maker_fill(o, e.exch_ts);
        }
    }

    template<bool Exchange>
    void market_event(FlatBook& book, const Event& e) noexcept {
        const bool bid = e.ev & BUY_EVENT;
        switch (event_kind(e.ev)) {
        case DEPTH_EVENT:
        case DEPTH_SNAPSHOT_EVENT: {
            const Time tick = book.price_to_tick(e.px);
            if (bid) depth_event<true, Exchange>(book, tick, e);
            else if (e.ev & SELL_EVENT) depth_event<false, Exchange>(book, tick, e);
            break;
        }
        case TRADE_EVENT:
            if constexpr (Exchange) {
                if (bid) trade_event<true>(book.price_to_tick(e.px), e);
                else if (e.ev & SELL_EVENT) trade_event<false>(book.price_to_tick(e.px), e);
            }
            break;
        case DEPTH_CLEAR_EVENT:
            if (bid) book.replay_clear_bid(book.price_to_tick(e.px));
            else if (e.ev & SELL_EVENT) book.replay_clear_ask(book.price_to_tick(e.px));
            else book.clear_all();
            break;
        }
    }

    void accept_request(Order o, Time t) noexcept {
        const bool cancel = o.req == req::CANCEL;
        o.req = req::NONE;
        if (cancel) {
            if (auto* active = resting_.find(o.order_id)) {
                o = *active;
                resting_.erase(o.order_id);
                o.status = status::CANCELED;
            } else {
                o.req = req::REJECTED;
            }
        } else {
            const bool buy = o.side > 0;
            const Time opposite = buy ? exch_book_.best_ask_tick() : exch_book_.best_bid_tick();
            const bool crosses = buy ? o.price_tick >= opposite : o.price_tick <= opposite;
            if (crosses && o.tif != GTX) {
                fill(o, false, opposite, t);
            } else if (crosses || o.tif == IOC || o.tif == FOK) {
                o.status = status::EXPIRED;
            } else {
                o.status = status::NEW;
                o.exch_ts = t;
                QueueModel::on_new_order(o, buy ? exch_book_.bid_qty_at_tick(o.price_tick)
                                                : exch_book_.ask_qty_at_tick(o.price_tick));
                [[maybe_unused]] auto* inserted = resting_.insert(o);
                assert(inserted);
            }
        }
        o.exch_ts = t;
        respond(o);
    }

    void receive(const Order& o) noexcept {
        if (o.status == status::FILLED) {
            const double value = config_.contract_size * local_book_.tick_to_price(o.exec_price_tick)
                                                       * o.exec_qty;
            state_.position += o.side * o.exec_qty;
            state_.balance -= o.side * value;
            state_.fee += value * (o.maker ? config_.maker_fee : config_.taker_fee);
            ++state_.num_trades;
            state_.trading_volume += o.exec_qty;
            state_.trading_value += value;
        }
        auto* local = local_orders_.find(o.order_id);
        if (o.req == req::REJECTED) {
            if (local && local->local_ts == o.local_ts) {
                if (local->req == req::NEW) local->status = status::EXPIRED;
                local->req = req::NONE;
            }
        } else if (local) {
            const Time local_ts = local->local_ts;
            *local = o;
            local->local_ts = local_ts;
        } else {
            [[maybe_unused]] auto* inserted = local_orders_.insert(o);
            assert(inserted);
        }
    }

    bool bus_available() const noexcept {
        // Each new request can generate an acceptance and a later maker fill.
        return requests_.size() < Cap &&
               2 * (requests_.size() + 1) + resting_.size() + responses_.size() <= 4 * Cap;
    }
    void send(const Order& o) noexcept {
        const Time delay = latency_.entry(now_);
        if (delay < 0) {
            Order rejected = o;
            rejected.req = req::REJECTED;
            responses_.push(rejected, later(now_, -std::max(delay, -never)));
        } else {
            requests_.push(o, later(now_, delay));
        }
    }
    bool submit(std::uint64_t id, double px, double qty, TimeInForce tif, std::int8_t side) noexcept {
        if (now_ == never || !bus_available()) return false;
        Order o{};
        o.order_id = id;
        o.price_tick = local_book_.price_to_tick(px);
        o.qty = o.leaves_qty = qty;
        o.side = side;
        o.tif = tif;
        o.req = req::NEW;
        o.local_ts = now_;
        if (!local_orders_.insert(o)) return false;
        send(o);
        return true;
    }

public:
    Backtest(std::span<const Event> events, AssetConfig config, LatencyModel latency)
        : config_(config), latency_(latency), exchange_{events}, local_{events} {
        for (auto* book : {&exch_book_, &local_book_})
            book->init(config.tick_size, config.lot_size, config.roi_lb_tick, config.roi_ub_tick);
    }

    // 0 = elapsed, 1 = exhausted (including pending requests and responses).
    int elapse(Time duration_ns) noexcept {
        assert(duration_ns >= 0);
        if (now_ == never) {
            exchange_.skip_unseen();
            local_.skip_unseen();
            now_ = std::min(exchange_.next_time(), local_.next_time());
            if (now_ == never) return 1;
        }
        const Time target = later(now_, duration_ns);
        // Between strategy calls local feed updates cannot affect exchange execution.
        // Batch each independent stream, preserving feed-before-request ties and FIFO acks.
        for (;;) {
            const Time arrival = requests_.next_time();
            const Time through = std::min(target, arrival);
            while (exchange_.next_time() <= through) {
                market_event<true>(exch_book_, exchange_.rows.front());
                exchange_.advance();
            }
            if (arrival > target) break;
            accept_request(requests_.pop(), arrival);
        }
        while (local_.next_time() <= target) {
            market_event<false>(local_book_, local_.rows.front());
            local_.advance();
        }
        while (responses_.next_time() <= target) receive(responses_.pop());
        if (exchange_.rows.empty() && local_.rows.empty() && !requests_.size() && !responses_.size())
            return 1;
        now_ = target;
        return 0;
    }

    bool submit_buy_order(std::uint64_t id, double px, double qty, TimeInForce tif) noexcept {
        return submit(id, px, qty, tif, 1);
    }
    bool submit_sell_order(std::uint64_t id, double px, double qty, TimeInForce tif) noexcept {
        return submit(id, px, qty, tif, -1);
    }
    bool cancel(std::uint64_t id) noexcept {
        auto* o = local_orders_.find(id);
        if (!o || o->req != req::NONE || !bus_available()) return false;
        o->req = req::CANCEL;
        o->local_ts = now_;
        send(*o);
        return true;
    }
    void clear_inactive_orders() noexcept {
        for (auto i = local_orders_.size(); i != 0; ) {
            const Order& o = local_orders_.begin()[--i];
            if (o.status == status::EXPIRED || o.status == status::FILLED || o.status == status::CANCELED)
                local_orders_.erase(o.order_id);
        }
    }
    Time current_timestamp() const noexcept { return now_; }
    const FlatBook& depth() const noexcept { return local_book_; }
    const FlatOrderMap<Cap>& orders() const noexcept { return local_orders_; }
    const StateValues& state_values() const noexcept { return state_; }
    double position() const noexcept { return state_.position; }
};

} // namespace xhftbacktest
#endif
