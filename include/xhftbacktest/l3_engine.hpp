#ifndef XHFTBACKTEST_L3_ENGINE_HPP
#define XHFTBACKTEST_L3_ENGINE_HPP

#include "engine.hpp"
#include "l3_book.hpp"

namespace xhftbacktest {

// Single-asset L3 replay. The caller owns the two-timeline event span. Strategy
// orders share FIFO priority with feed orders but never change published depth.
template<class LatencyModel, std::size_t Cap = 4096, std::size_t MaxBookOrders = 65536>
class L3Backtest {
    using Time = std::int64_t;
    using Id = std::uint64_t;
    static constexpr Time never = detail::never;

    AssetConfig config_;
    LatencyModel latency_;
    L3Book<MaxBookOrders> local_book_;
    L3Book<MaxBookOrders + Cap> exchange_book_;
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

    void respond(const Order& o) noexcept {
        responses_.push(o, std::max(Time{0}, later(o.exch_ts, latency_.response(o.exch_ts))));
    }

    // Older Order layouts omit the fresh-execution flag.
    template<class O>
    static void mark_execution(O& o, bool value) noexcept {
        if constexpr (requires { o.execution; }) o.execution = value;
    }

    static void fill(Order& o, bool maker, Time price_tick, Time t) noexcept {
        mark_execution(o, true);
        o.status = status::FILLED;
        o.maker = maker;
        o.exec_price_tick = price_tick;
        o.exec_qty = o.leaves_qty;
        o.leaves_qty = 0;
        o.exch_ts = t;
    }

    void maker_fill(Id id, Time t) noexcept {
        Order& o = *resting_.find(id);
        fill(o, true, o.price_tick, t);
        respond(o);
        exchange_book_.delete_strategy(id);
        resting_.erase(id);
    }

    // For a wide move Rust scans every strategy order through the new price.
    // Short moves only visit the inclusive interval. In both cases each price
    // level fills in queue order. Rust's cross-level hash iteration is unordered.
    void fill_between(int side, Time low, Time high, Time t, bool open_end = false) noexcept {
        if (!resting_.size()) return;
        if (open_end || std::uint64_t(high) - std::uint64_t(low) > resting_.size()) {
            if (side > 0) high = FlatBook::no_ask;
            else low = FlatBook::no_bid;
        }
        exchange_book_.for_each_strategy_in_range(side, low, high, [&](Id id) { maker_fill(id, t); });
    }

    bool market_fill(const Event& e) noexcept {
        if (!(e.ev & (BUY_EVENT | SELL_EVENT))) return true;
        const auto* order = exchange_book_.find_order(e.order_id);
        if (!order) return false;
        const Time tick = exchange_book_.price_to_tick(e.px);
        if (order->side > 0) {
            const Time bid = exchange_book_.best_bid_tick();
            if (tick < bid) fill_between(1, tick + 1, bid, e.exch_ts);
        } else {
            const Time ask = exchange_book_.best_ask_tick();
            if (tick > ask) fill_between(-1, ask, tick - 1, e.exch_ts);
        }
        exchange_book_.for_each_strategy_before(*order, [&](Id id) { maker_fill(id, e.exch_ts); });
        // The feed reports its own quantity change on a later modify/delete.
        return true;
    }

    void expire_side(int side, Time t) noexcept {
        for (auto i = resting_.size(); i != 0; ) {
            Order& o = resting_.begin()[--i];
            if (side && o.side != side) continue;
            o.status = status::EXPIRED;
            o.leaves_qty = o.exec_qty = 0;
            o.exch_ts = t;
            respond(o);
            resting_.erase(o.order_id);
        }
    }

    template<bool Exchange, std::size_t N>
    bool market_event(L3Book<N>& book, const Event& e) noexcept {
        const int side = (e.ev & BUY_EVENT) ? 1 : (e.ev & SELL_EVENT) ? -1 : 0;
        const auto kind = event_kind(e.ev);
        const Time t = Exchange ? e.exch_ts : e.local_ts;
        switch (kind) {
        case ADD_ORDER_EVENT:
        case MODIFY_ORDER_EVENT: {
            const Time bid = book.best_bid_tick(), ask = book.best_ask_tick();
            if (kind == MODIFY_ORDER_EVENT) {
                if (!book.modify_order(e.order_id, e.px, e.qty, t)) return false;
            } else if (side > 0) {
                if (!book.add_buy_order(e.order_id, e.px, e.qty, t)) return false;
            } else if (side < 0) {
                if (!book.add_sell_order(e.order_id, e.px, e.qty, t)) return false;
            }
            if constexpr (Exchange) {
                if (book.best_bid_tick() > bid)
                    fill_between(-1, bid + 1, book.best_bid_tick(), t, bid == FlatBook::no_bid);
                if (book.best_ask_tick() < ask)
                    fill_between(1, book.best_ask_tick(), ask - 1, t, ask == FlatBook::no_ask);
            }
            return true;
        }
        case CANCEL_ORDER_EVENT:
            return book.delete_order(e.order_id, t);
        case DEPTH_CLEAR_EVENT:
            book.clear_orders(side);
            if constexpr (Exchange) expire_side(side, t);
            break;
        case FILL_EVENT:
            if constexpr (Exchange) return market_fill(e);
            break;
        }
        return true;
    }

    void accept(Order o, Time t) noexcept {
        const auto request = std::exchange(o.req, req::NONE);
        if (request == req::NEW) {
            const bool buy = o.side > 0;
            const Time other = buy ? exchange_book_.best_ask_tick() : exchange_book_.best_bid_tick();
            const bool crosses = buy ? o.price_tick >= other : o.price_tick <= other;
            if (crosses && o.tif != GTX) {
                fill(o, false, other, t);
            } else if (crosses || o.tif == IOC || o.tif == FOK) {
                o.status = status::EXPIRED;
            } else if (exchange_book_.add_strategy(o.order_id, o.side, o.price_tick, o.qty)) {
                o.status = status::NEW;
                o.exch_ts = t;
                [[maybe_unused]] auto* inserted = resting_.insert(o);
                assert(inserted);
            } else {
                o.req = req::REJECTED;
            }
        } else if (auto* active = resting_.find(o.order_id)) {
            if (request == req::CANCEL) {
                o = *active;
                exchange_book_.delete_strategy(o.order_id);
                resting_.erase(o.order_id);
                o.status = status::CANCELED;
            } else if (exchange_book_.modify_strategy(o.order_id, o.price_tick, o.qty)) {
                // An amendment keeps the original local timestamp in the FIFO node.
                active->price_tick = o.price_tick;
                active->qty = active->leaves_qty = o.leaves_qty = o.qty;
                active->exch_ts = o.exch_ts;
            } else {
                o.req = req::REJECTED;
            }
        } else {
            o.req = req::REJECTED;
        }
        o.exch_ts = t;
        respond(o);
    }

    void receive(const Order& o) noexcept {
        if (o.status == status::FILLED) {
            const double value = config_.contract_size * local_book_.tick_to_price(o.exec_price_tick) * o.exec_qty;
            state_.position += o.exec_qty * o.side;
            state_.balance -= value * o.side;
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
            const auto local_ts = local->local_ts, exch_ts = local->exch_ts;
            *local = o;
            local->local_ts = local_ts;
            if (o.exch_ts <= 0) local->exch_ts = exch_ts;
        } else {
            [[maybe_unused]] auto* inserted = local_orders_.insert(o);
            assert(inserted);
        }
    }

    bool bus_available() const noexcept {
        return requests_.size() < Cap &&
               2 * (requests_.size() + 1) + resting_.size() + responses_.size() <= 4 * Cap;
    }

    void send(Order o) noexcept {
        mark_execution(o, false);
        const Time delay = latency_.entry(now_);
        if (delay < 0) {
            o.req = req::REJECTED;
            responses_.push(o, std::max(Time{0}, later(now_, -std::max(delay, -never))));
        } else {
            requests_.push(o, std::max(Time{0}, later(now_, delay)));
        }
    }

    bool submit(Id id, double px, double qty, TimeInForce tif, std::int8_t side) noexcept {
        if (now_ == never || !bus_available()) return false;
        Order o{};
        o.order_id = id;
        o.price_tick = local_book_.price_to_tick(px);
        o.qty = o.leaves_qty = qty;
        o.tif = tif;
        o.side = side;
        o.req = req::NEW;
        o.local_ts = now_;
        if (!local_orders_.insert(o)) return false;
        send(o);
        return true;
    }

    Order* pending_request(Id id, req::Code request) noexcept {
        auto* o = local_orders_.find(id);
        if (!o || o->req != req::NONE || !bus_available()) return nullptr;
        o->req = request;
        o->local_ts = now_;
        return o;
    }

public:
    L3Backtest(std::span<const Event> events, AssetConfig config, LatencyModel latency)
        : config_(config), latency_(std::move(latency)), exchange_{events}, local_{events} {
        local_book_.init(config.tick_size, config.lot_size, config.roi_lb_tick, config.roi_ub_tick);
        exchange_book_.init(config.tick_size, config.lot_size, config.roi_lb_tick, config.roi_ub_tick);
    }

    // 0 = elapsed, 1 = exhausted, 2 = invalid feed order ID or exhausted feed pool.
    int elapse(Time duration_ns) noexcept {
        assert(duration_ns >= 0);
        if (now_ == never) {
            exchange_.skip_unseen();
            local_.skip_unseen();
            now_ = std::min(exchange_.next_time(), local_.next_time());
            if (now_ == never) return 1;
        }
        const Time target = later(now_, duration_ns);
        for (;;) {
            const Time arrival = requests_.next_time();
            const Time through = std::min(arrival, target);
            while (exchange_.next_time() <= through) {
                if (!market_event<true>(exchange_book_, exchange_.rows.front())) return 2;
                exchange_.advance();
            }
            if (arrival > target) break;
            accept(requests_.pop(), arrival);
        }
        while (local_.next_time() <= target) {
            if (!market_event<false>(local_book_, local_.rows.front())) return 2;
            local_.advance();
        }
        while (responses_.next_time() <= target) receive(responses_.pop());
        if (exchange_.rows.empty() && local_.rows.empty() && !requests_.size() && !responses_.size())
            return 1;
        now_ = target;
        return 0;
    }

    bool submit_buy_order(Id id, double px, double qty, TimeInForce tif) noexcept {
        return submit(id, px, qty, tif, 1);
    }
    bool submit_sell_order(Id id, double px, double qty, TimeInForce tif) noexcept {
        return submit(id, px, qty, tif, -1);
    }
    bool cancel(Id id) noexcept {
        auto* o = pending_request(id, req::CANCEL);
        if (!o) return false;
        send(*o);
        return true;
    }
    bool modify(Id id, double px, double qty) noexcept {
        auto* o = pending_request(id, req::MODIFY);
        if (!o) return false;
        o->price_tick = local_book_.price_to_tick(px);
        o->qty = qty;
        send(*o);
        return true;
    }
    void clear_inactive_orders() noexcept {
        for (auto i = local_orders_.size(); i != 0; ) {
            const auto& o = local_orders_.begin()[--i];
            if (o.status == status::EXPIRED || o.status == status::FILLED || o.status == status::CANCELED)
                local_orders_.erase(o.order_id);
        }
    }

    Time current_timestamp() const noexcept { return now_; }
    const L3Book<MaxBookOrders>& depth() const noexcept { return local_book_; }
    const FlatOrderMap<Cap>& orders() const noexcept { return local_orders_; }
    const StateValues& state_values() const noexcept { return state_; }
    double position() const noexcept { return state_.position; }
};

} // namespace xhftbacktest
#endif
