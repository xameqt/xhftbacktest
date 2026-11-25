#ifndef XHFTBACKTEST_ENGINE_HPP
#define XHFTBACKTEST_ENGINE_HPP

#include "asset_type.hpp"
#include "book.hpp"
#include "latency_models.hpp"
#include "mmap_loader.hpp"
#include "queue_models.hpp"
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>

namespace xhftbacktest {

using LinearAssetMode = LinearAsset;
using InverseAssetMode = InverseAsset;

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

template<std::size_t N, bool Dynamic = false>
class OrderBus {
    static_assert(std::has_single_bit(N));
    struct Message { Order order; std::int64_t time; };
    static_assert(std::is_trivially_copyable_v<Message>);
    struct Reserved {
        std::unique_ptr<Message[], decltype(&std::free)> data{nullptr, &std::free};
        std::size_t capacity{}, mask{};
    };
    std::conditional_t<Dynamic, Reserved, std::array<Message, N>> slots_;
    std::size_t head_ = 0, tail_ = 0;
    const Message& at(std::size_t i) const noexcept {
        if constexpr (Dynamic) return slots_.data[i & slots_.mask];
        else return slots_[i % N];
    }
public:
    OrderBus() = default;
    OrderBus(const OrderBus&) requires (!Dynamic) = default;
    OrderBus(const OrderBus& other) requires Dynamic {
        reserve(other.capacity());
        for (auto i = other.head_; i != other.tail_; ++i)
            push(other.at(i).order, other.at(i).time);
    }
    OrderBus& operator=(const OrderBus&) requires (!Dynamic) = default;
    OrderBus& operator=(const OrderBus& other) requires Dynamic { return *this = OrderBus(other); }
    OrderBus(OrderBus&&) noexcept = default;
    OrderBus& operator=(OrderBus&&) noexcept = default;
    void reserve(std::size_t n) requires Dynamic {
        if (n > std::bit_floor(SIZE_MAX / sizeof(Message))) throw std::length_error("mailbox too large");
        slots_.capacity = std::bit_ceil(n);
        slots_.data.reset(static_cast<Message*>(std::calloc(slots_.capacity, sizeof(Message))));
        if (!slots_.data) throw std::bad_alloc{};
        slots_.mask = std::min(N, slots_.capacity) - 1;
    }
    std::size_t capacity() const noexcept {
        if constexpr (Dynamic) return slots_.capacity;
        else return N;
    }
    std::size_t size() const noexcept { return tail_ - head_; }
    std::int64_t next_time() const noexcept {
        return size() ? at(head_).time : never;
    }
    void push(const Order& o, std::int64_t time) noexcept {
        assert(size() < capacity());
        if (size()) time = std::max(time, at(tail_ - 1).time);
        if constexpr (Dynamic) {
            // Double the ring inside the existing allocation only at a new high-water mark.
            if (size() > slots_.mask) {
                const auto old = slots_.mask + 1;
                head_ &= slots_.mask;
                std::memcpy(slots_.data.get() + old, slots_.data.get(), head_ * sizeof(Message));
                tail_ = head_ + old;
                slots_.mask = 2 * old - 1;
            }
            slots_.data[tail_++ & slots_.mask] = {o, time};
        } else {
            slots_[tail_++ % N] = {o, time};
        }
    }
    Order pop() noexcept { return at(head_++).order; }
};

} // namespace detail

// Single-asset L2 limit-order replay. The event mapping outlives this object.
// Cap bounds resident local orders. Request methods return false on invalid ID,
// a pending request, or a full order/mailbox table.
template<class QueueModel, class LatencyModel, std::size_t Cap = 64,
         class FillModel = NoPartialFill, class AssetMode = LinearAssetMode>
class Backtest {
    template<class, class, std::size_t, class, class> friend class MultiAssetBacktest;
    static constexpr bool partial = std::is_same_v<FillModel, PartialFill>;
    static_assert(partial || std::is_same_v<FillModel, NoPartialFill>);
    using Time = std::int64_t;
    static constexpr Time never = detail::never;
    AssetConfig config_;
    LatencyModel latency_;
    FlatBook local_book_, exch_book_;
    FlatOrderMap<Cap> local_orders_, resting_;
    detail::OrderBus<Cap> requests_;
    detail::OrderBus<4 * Cap, partial> responses_;
    struct ResponseBudget { std::size_t request, market; };
    struct Empty {};
    [[no_unique_address]] std::conditional_t<partial, ResponseBudget, Empty> response_budget_;
    detail::Feed<EXCH_EVENT> exchange_;
    detail::Feed<LOCAL_EVENT> local_;
    StateValues state_;
    Time now_ = never;

    static Time later(Time t, Time delay) noexcept {
        assert(delay >= 0);
        return t > never - 1 - delay ? never - 1 : t + delay;
    }

    // Only a trade at the order price can generate repeated maker reports.
    // Bound their overlap by the latency horizon; custom latency models may
    // delay them for the whole tape. A taker can sweep each displayed tick once.
    void reserve_partial_responses(std::span<const Event> rows) requires partial {
        std::uint64_t horizon = UINT64_MAX;
        const auto delay_bound = [](Time entry, Time response) {
            const auto rejection = entry < 0 ? 0 - static_cast<std::uint64_t>(entry) : 0;
            return std::max(rejection, static_cast<std::uint64_t>(std::max<Time>(response, 0)));
        };
        if constexpr (std::is_same_v<LatencyModel, ConstantLatency>)
            horizon = delay_bound(latency_.entry_ns, latency_.resp_ns);
        else if constexpr (std::is_same_v<LatencyModel, IntpOrderLatency>)
            horizon = delay_bound(*std::min_element(latency_.entry_ns.begin(), latency_.entry_ns.end()),
                                  *std::max_element(latency_.resp_ns.begin(), latency_.resp_ns.end()));

        std::vector<Time> levels;
        std::size_t first = 0, trades = 0, peak = 0;
        for (const auto& e : rows) {
            if (!(e.ev & EXCH_EVENT)) continue;
            const auto kind = event_kind(e.ev);
            if (kind == TRADE_EVENT) {
                while (static_cast<std::uint64_t>(e.exch_ts) - static_cast<std::uint64_t>(rows[first].exch_ts) > horizon
                       || !(rows[first].ev & EXCH_EVENT)) {
                    if ((rows[first].ev & EXCH_EVENT) && event_kind(rows[first].ev) == TRADE_EVENT) --trades;
                    ++first;
                }
                peak = std::max(peak, ++trades);
            } else if ((kind == DEPTH_EVENT || kind == DEPTH_SNAPSHOT_EVENT) && e.qty > 0) {
                const Time tick = exch_book_.price_to_tick(e.px);
                if (tick >= config_.roi_lb_tick && tick <= config_.roi_ub_tick) levels.push_back(tick);
            }
        }
        std::sort(levels.begin(), levels.end());
        const std::size_t ticks = std::unique(levels.begin(), levels.end()) - levels.begin();
        // Admission reserves a sweep and its ack for each request, plus capacity
        // for resting orders to finish. A second trade window allows new requests
        // while a full window of maker reports is still in flight.
        const auto slots = static_cast<__uint128_t>(Cap) * (2 * static_cast<__uint128_t>(peak) + ticks + 4);
        if (slots > SIZE_MAX) throw std::length_error("partial-fill mailbox too large");
        response_budget_ = {ticks + 2, Cap * (peak + 1)};
        responses_.reserve(static_cast<std::size_t>(slots));
    }

    void fill(Order& o, bool maker, Time tick, double qty, Time t) noexcept {
        o.maker = maker;
        o.exec_price_tick = tick;
        o.execution = true;
        o.exec_qty = qty;
        if constexpr (partial) {
            o.leaves_qty -= qty;
            o.status = std::round(o.leaves_qty / config_.lot_size) > 0
                ? status::PARTIALLY_FILLED : status::FILLED;
        } else {
            o.leaves_qty = 0;
            o.status = status::FILLED;
        }
        o.exch_ts = t;
    }

    void respond(const Order& o) noexcept {
        if constexpr (partial) {
            // Release older deliveries while batching a long exchange advance.
            // A response tied with the current input remains visible to wait_next_feed.
            while (responses_.next_time() < o.exch_ts) receive(responses_.pop());
        }
        responses_.push(o, later(o.exch_ts, latency_.response(o.exch_ts)));
    }

    void maker_fill(Order& o, double qty, Time t) noexcept {
        fill(o, true, o.price_tick, qty, t);
        respond(o);
        if constexpr (partial) {
            if (o.status == status::PARTIALLY_FILLED) return;
        }
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
                        maker_fill(o, o.leaves_qty, e.exch_ts);
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
            double qty = o.leaves_qty;
            if (tick == o.price_tick) {
                QueueModel::trade(o, e.qty);
                const double available = QueueModel::filled_qty(o, config_.lot_size);
                if (!(available > 0)) continue;
                if constexpr (partial) qty = std::min(qty, available);
            } else if (BuyTrade ? tick < o.price_tick : tick > o.price_tick) {
                continue;
            }
            maker_fill(o, qty, e.exch_ts);
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

    // A taker walks the historical book without consuming its displayed depth.
    template<bool Buy>
    void take_depth(Order& o, Time best, Time t) noexcept {
        const auto at = [&](Time tick) {
            if constexpr (Buy) return exch_book_.ask_qty_at_tick(tick);
            else return exch_book_.bid_qty_at_tick(tick);
        };
        Time last = Buy ? std::min(o.price_tick, config_.roi_ub_tick)
                        : std::max(o.price_tick, config_.roi_lb_tick);
        constexpr int step = Buy ? 1 : -1;
        if (o.tif == FOK) {
            const double lots = std::round(o.qty / config_.lot_size);
            double available = 0;
            for (Time k = best; Buy ? k <= last : k >= last; k += step) {
                available += at(k);
                if (std::round(available / config_.lot_size) >= lots) break;
            }
            if (!(std::round(available / config_.lot_size) >= lots)) {
                o.status = status::EXPIRED;
                return;
            }
        }
        // 2.4.4 rolls a GTC buy's entire residual into one fill at its limit.
        // A GTC sell first takes displayed quantity at the limit too.
        if constexpr (Buy) {
            if (o.tif == GTC && last == o.price_tick) --last;
        }
        for (Time k = best; Buy ? k <= last : k >= last; k += step) {
            const double displayed = at(k);
            if (displayed > 0) {
                fill(o, false, k, std::min(displayed, o.leaves_qty), t);
                if (o.status == status::FILLED) return;
                respond(o);
            }
        }
        if (o.tif == GTC) {
            fill(o, false, o.price_tick, o.leaves_qty, t);
        } else {
            o.status = status::EXPIRED;
            o.execution = false;
        }
    }

    void accept_new(Order& o, Time t) noexcept {
        const bool buy = o.side > 0;
        const Time opposite = buy ? exch_book_.best_ask_tick() : exch_book_.best_bid_tick();
        const bool crosses = buy ? o.price_tick >= opposite : o.price_tick <= opposite;
        if (crosses && o.tif != GTX) {
            if constexpr (partial) {
                if (buy) take_depth<true>(o, opposite, t);
                else take_depth<false>(o, opposite, t);
            } else {
                fill(o, false, opposite, o.leaves_qty, t);
            }
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

    void accept_request(Order o, Time t) noexcept {
        const auto request = std::exchange(o.req, req::NONE);
        o.execution = false;
        if (request == req::NEW) {
            accept_new(o, t);
        } else if (auto* active = resting_.find(o.order_id)) {
            if (request == req::CANCEL) {
                o = *active;
                resting_.erase(o.order_id);
                o.status = status::CANCELED;
                o.execution = false;
            } else if (o.price_tick != active->price_tick || o.qty > active->leaves_qty) {
                resting_.erase(o.order_id);
                // The 2.4.4 replacement carries the request's original leaves_qty.
                accept_new(o, t);
            } else {
                active->qty = active->leaves_qty = o.leaves_qty = o.qty;
                active->exch_ts = t;
            }
        } else {
            o.req = req::REJECTED;
        }
        o.exch_ts = t;
        respond(o);
    }

    void receive(const Order& o) noexcept {
        if (o.execution) {
            const double value = AssetMode{config_.contract_size}.amount(
                local_book_.tick_to_price(o.exec_price_tick), o.exec_qty);
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
        if constexpr (partial)
            return requests_.size() < Cap && responses_.size() + response_budget_.market
                + (requests_.size() + 1) * response_budget_.request <= responses_.capacity();
        // Each new request can generate an acceptance and a later maker fill.
        return requests_.size() < Cap &&
               2 * (requests_.size() + 1) + resting_.size() + responses_.size() <= 4 * Cap;
    }
    void send(const Order& o) noexcept {
        const Time delay = latency_.entry(now_);
        if (delay < 0) {
            Order rejected = o;
            rejected.req = req::REJECTED;
            rejected.execution = false;
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
        if constexpr (partial) reserve_partial_responses(events);
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
    bool modify(std::uint64_t id, double price, double qty) noexcept {
        auto* o = local_orders_.find(id);
        if (!o || o->req != req::NONE || !bus_available()) return false;
        o->price_tick = local_book_.price_to_tick(price);
        o->qty = qty;
        o->req = req::MODIFY;
        o->local_ts = now_;
        send(*o);
        return true;
    }
    double equity(double mark_price) const noexcept {
        return AssetMode{config_.contract_size}.equity(mark_price, state_.balance, state_.position, state_.fee);
    }
    // Positive rates debit a long's settlement equity. Inverse equity negates raw balance.
    void apply_funding(double funding_rate, double mark_price) noexcept {
        const double cash = AssetMode{config_.contract_size}.funding_cashflow(mark_price, state_.position, funding_rate);
        if constexpr (std::is_same_v<AssetMode, InverseAssetMode>) state_.balance -= cash;
        else state_.balance += cash;
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
