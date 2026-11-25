#ifndef XHFTBACKTEST_MULTI_ASSET_HPP
#define XHFTBACKTEST_MULTI_ASSET_HPP

#include "engine.hpp"
#include <vector>

namespace xhftbacktest {

template<class QueueModel, class LatencyModel, std::size_t Cap = 4096>
class MultiAssetBacktest {
    using Asset = Backtest<QueueModel, LatencyModel, Cap>;
    using Time = std::int64_t;
    std::vector<Asset> assets_;
    Time current_ts_ = detail::never;

    // v2.4.4 starts lazily at the earliest event on either visible timeline.
    bool start() noexcept {
        if (current_ts_ == detail::never) {
            for (const auto& a : assets_)
                current_ts_ = std::min({current_ts_, a.exchange_.next_time(), a.local_.next_time()});
            for (auto& a : assets_) a.now_ = current_ts_;
        }
        return current_ts_ != detail::never;
    }

    bool advance_to(Time target) noexcept {
        const Time duration = target - current_ts_;
        bool more = false;
        for (auto& a : assets_) {
            more |= a.elapse(duration) == 0;
            a.now_ = target;  // Also synchronize assets whose feeds have ended.
        }
        if (more) {
            current_ts_ = target;
        } else {
            // Rust applies the final events but retains the clock at EOF.
            for (auto& a : assets_) a.now_ = current_ts_;
        }
        return more;
    }

    // An exchange event can create a response earlier than the next local feed.
    // Stop exchange processing as soon as such a response becomes observable.
    Time until_response(Time limit) noexcept {
        for (;;) {
            Time next = detail::never;
            for (const auto& a : assets_) {
                limit = std::min(limit, a.responses_.next_time());
                next = std::min(next, a.requests_.next_time());
                if (a.resting_.size()) next = std::min(next, a.exchange_.next_time());
            }
            if (next > limit) return limit;
            for (auto& a : assets_) {
                while (a.exchange_.next_time() <= next) {
                    a.template market_event<true>(a.exch_book_, a.exchange_.rows.front());
                    a.exchange_.advance();
                }
                while (a.requests_.next_time() == next)
                    a.accept_request(a.requests_.pop(), next);
            }
        }
    }

public:
    // Register before the first advance. Event storage must outlive the backtest.
    // Like Backtest, the queue model uses static callbacks.
    std::size_t add_asset(AssetConfig config, const Event* data, std::size_t count,
                          QueueModel, LatencyModel latency) {
        assert(current_ts_ == detail::never);
        auto& a = assets_.emplace_back(std::span<const Event>(data, count), config, latency);
        a.exchange_.skip_unseen();
        a.local_.skip_unseen();
        return assets_.size() - 1;
    }

    Time current_timestamp() const noexcept { return current_ts_; }
    std::size_t asset_count() const noexcept { return assets_.size(); }

    // True while any feed, order request or response remains after the advance.
    bool elapse(Time duration_ns) noexcept {
        return start() && advance_to(Asset::later(current_ts_, duration_ns));
    }

    // Official wait codes: 0 timeout, 1 end of data, 2 local feed, 3 order response.
    int wait_next_feed(bool include_order_resp, Time timeout_ns) noexcept {
        if (!start()) return 1;
        Time limit = Asset::later(current_ts_, timeout_ns);
        for (const auto& a : assets_) limit = std::min(limit, a.local_.next_time());
        if (include_order_resp) limit = until_response(limit);

        // At a tie the last qualifying event wins, in registration order, with
        // each asset's local feed preceding its order response.
        int result = 0;
        for (const auto& a : assets_) {
            if (a.local_.next_time() == limit) result = 2;
            if (include_order_resp && a.responses_.next_time() == limit) result = 3;
        }
        return advance_to(limit) ? result : 1;
    }

    const FlatBook& depth(std::size_t i) const noexcept { return assets_[i].depth(); }
    const StateValues& state_values(std::size_t i) const noexcept { return assets_[i].state_values(); }
    const FlatOrderMap<Cap>& orders(std::size_t i) const noexcept { return assets_[i].orders(); }
    double position(std::size_t i) const noexcept { return assets_[i].position(); }
    bool submit_buy_order(std::size_t i, std::uint64_t id, double px, double qty,
                          TimeInForce tif) noexcept {
        return assets_[i].submit_buy_order(id, px, qty, tif);
    }
    bool submit_sell_order(std::size_t i, std::uint64_t id, double px, double qty,
                           TimeInForce tif) noexcept {
        return assets_[i].submit_sell_order(id, px, qty, tif);
    }
    bool cancel(std::size_t i, std::uint64_t id) noexcept { return assets_[i].cancel(id); }
    void clear_inactive_orders(std::size_t i) noexcept { assets_[i].clear_inactive_orders(); }

    // Mid-marked linear contracts in a common quote currency, net of fees.
    double total_equity() const noexcept {
        double total = 0.0;
        for (const auto& a : assets_) {
            const auto& s = a.state_values();
            double value = s.balance;
            if (s.position != 0.0) {
                const double mid = 0.5 * (a.depth().best_bid() + a.depth().best_ask());
                value += a.config_.contract_size * mid * s.position;
            }
            total += value - s.fee;
        }
        return total;
    }
};

} // namespace xhftbacktest
#endif
