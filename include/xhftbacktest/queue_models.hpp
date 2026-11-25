#ifndef XHFTBACKTEST_QUEUE_MODELS_HPP
#define XHFTBACKTEST_QUEUE_MODELS_HPP

#include "order.hpp"
#include <algorithm>
#include <cmath>
#include <utility>

namespace xhftbacktest {

struct RiskAdverseQueueModel {
    static void on_new_order(Order& o, double qty) noexcept { o.q_ahead = qty; }
    static void on_depth(Order& o, double, double qty) noexcept {
        o.q_ahead = std::fmin(o.q_ahead, qty);
    }
    static void trade(Order& o, double qty) noexcept { o.q_ahead -= qty; }

    static double filled_qty(Order& o, double lot) noexcept {
        const double lots = std::round(-o.q_ahead / lot);
        if (!(lots > 0)) return 0;
        o.q_ahead = 0;
        return lots * lot;
    }

    // Quantity available to a caller that handles partial executions itself.
    static double on_trade(Order& o, double qty) noexcept {
        trade(o, qty);
        const double excess = std::max(0.0, -o.q_ahead);
        o.q_ahead = std::max(0.0, o.q_ahead);
        return std::min(excess, o.leaves_qty);
    }
};

template<class Probability>
struct ProbQueueModel : RiskAdverseQueueModel {
    static void on_new_order(Order& o, double qty) noexcept {
        o.q_ahead = qty;
        o.cum_trade_qty = 0;
    }
    static void trade(Order& o, double qty) noexcept {
        o.q_ahead -= qty;
        o.cum_trade_qty += qty;
    }
    static double on_trade(Order& o, double qty) noexcept {
        o.cum_trade_qty += qty;
        return RiskAdverseQueueModel::on_trade(o, qty);
    }
    static void on_depth(Order& o, double prev, double qty) noexcept {
        const double change = (prev - qty) - std::exchange(o.cum_trade_qty, 0.0);
        if (change < 0) {
            o.q_ahead = std::fmin(o.q_ahead, qty);
            return;
        }
        const double back = prev - o.q_ahead;
        double p = Probability::behind(o.q_ahead, back);
        if (std::isinf(p)) p = 1.0;
        // Rust f64::min selects the numeric operand when the other is NaN.
        o.q_ahead = std::fmin(qty, o.q_ahead - (1.0 - p) * change
                                   + std::fmin(back - p * change, 0.0));
    }
};

template<int Power>
struct PowerProbQueueFunc {
    static_assert(Power > 0);
    static double behind(double front, double back) noexcept {
        const double b = std::pow(back, Power);
        return b / (b + std::pow(front, Power));
    }
};

template<int Power = 3>
using PowerProbQueueModel = ProbQueueModel<PowerProbQueueFunc<Power>>;

struct LogProbQueueFunc {
    static double behind(double front, double back) noexcept {
        const double b = std::log(1.0 + back);
        return b / (b + std::log(1.0 + front));
    }
};
using LogProbQueueModel = ProbQueueModel<LogProbQueueFunc>;

} // namespace xhftbacktest
#endif
