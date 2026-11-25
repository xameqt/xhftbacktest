#ifndef XHFTBACKTEST_RECORDER_HPP
#define XHFTBACKTEST_RECORDER_HPP

#include "asset_type.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <vector>

namespace xhftbacktest {

struct RecordRow {
    std::int64_t timestamp;
    double mid, balance, position, fee;
    std::int64_t num_trades;
    double trading_volume, trading_value, equity;
};

struct SummaryStats {
    static constexpr double undefined = std::numeric_limits<double>::quiet_NaN();
    double total_return = undefined, annualized_sharpe = undefined, annualized_sortino = undefined;
    double max_drawdown = undefined, return_over_mdd = undefined;
    std::int64_t num_trades{};
    double total_volume{}, total_value{}, total_fee{}, final_equity = undefined;
};

class Recorder {
    std::vector<std::int64_t> timestamp_, num_trades_;
    std::vector<double> mid_, balance_, position_, fee_, trading_volume_, trading_value_, equity_;
    std::size_t used_ = 0;

public:
    explicit Recorder(std::size_t capacity)
        : timestamp_(capacity), num_trades_(capacity), mid_(capacity), balance_(capacity),
          position_(capacity), fee_(capacity), trading_volume_(capacity), trading_value_(capacity),
          equity_(capacity) {}
    Recorder(const Recorder&) = delete;
    Recorder& operator=(const Recorder&) = delete;

    std::size_t size() const noexcept { return used_; }

    // Takes the engine's StateValues. A full recorder returns false and keeps earlier samples.
    template<class State, class Asset>
    bool record(std::int64_t ts, double mid, const State& s, const Asset& asset) noexcept {
        const auto i = used_;
        if (i == timestamp_.size()) return false;
        equity_[i] = asset.equity(mid, s.balance, s.position, s.fee);
        timestamp_[i] = ts;
        mid_[i] = mid;
        balance_[i] = s.balance;
        position_[i] = s.position;
        fee_[i] = s.fee;
        num_trades_[i] = s.num_trades;
        trading_volume_[i] = s.trading_volume;
        trading_value_[i] = s.trading_value;
        used_ = i + 1;
        return true;
    }

    RecordRow operator[](std::size_t i) const noexcept {
        assert(i < used_);
        return {timestamp_[i], mid_[i], balance_[i], position_[i], fee_[i], num_trades_[i],
                trading_volume_[i], trading_value_[i], equity_[i]};
    }

    // Return and drawdown are in settlement units. Ratios use changes in net equity,
    // sample standard deviation and downside RMS (zero target, all intervals).
    // Annualization uses the mean sampling interval in ns. Undefined metrics are NaN.
    // Trade, volume, value and fee totals come from the final cumulative engine state.
    SummaryStats summary(double trading_days_per_year = 365.0) const noexcept {
        SummaryStats s;
        if (!used_) return s;
        const auto last = used_ - 1;
        s.num_trades = num_trades_[last];
        s.total_volume = trading_volume_[last];
        s.total_value = trading_value_[last];
        s.total_fee = fee_[last];
        s.final_equity = equity_[last];

        double peak = equity_[0], drawdown = 0.0;
        double mean = 0.0, m2 = 0.0, downside_squares = 0.0;
        bool increasing_time = true;
        if (!std::isfinite(peak)) return s;
        for (std::size_t i = 1; i < used_; ++i) {
            const double eq = equity_[i];
            if (!std::isfinite(eq)) return s;
            peak = std::max(peak, eq);
            drawdown = std::max(drawdown, peak - eq);
            const double pnl = eq - equity_[i - 1];
            const double delta = pnl - mean;
            mean += delta / static_cast<double>(i);
            m2 += delta * (pnl - mean);
            const double loss = std::min(0.0, pnl);
            downside_squares += loss * loss;
            increasing_time &= timestamp_[i] > timestamp_[i - 1];
        }
        s.total_return = s.final_equity - equity_[0];
        s.max_drawdown = drawdown;
        if (drawdown > 0.0) s.return_over_mdd = s.total_return / drawdown;
        if (!last || !increasing_time || !(trading_days_per_year > 0.0)) return s;
        const auto elapsed = static_cast<std::uint64_t>(timestamp_[last])
                           - static_cast<std::uint64_t>(timestamp_[0]);
        const double periods = static_cast<double>(last);
        constexpr double ns_per_day = 86'400'000'000'000.0;
        const double scale = std::sqrt(trading_days_per_year * ns_per_day * periods / elapsed);
        if (last > 1 && m2 > 0.0) s.annualized_sharpe = mean / std::sqrt(m2 / (periods - 1.0)) * scale;
        if (downside_squares > 0.0) s.annualized_sortino = mean / std::sqrt(downside_squares / periods) * scale;
        return s;
    }

    bool write_csv(const std::filesystem::path& path) const {
        std::ofstream out(path);
        if (!out) return false;
        out.imbue(std::locale::classic());
        out << std::setprecision(std::numeric_limits<double>::max_digits10)
            << "timestamp,mid,balance,position,fee,num_trades,trading_volume,trading_value,equity\n";
        for (std::size_t i = 0; i < used_ && out; ++i) {
            const auto r = (*this)[i];
            out << r.timestamp << ',' << r.mid << ',' << r.balance << ',' << r.position << ','
                << r.fee << ',' << r.num_trades << ',' << r.trading_volume << ','
                << r.trading_value << ',' << r.equity << '\n';
        }
        out.close();
        return static_cast<bool>(out);
    }
};

} // namespace xhftbacktest
#endif
