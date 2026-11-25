#ifndef XHFTBACKTEST_ASSET_TYPE_HPP
#define XHFTBACKTEST_ASSET_TYPE_HPP

namespace xhftbacktest {

// amount and equity follow hftbacktest py-v2.4.4, including operation order.
// Funding is expressed as a change in settlement equity (positive = receipt).
struct LinearAsset {
    double contract_size = 1.0;

    constexpr double amount(double exec_price, double qty) const noexcept {
        return contract_size * exec_price * qty;
    }
    constexpr double equity(double mid, double balance, double position, double fee) const noexcept {
        return balance + contract_size * position * mid - fee;
    }
    constexpr double funding_cashflow(double mark_px, double position, double rate) const noexcept {
        return -contract_size * mark_px * position * rate;
    }
};

struct InverseAsset {
    double contract_size = 1.0;

    constexpr double amount(double exec_price, double qty) const noexcept {
        return contract_size * qty / exec_price;
    }
    constexpr double equity(double mid, double balance, double position, double fee) const noexcept {
        // A buy debits the raw balance by its inverse amount.
        return -balance - contract_size * position / mid - fee;
    }
    constexpr double funding_cashflow(double mark_px, double position, double rate) const noexcept {
        return -contract_size * position / mark_px * rate;
    }
};

} // namespace xhftbacktest
#endif
