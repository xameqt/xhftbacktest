#ifndef XHFTBACKTEST_ORDER_HPP
#define XHFTBACKTEST_ORDER_HPP

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>

namespace xhftbacktest {

struct NoPartialFill {};
struct PartialFill {};

enum TimeInForce : std::uint8_t { GTC = 0, GTX = 1, FOK = 2, IOC = 3 };
namespace status {
enum Code : std::uint8_t { NONE = 0, NEW = 1, EXPIRED = 2, FILLED = 3, CANCELED = 4, PARTIALLY_FILLED = 5 };
}
namespace req {
enum Code : std::uint8_t { NONE = 0, NEW = 1, CANCEL = 4, REJECTED = 6, MODIFY = 7 };
}

struct Order {
    std::uint64_t order_id;
    std::int64_t price_tick;
    double qty;
    double leaves_qty;
    double exec_qty;
    std::int64_t exec_price_tick;
    std::int8_t side;  // +1 buy, -1 sell
    std::uint8_t tif;
    std::uint8_t status;
    std::uint8_t req;
    double q_ahead;
    std::int64_t exch_ts;
    std::int64_t local_ts;
    double cum_trade_qty{};
    bool maker{};
    bool execution{};  // A fresh fill report, as opposed to an ack carrying the last exec_qty.
};

// Orders are contiguous for iteration. active_[i] is the hash bucket of orders_[i].
// Keep order_id fixed during membership. Erase moves the last order into the gap.
template<std::size_t Cap = 1024>
class FlatOrderMap {
    static_assert(std::has_single_bit(Cap) && Cap <= UINT32_MAX / 2);
    static constexpr std::size_t mask = 2 * Cap - 1;
    std::array<Order, Cap> orders_{};
    std::array<std::uint32_t, Cap> active_{};
    std::array<std::uint32_t, 2 * Cap> buckets_{};  // dense index + 1, zero is empty
    std::size_t size_ = 0;

    static constexpr std::size_t home(std::uint64_t id) noexcept {
        return (id * 0x9e3779b97f4a7c15ULL) >> (64 - std::bit_width(mask));
    }

    std::size_t locate(std::uint64_t id) const noexcept {
        auto slot = home(id);
        while (buckets_[slot] && orders_[buckets_[slot] - 1].order_id != id)
            slot = (slot + 1) & mask;
        return slot;
    }

public:
    std::size_t size() const noexcept { return size_; }
    Order* begin() noexcept { return orders_.data(); }
    Order* end() noexcept { return begin() + size_; }
    const Order* begin() const noexcept { return orders_.data(); }
    const Order* end() const noexcept { return begin() + size_; }

    Order* find(std::uint64_t id) noexcept {
        const auto index = buckets_[locate(id)];
        return index ? &orders_[index - 1] : nullptr;
    }
    const Order* find(std::uint64_t id) const noexcept {
        const auto index = buckets_[locate(id)];
        return index ? &orders_[index - 1] : nullptr;
    }

    // Duplicate ID or exhausted capacity returns nullptr.
    Order* insert(const Order& order) noexcept {
        if (size_ == Cap) return nullptr;
        const auto slot = locate(order.order_id);
        if (buckets_[slot]) return nullptr;
        orders_[size_] = order;
        active_[size_] = static_cast<std::uint32_t>(slot);
        buckets_[slot] = static_cast<std::uint32_t>(size_ + 1);
        return &orders_[size_++];
    }

    bool erase(std::uint64_t id) noexcept {
        auto hole = locate(id);
        const auto index = buckets_[hole];
        if (!index) return false;

        const auto last = --size_;
        orders_[index - 1] = orders_[last];
        active_[index - 1] = active_[last];
        buckets_[active_[last]] = index;

        // Shift a bucket back only when the hole lies on its probe path.
        for (auto slot = (hole + 1) & mask; buckets_[slot]; slot = (slot + 1) & mask) {
            const auto entry = buckets_[slot];
            const auto start = home(orders_[entry - 1].order_id);
            if (((hole - start) & mask) < ((slot - start) & mask)) {
                buckets_[hole] = entry;
                active_[entry - 1] = static_cast<std::uint32_t>(hole);
                hole = slot;
            }
        }
        buckets_[hole] = 0;
        return true;
    }
};

}  // namespace xhftbacktest
#endif
