#ifndef XHFTBACKTEST_EVENT_HPP
#define XHFTBACKTEST_EVENT_HPP

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace xhftbacktest {

struct alignas(8) Event {
    std::uint64_t ev;
    std::int64_t exch_ts;
    std::int64_t local_ts;
    double px;
    double qty;
    std::uint64_t order_id;
    std::int64_t ival;
    double fval;
};

static_assert(std::is_trivial_v<Event> && std::is_standard_layout_v<Event>);
static_assert(sizeof(Event) == 64);
static_assert(alignof(Event) == 8);
static_assert(offsetof(Event, ev)       == 0);
static_assert(offsetof(Event, exch_ts)  == 8);
static_assert(offsetof(Event, local_ts) == 16);
static_assert(offsetof(Event, px)       == 24);
static_assert(offsetof(Event, qty)      == 32);
static_assert(offsetof(Event, order_id) == 40);
static_assert(offsetof(Event, ival)     == 48);
static_assert(offsetof(Event, fval)     == 56);

inline constexpr std::uint64_t DEPTH_EVENT          = 0x1;
inline constexpr std::uint64_t TRADE_EVENT          = 0x2;
inline constexpr std::uint64_t DEPTH_CLEAR_EVENT    = 0x3;
inline constexpr std::uint64_t DEPTH_SNAPSHOT_EVENT = 0x4;
inline constexpr std::uint64_t DEPTH_BBO_EVENT      = 0x5;
inline constexpr std::uint64_t ADD_ORDER_EVENT      = 0xa;
inline constexpr std::uint64_t CANCEL_ORDER_EVENT   = 0xb;
inline constexpr std::uint64_t MODIFY_ORDER_EVENT   = 0xc;
inline constexpr std::uint64_t FILL_EVENT           = 0xd;
inline constexpr std::uint64_t SELL_EVENT           = 0x10000000;
inline constexpr std::uint64_t BUY_EVENT            = 0x20000000;
inline constexpr std::uint64_t LOCAL_EVENT          = 0x40000000;
inline constexpr std::uint64_t EXCH_EVENT           = 0x80000000;

constexpr std::uint64_t event_kind(std::uint64_t ev) noexcept { return ev & 0xff; }

constexpr bool is_event(std::uint64_t ev, std::uint64_t mask) noexcept {
    return (ev & mask) == mask && (!event_kind(mask) || event_kind(ev) == event_kind(mask));
}

} // namespace xhftbacktest
#endif
