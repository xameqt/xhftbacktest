#include <nanobind/nanobind.h>
#include <nanobind/stl/filesystem.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#include "xhftbacktest/multi_asset.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {
using namespace xhftbacktest;
using Path = std::filesystem::path;
using Ns = std::int64_t;
using Id = std::uint64_t;
enum class Side : std::int8_t { BUY = 1, SELL = -1 };

Ns nonnegative(Ns ns) {
    if (ns < 0) throw nb::value_error("expected a nonnegative time interval");
    return ns;
}

// The engine borrows rows. Only the mmap or the Python reference owns the payload.
struct EventSource {
    MmapLoader file;
    nb::object array;
    std::span<const Event> rows;

    explicit EventSource(const Path& path) : file(path), rows(file.events()) {}
    explicit EventSource(nb::object obj) : array(std::move(obj)) {
        if (!PyArray_Check(array.ptr())) throw nb::type_error("expected a NumPy event array");
        auto* a = reinterpret_cast<PyArrayObject*>(array.ptr());
        auto* data = static_cast<const Event*>(PyArray_DATA(a));
        if (PyArray_NDIM(a) != 1 || !PyArray_IS_C_CONTIGUOUS(a) ||
            reinterpret_cast<std::uintptr_t>(data) % alignof(Event))
            throw nb::value_error("expected a 1-D C-contiguous, aligned event array");

        nb::dict fields;
        fields["names"] = nb::make_tuple("ev", "exch_ts", "local_ts", "px", "qty", "order_id", "ival", "fval");
        fields["formats"] = nb::make_tuple("u8", "i8", "i8", "f8", "f8", "u8", "i8", "f8");
        auto dtype = nb::module_::import_("numpy").attr("dtype")(fields);
        if (!nb::handle(reinterpret_cast<PyObject*>(PyArray_DESCR(a))).equal(dtype))
            throw nb::type_error("expected the native 64-byte hftbacktest event_dtype");
        rows = {data, static_cast<std::size_t>(PyArray_DIM(a, 0))};
    }
};

nb::list order_snapshot(const auto& orders) {
    nb::list result;
    for (const Order& order : orders)
        result.append(nb::cast(order, nb::rv_policy::copy));
    return result;
}

template<class Queue>
struct Single {
    EventSource source;
    Backtest<Queue, ConstantLatency> bt;

    template<class Data>
    Single(AssetConfig cfg, Data data, Ns entry, Ns resp)
        : source(std::move(data)), bt(source.rows, cfg, {entry, nonnegative(resp)}) {}
};

template<class Queue>
struct Multi {
    std::vector<EventSource> sources;
    MultiAssetBacktest<Queue, ConstantLatency> bt;
    Multi() = default;
    Multi(const Multi&) = delete;

    template<class Data>
    std::size_t add(AssetConfig cfg, Data data, Ns entry, Ns resp) {
        if (bt.current_timestamp() != detail::never)
            throw nb::value_error("assets must be registered before replay starts");
        ConstantLatency latency{entry, nonnegative(resp)};
        const auto rows = sources.emplace_back(std::move(data)).rows;
        try {
            return bt.add_asset(cfg, rows.data(), rows.size(), Queue{}, latency);
        } catch (...) {
            sources.pop_back();
            throw;
        }
    }

    std::size_t index(std::size_t i) const {
        if (i >= bt.asset_count()) throw nb::index_error("asset index out of range");
        return i;
    }
};

template<class Queue>
void bind_single(nb::module_& m, const char* name) {
    using S = Single<Queue>;
    nb::class_<S>(m, name)
        .def(nb::init<AssetConfig, Path, Ns, Ns>(), "config"_a, "data"_a, "entry_latency_ns"_a, "resp_latency_ns"_a)
        .def(nb::init<AssetConfig, nb::object, Ns, Ns>(), "config"_a, "data"_a, "entry_latency_ns"_a, "resp_latency_ns"_a)
        .def("elapse", [](S& s, Ns ns) { return s.bt.elapse(nonnegative(ns)) == 0; }, "duration_ns"_a)
        .def_prop_ro("current_timestamp", [](const S& s) { return s.bt.current_timestamp(); })
        .def_prop_ro("position", [](const S& s) { return s.bt.position(); })
        .def_prop_ro("best_bid", [](const S& s) { return s.bt.depth().best_bid(); })
        .def_prop_ro("best_ask", [](const S& s) { return s.bt.depth().best_ask(); })
        .def_prop_ro("best_bid_tick", [](const S& s) { return s.bt.depth().best_bid_tick(); })
        .def_prop_ro("best_ask_tick", [](const S& s) { return s.bt.depth().best_ask_tick(); })
        .def("bid_qty_at_tick", [](const S& s, Ns tick) { return s.bt.depth().bid_qty_at_tick(tick); }, "tick"_a)
        .def("ask_qty_at_tick", [](const S& s, Ns tick) { return s.bt.depth().ask_qty_at_tick(tick); }, "tick"_a)
        .def_prop_ro("state_values", [](const S& s) { return s.bt.state_values(); })
        .def("submit_buy_order", [](S& s, Id id, double px, double qty, TimeInForce tif) {
            return s.bt.submit_buy_order(id, px, qty, tif);
        }, "order_id"_a, "price"_a, "qty"_a, "tif"_a)
        .def("submit_sell_order", [](S& s, Id id, double px, double qty, TimeInForce tif) {
            return s.bt.submit_sell_order(id, px, qty, tif);
        }, "order_id"_a, "price"_a, "qty"_a, "tif"_a)
        .def("cancel", [](S& s, Id id) { return s.bt.cancel(id); }, "order_id"_a)
        .def("clear_inactive_orders", [](S& s) { s.bt.clear_inactive_orders(); })
        .def_prop_ro("active_orders", [](const S& s) { return order_snapshot(s.bt.orders()); },
                     "Snapshot of retained orders, including terminal orders until cleared.");
}

template<class Queue>
void bind_multi(nb::module_& m, const char* name) {
    using M = Multi<Queue>;
    nb::class_<M>(m, name)
        .def(nb::init<>())
        .def("add_asset_file", &M::template add<Path>, "config"_a, "path"_a, "entry_latency_ns"_a, "resp_latency_ns"_a)
        .def("add_asset_array", &M::template add<nb::object>, "config"_a, "ndarray"_a, "entry_latency_ns"_a, "resp_latency_ns"_a)
        .def("elapse", [](M& m, Ns ns) { return m.bt.elapse(nonnegative(ns)); }, "duration_ns"_a)
        .def("wait_next_feed", [](M& m, bool resp, Ns ns) { return m.bt.wait_next_feed(resp, nonnegative(ns)); },
             "include_order_resp"_a, "timeout_ns"_a, "Return 0 on timeout, 1 at EOF, 2 for feed, or 3 for an order response.")
        .def_prop_ro("current_timestamp", [](const M& m) { return m.bt.current_timestamp(); })
        .def_prop_ro("asset_count", [](const M& m) { return m.bt.asset_count(); })
        .def("position", [](const M& m, std::size_t i) { return m.bt.position(m.index(i)); }, "i"_a)
        .def("best_bid", [](const M& m, std::size_t i) { return m.bt.depth(m.index(i)).best_bid(); }, "i"_a)
        .def("best_ask", [](const M& m, std::size_t i) { return m.bt.depth(m.index(i)).best_ask(); }, "i"_a)
        .def("best_bid_tick", [](const M& m, std::size_t i) { return m.bt.depth(m.index(i)).best_bid_tick(); }, "i"_a)
        .def("best_ask_tick", [](const M& m, std::size_t i) { return m.bt.depth(m.index(i)).best_ask_tick(); }, "i"_a)
        .def("state_values", [](const M& m, std::size_t i) { return m.bt.state_values(m.index(i)); }, "i"_a)
        .def("submit_buy_order", [](M& m, std::size_t i, Id id, double px, double qty, TimeInForce tif) {
            return m.bt.submit_buy_order(m.index(i), id, px, qty, tif);
        }, "i"_a, "order_id"_a, "price"_a, "qty"_a, "tif"_a)
        .def("submit_sell_order", [](M& m, std::size_t i, Id id, double px, double qty, TimeInForce tif) {
            return m.bt.submit_sell_order(m.index(i), id, px, qty, tif);
        }, "i"_a, "order_id"_a, "price"_a, "qty"_a, "tif"_a)
        .def("cancel", [](M& m, std::size_t i, Id id) { return m.bt.cancel(m.index(i), id); }, "i"_a, "order_id"_a)
        .def("clear_inactive_orders", [](M& m, std::size_t i) { m.bt.clear_inactive_orders(m.index(i)); }, "i"_a)
        .def("active_orders", [](const M& m, std::size_t i) { return order_snapshot(m.bt.orders(m.index(i))); }, "i"_a)
        .def("total_equity", [](const M& m) { return m.bt.total_equity(); });
}
} // namespace

NB_MODULE(xhftbacktest, m) {
    if (PyArray_ImportNumPyAPI() < 0) throw nb::python_error();
    m.doc() = "Zero-copy L2 event replay with constant order latency.";
    nb::enum_<TimeInForce>(m, "TimeInForce")
        .value("GTC", GTC).value("GTX", GTX).value("FOK", FOK).value("IOC", IOC);
    nb::enum_<status::Code>(m, "OrderStatus")
        .value("NONE", status::NONE).value("NEW", status::NEW).value("EXPIRED", status::EXPIRED)
        .value("FILLED", status::FILLED).value("CANCELED", status::CANCELED).value("PARTIALLY_FILLED", status::PARTIALLY_FILLED);
    nb::enum_<Side>(m, "Side").value("BUY", Side::BUY).value("SELL", Side::SELL);

    nb::class_<AssetConfig>(m, "AssetConfig")
        .def(nb::init<>())
        .def(nb::init<double, double, Ns, Ns, double, double, double>(),
             "tick_size"_a, "lot_size"_a, "roi_lb_tick"_a, "roi_ub_tick"_a,
             "contract_size"_a = 1., "maker_fee"_a = 0., "taker_fee"_a = 0.)
        .def_rw("tick_size", &AssetConfig::tick_size)
        .def_rw("lot_size", &AssetConfig::lot_size)
        .def_rw("roi_lb_tick", &AssetConfig::roi_lb_tick)
        .def_rw("roi_ub_tick", &AssetConfig::roi_ub_tick)
        .def_rw("contract_size", &AssetConfig::contract_size)
        .def_rw("maker_fee", &AssetConfig::maker_fee)
        .def_rw("taker_fee", &AssetConfig::taker_fee);

    nb::class_<StateValues>(m, "StateValues")
        .def_ro("position", &StateValues::position)
        .def_ro("balance", &StateValues::balance)
        .def_ro("fee", &StateValues::fee)
        .def_ro("num_trades", &StateValues::num_trades)
        .def_ro("trading_volume", &StateValues::trading_volume)
        .def_ro("trading_value", &StateValues::trading_value);
    nb::class_<Order>(m, "Order")
        .def_ro("order_id", &Order::order_id)
        .def_ro("price_tick", &Order::price_tick)
        .def_ro("qty", &Order::qty)
        .def_ro("leaves_qty", &Order::leaves_qty)
        .def_ro("exec_qty", &Order::exec_qty)
        .def_ro("exec_price_tick", &Order::exec_price_tick)
        .def_prop_ro("side", [](const Order& o) { return static_cast<Side>(o.side); })
        .def_prop_ro("tif", [](const Order& o) { return static_cast<TimeInForce>(o.tif); })
        .def_prop_ro("status", [](const Order& o) { return static_cast<status::Code>(o.status); })
        .def_ro("req", &Order::req)
        .def_ro("q_ahead", &Order::q_ahead)
        .def_ro("exch_ts", &Order::exch_ts)
        .def_ro("local_ts", &Order::local_ts)
        .def_ro("cum_trade_qty", &Order::cum_trade_qty)
        .def_ro("maker", &Order::maker);

    bind_single<RiskAdverseQueueModel>(m, "BacktestRiskAdverse");
    bind_single<PowerProbQueueModel<3>>(m, "BacktestPowerProb");
    bind_multi<RiskAdverseQueueModel>(m, "MultiAssetRiskAdverse");
    bind_multi<PowerProbQueueModel<3>>(m, "MultiAssetPowerProb");
}
