#ifndef XHFTBACKTEST_SWEEP_HPP
#define XHFTBACKTEST_SWEEP_HPP

#include "event.hpp"
#include <algorithm>
#include <atomic>
#include <exception>
#include <mutex>
#include <span>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

namespace xhftbacktest {

// Each invocation owns its engine. The mapping and configs are borrowed until
// all workers join. Results are indexed in the caller's configuration order.
template<class Config, class Replay>
auto sweep(std::span<const Event> events, std::span<const Config> configs,
           std::size_t workers, Replay replay) {
    using Result = std::invoke_result_t<Replay&, std::span<const Event>, const Config&>;
    static_assert(!std::is_same_v<Result, bool>, "independent result slots required");
    if (workers == 0) throw std::invalid_argument("sweep needs at least one worker");
    std::vector<Result> results(configs.size());
    std::atomic_size_t next{};
    std::exception_ptr error;
    std::mutex error_mutex;
    {
        std::vector<std::jthread> pool;
        workers = std::min(workers, configs.size());
        pool.reserve(workers);
        for (std::size_t i = 0; i < workers; ++i) {
            pool.emplace_back([&, replay]() mutable {
                try {
                    for (;;) {
                        const auto index = next.fetch_add(1, std::memory_order_relaxed);
                        if (index >= configs.size()) return;
                        results[index] = replay(events, configs[index]);
                    }
                } catch (...) {
                    std::lock_guard lock(error_mutex);
                    if (!error) error = std::current_exception();
                    next.store(configs.size(), std::memory_order_relaxed);
                }
            });
        }
    }
    if (error) std::rethrow_exception(error);
    return results;
}

} // namespace xhftbacktest
#endif
