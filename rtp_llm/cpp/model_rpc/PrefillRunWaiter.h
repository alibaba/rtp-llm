#pragma once

#include <algorithm>
#include <chrono>
#include <thread>
#include <utility>

namespace rtp_llm {

using PrefillRunSteadyClock = std::chrono::steady_clock;
using PrefillRunSystemClock = std::chrono::system_clock;

struct PrefillRunDeadline {
    PrefillRunSteadyClock::time_point value;
    bool                              limited_by_server_context = false;
};

enum class PrefillRunWaitResult {
    Ready,
    StreamError,
    Cancelled,
    DeadlineExceeded,
};

namespace prefill_run_waiter_detail {

inline PrefillRunSteadyClock::time_point
saturatingAdd(PrefillRunSteadyClock::time_point now, std::chrono::milliseconds delay) {
    if (delay <= std::chrono::milliseconds::zero()) {
        return now;
    }
    const auto max_delay =
        std::chrono::duration_cast<std::chrono::milliseconds>(PrefillRunSteadyClock::time_point::max() - now);
    if (delay >= max_delay) {
        return PrefillRunSteadyClock::time_point::max();
    }
    return now + std::chrono::duration_cast<PrefillRunSteadyClock::duration>(delay);
}

}  // namespace prefill_run_waiter_detail

inline PrefillRunDeadline makePrefillRunDeadline(PrefillRunSteadyClock::time_point steady_now,
                                                 std::chrono::milliseconds         configured_timeout,
                                                 PrefillRunSystemClock::time_point system_now,
                                                 PrefillRunSystemClock::time_point server_deadline) {
    const auto configured_deadline = prefill_run_waiter_detail::saturatingAdd(steady_now, configured_timeout);
    if (server_deadline == PrefillRunSystemClock::time_point::max()) {
        return {configured_deadline, false};
    }

    const auto server_remaining =
        server_deadline <= system_now ? std::chrono::milliseconds::zero() :
                                        std::chrono::duration_cast<std::chrono::milliseconds>(server_deadline
                                                                                             - system_now);
    const auto steady_server_deadline = prefill_run_waiter_detail::saturatingAdd(steady_now, server_remaining);
    if (steady_server_deadline <= configured_deadline) {
        return {steady_server_deadline, true};
    }
    return {configured_deadline, false};
}

template<typename HasStreamError,
         typename IsReady,
         typename IsCancelled,
         typename Now,
         typename WaitUntil>
PrefillRunWaitResult waitForPrefillRun(HasStreamError&&                   has_stream_error,
                                       IsReady&&                          is_ready,
                                       IsCancelled&&                      is_cancelled,
                                       PrefillRunSteadyClock::time_point deadline,
                                       std::chrono::milliseconds         poll_interval,
                                       Now&&                              now,
                                       WaitUntil&&                        wait_until) {
    if (poll_interval <= std::chrono::milliseconds::zero()) {
        poll_interval = std::chrono::milliseconds(1);
    }
    while (true) {
        if (has_stream_error()) {
            return PrefillRunWaitResult::StreamError;
        }

        const auto current_time = now();
        if (current_time >= deadline) {
            return PrefillRunWaitResult::DeadlineExceeded;
        }
        if (is_cancelled()) {
            return PrefillRunWaitResult::Cancelled;
        }
        if (is_ready()) {
            return PrefillRunWaitResult::Ready;
        }

        const auto next_poll = prefill_run_waiter_detail::saturatingAdd(current_time, poll_interval);
        wait_until(std::min(deadline, next_poll));
    }
}

template<typename HasStreamError, typename IsReady, typename IsCancelled>
PrefillRunWaitResult waitForPrefillRun(HasStreamError&&                   has_stream_error,
                                       IsReady&&                          is_ready,
                                       IsCancelled&&                      is_cancelled,
                                       PrefillRunSteadyClock::time_point deadline,
                                       std::chrono::milliseconds         poll_interval = std::chrono::milliseconds(1)) {
    return waitForPrefillRun(std::forward<HasStreamError>(has_stream_error),
                             std::forward<IsReady>(is_ready),
                             std::forward<IsCancelled>(is_cancelled),
                             deadline,
                             poll_interval,
                             []() { return PrefillRunSteadyClock::now(); },
                             [](PrefillRunSteadyClock::time_point wakeup) { std::this_thread::sleep_until(wakeup); });
}

}  // namespace rtp_llm
