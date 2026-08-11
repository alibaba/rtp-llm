#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>

namespace rtp_llm {

using RemoteLoadSystemClock = std::chrono::system_clock;
using RemoteLoadSteadyClock = std::chrono::steady_clock;

struct RemoteLoadBudget {
    int64_t                                   deadline_unix_ms = 0;
    int64_t                                   remaining_ms     = 0;
    RemoteLoadSystemClock::time_point         system_deadline;
    RemoteLoadSteadyClock::time_point         steady_deadline;

    bool expired() const noexcept {
        return remaining_ms <= 0;
    }
};

inline bool canAdmitRemoteLoad(const RemoteLoadBudget& budget, bool cancelled) noexcept {
    return !budget.expired() && !cancelled;
}

inline int64_t saturatingDeadlineUnixMs(int64_t start_unix_ms, int64_t timeout_ms) noexcept {
    if (timeout_ms <= 0) {
        return start_unix_ms;
    }
    if (start_unix_ms > std::numeric_limits<int64_t>::max() - timeout_ms) {
        return std::numeric_limits<int64_t>::max();
    }
    return start_unix_ms + timeout_ms;
}

inline int64_t remoteLoadUnixMillis(RemoteLoadSystemClock::time_point time_point) noexcept {
    return std::chrono::duration_cast<std::chrono::milliseconds>(time_point.time_since_epoch()).count();
}

inline RemoteLoadSteadyClock::time_point
saturatingSteadyDeadline(RemoteLoadSteadyClock::time_point now, int64_t remaining_ms) noexcept {
    if (remaining_ms <= 0) {
        return now;
    }

    using FloatMilliseconds = std::chrono::duration<long double, std::milli>;
    const auto max_time = RemoteLoadSteadyClock::time_point::max();
    const auto headroom_ms = FloatMilliseconds(max_time.time_since_epoch()).count()
                             - FloatMilliseconds(now.time_since_epoch()).count();
    const auto duration_limit_ms = FloatMilliseconds(RemoteLoadSteadyClock::duration::max()).count();
    if (static_cast<long double>(remaining_ms) >= headroom_ms
        || static_cast<long double>(remaining_ms) >= duration_limit_ms) {
        return max_time;
    }
    return now + std::chrono::duration_cast<RemoteLoadSteadyClock::duration>(std::chrono::milliseconds(remaining_ms));
}

inline RemoteLoadBudget makeRemoteLoadBudget(
    int64_t                           absolute_deadline_unix_ms,
    RemoteLoadSystemClock::time_point parent_deadline,
    RemoteLoadSystemClock::time_point system_now,
    RemoteLoadSteadyClock::time_point steady_now,
    int64_t relative_timeout_cap_ms = std::numeric_limits<int64_t>::max(),
    bool    parent_deadline_authoritative = false) noexcept {
    const auto system_now_unix_ms = remoteLoadUnixMillis(system_now);
    const auto parent_deadline_unix_ms = remoteLoadUnixMillis(parent_deadline);
    auto       effective_deadline      = absolute_deadline_unix_ms;
    if (parent_deadline != RemoteLoadSystemClock::time_point::max()) {
        effective_deadline = parent_deadline_authoritative ?
                                 parent_deadline_unix_ms :
                                 std::min(effective_deadline, parent_deadline_unix_ms);
    }
    if (relative_timeout_cap_ms != std::numeric_limits<int64_t>::max()) {
        effective_deadline = std::min(
            effective_deadline, saturatingDeadlineUnixMs(system_now_unix_ms, relative_timeout_cap_ms));
    }

    RemoteLoadBudget budget;
    budget.deadline_unix_ms = effective_deadline;
    budget.system_deadline  = system_now;
    budget.steady_deadline  = steady_now;
    if (effective_deadline <= system_now_unix_ms) {
        return budget;
    }

    budget.remaining_ms = effective_deadline - system_now_unix_ms;
    budget.system_deadline = RemoteLoadSystemClock::time_point(
        std::chrono::duration_cast<RemoteLoadSystemClock::duration>(std::chrono::milliseconds(effective_deadline)));
    budget.steady_deadline = saturatingSteadyDeadline(steady_now, budget.remaining_ms);
    return budget;
}

}  // namespace rtp_llm
