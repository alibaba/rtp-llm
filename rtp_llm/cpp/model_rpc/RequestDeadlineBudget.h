#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>

namespace rtp_llm {

struct RequestDeadlineBudget {
    int64_t begin_time_us    = 0;
    int64_t timeout_ms       = 0;
    int64_t deadline_unix_us = 0;
    bool    has_deadline     = false;
    bool    expired          = false;
};

inline int64_t clampRequestDeadlineToInt64(__int128 value) {
    if (value > std::numeric_limits<int64_t>::max()) {
        return std::numeric_limits<int64_t>::max();
    }
    if (value < std::numeric_limits<int64_t>::min()) {
        return std::numeric_limits<int64_t>::min();
    }
    return static_cast<int64_t>(value);
}

inline int64_t clampRequestDeadlineTimeoutMs(__int128 value) {
    return std::max<int64_t>(1,
                             std::min<int64_t>(std::numeric_limits<int>::max(),
                                               clampRequestDeadlineToInt64(value)));
}

inline int64_t requestDeadlineUnixUs(std::chrono::system_clock::time_point deadline) {
    if (deadline == std::chrono::system_clock::time_point::max()) {
        return 0;
    }
    return clampRequestDeadlineToInt64(
        std::chrono::duration_cast<std::chrono::microseconds>(deadline.time_since_epoch()).count());
}

inline bool requestDeadlineReached(const RequestDeadlineBudget& budget, int64_t now_unix_us) {
    return budget.has_deadline && now_unix_us >= budget.deadline_unix_us;
}

inline bool requestDeadlineReached(int64_t begin_time_us, int64_t timeout_ms, int64_t now_unix_us) {
    if (timeout_ms <= 0) {
        return false;
    }
    const auto deadline_us = clampRequestDeadlineToInt64(
        static_cast<__int128>(begin_time_us) + static_cast<__int128>(timeout_ms) * 1000);
    return now_unix_us >= deadline_us;
}

inline int64_t deriveBatchItemDeadlineUnixUs(int64_t batch_deadline_unix_us,
                                             int64_t batch_timeout_ms,
                                             int64_t item_timeout_ms) {
    if (batch_deadline_unix_us <= 0 || batch_timeout_ms <= 0 || item_timeout_ms <= 0
        || item_timeout_ms >= batch_timeout_ms) {
        return batch_deadline_unix_us;
    }
    return clampRequestDeadlineToInt64(static_cast<__int128>(batch_deadline_unix_us)
                                       - static_cast<__int128>(batch_timeout_ms - item_timeout_ms) * 1000);
}

inline RequestDeadlineBudget
makeRequestDeadlineBudget(int64_t request_deadline_unix_ms,
                          int64_t relative_timeout_ms,
                          int64_t now_unix_us,
                          int64_t authoritative_deadline_unix_us = 0) {
    if (request_deadline_unix_ms < 0) {
        return {now_unix_us,
                clampRequestDeadlineTimeoutMs(relative_timeout_ms),
                now_unix_us,
                true,
                true};
    }
    if (request_deadline_unix_ms == 0 && authoritative_deadline_unix_us <= 0) {
        return {now_unix_us, relative_timeout_ms, 0, false, false};
    }

    auto deadline_us = authoritative_deadline_unix_us > 0 ?
                           authoritative_deadline_unix_us :
                           clampRequestDeadlineToInt64(static_cast<__int128>(request_deadline_unix_ms) * 1000);
    if (relative_timeout_ms <= 0) {
        if (deadline_us <= now_unix_us) {
            return {clampRequestDeadlineToInt64(static_cast<__int128>(now_unix_us) - 1000),
                    1,
                    deadline_us,
                    true,
                    true};
        }
        const auto remaining_us = static_cast<__int128>(deadline_us) - now_unix_us;
        const auto timeout_ms   = clampRequestDeadlineTimeoutMs((remaining_us + 999) / 1000);
        return {now_unix_us, timeout_ms, deadline_us, true, false};
    }

    const auto relative_deadline_us = clampRequestDeadlineToInt64(
        static_cast<__int128>(now_unix_us) + static_cast<__int128>(relative_timeout_ms) * 1000);
    deadline_us = std::min(deadline_us, relative_deadline_us);
    const auto begin_time_us = clampRequestDeadlineToInt64(
        static_cast<__int128>(deadline_us) - static_cast<__int128>(relative_timeout_ms) * 1000);
    return {begin_time_us, relative_timeout_ms, deadline_us, true, deadline_us <= now_unix_us};
}

}  // namespace rtp_llm
