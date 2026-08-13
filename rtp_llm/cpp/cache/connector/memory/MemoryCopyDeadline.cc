#include "rtp_llm/cpp/cache/connector/memory/MemoryCopyDeadline.h"

#include <algorithm>
#include <limits>

namespace rtp_llm {
namespace {

MemoryCopyDeadlineDecision evaluate(int64_t operation_deadline_unix_ms,
                                    int64_t requested_retention_ms,
                                    int64_t safety_window_ms,
                                    int64_t now_unix_ms,
                                    bool    reject_expired) {
    if (operation_deadline_unix_ms <= 0) {
        return {std::chrono::milliseconds(0), "memory copy operation deadline is invalid"};
    }
    if (!MemoryCopyDeadline::validWireDuration(requested_retention_ms)
        || !MemoryCopyDeadline::validWireDuration(safety_window_ms) || now_unix_ms < 0) {
        return {std::chrono::milliseconds(0), "memory copy retention inputs are invalid"};
    }
    if (operation_deadline_unix_ms > now_unix_ms
        && operation_deadline_unix_ms - now_unix_ms > MemoryCopyDeadline::kMaxWireDurationMs) {
        return {std::chrono::milliseconds(0), "memory copy operation deadline is too far in the future"};
    }
    const int64_t tombstone_base = std::max(operation_deadline_unix_ms, now_unix_ms);
    if (tombstone_base > std::numeric_limits<int64_t>::max() - safety_window_ms) {
        return {std::chrono::milliseconds(0), "memory copy operation deadline overflows its safety window"};
    }
    if (reject_expired && now_unix_ms >= operation_deadline_unix_ms) {
        return {std::chrono::milliseconds(0), "memory copy operation deadline has expired"};
    }

    const int64_t tombstone_deadline = tombstone_base + safety_window_ms;
    const int64_t remaining_ms       = tombstone_deadline - now_unix_ms;
    return {std::chrono::milliseconds(std::max(requested_retention_ms, remaining_ms)), {}};
}

}  // namespace

int64_t MemoryCopyDeadline::unixMillisNow() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

int64_t MemoryCopyDeadline::make(int64_t now_unix_ms, int64_t timeout_ms) {
    if (now_unix_ms < 0 || !validWireDuration(timeout_ms)
        || now_unix_ms > std::numeric_limits<int64_t>::max() - timeout_ms) {
        return 0;
    }
    return now_unix_ms + timeout_ms;
}

int64_t MemoryCopyDeadline::resolve(int64_t now_unix_ms,
                                    int64_t local_timeout_ms,
                                    int64_t request_deadline_unix_ms) {
    const int64_t local_deadline = make(now_unix_ms, local_timeout_ms);
    if (local_deadline <= 0) {
        return 0;
    }
    if (request_deadline_unix_ms <= 0) {
        return local_deadline;
    }
    if (request_deadline_unix_ms <= now_unix_ms) {
        return 0;
    }
    return std::min(local_deadline, request_deadline_unix_ms);
}

int64_t MemoryCopyDeadline::rpcTimeout(int64_t operation_deadline_unix_ms,
                                       int64_t local_timeout_ms,
                                       int64_t now_unix_ms) {
    if (!validWireDuration(local_timeout_ms) || operation_deadline_unix_ms <= now_unix_ms) {
        return 0;
    }
    return std::min(local_timeout_ms, operation_deadline_unix_ms - now_unix_ms);
}

bool MemoryCopyDeadline::validWireDuration(int64_t duration_ms) {
    return duration_ms > 0 && duration_ms <= kMaxWireDurationMs;
}

MemoryCopyDeadlineDecision MemoryCopyDeadline::evaluateCopy(int64_t operation_deadline_unix_ms,
                                                            int64_t requested_retention_ms,
                                                            int64_t safety_window_ms,
                                                            int64_t now_unix_ms) {
    return evaluate(operation_deadline_unix_ms,
                    requested_retention_ms,
                    safety_window_ms,
                    now_unix_ms,
                    /*reject_expired=*/true);
}

MemoryCopyDeadlineDecision MemoryCopyDeadline::evaluateQuiesce(int64_t operation_deadline_unix_ms,
                                                               int64_t requested_retention_ms,
                                                               int64_t safety_window_ms,
                                                               int64_t now_unix_ms) {
    return evaluate(operation_deadline_unix_ms,
                    requested_retention_ms,
                    safety_window_ms,
                    now_unix_ms,
                    /*reject_expired=*/false);
}

}  // namespace rtp_llm
