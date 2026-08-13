#pragma once

#include <chrono>
#include <cstdint>
#include <string>

namespace rtp_llm {

struct MemoryCopyDeadlineDecision {
    std::chrono::milliseconds retention{0};
    std::string               error;

    explicit operator bool() const {
        return error.empty();
    }
};

class MemoryCopyDeadline {
public:
    static constexpr int64_t kMaxWireDurationMs = 60 * 60 * 1000;

    static int64_t unixMillisNow();
    static int64_t make(int64_t now_unix_ms, int64_t timeout_ms);
    static int64_t resolve(int64_t now_unix_ms, int64_t local_timeout_ms, int64_t request_deadline_unix_ms);
    static int64_t rpcTimeout(int64_t operation_deadline_unix_ms, int64_t local_timeout_ms, int64_t now_unix_ms);
    static bool    validWireDuration(int64_t duration_ms);

    static MemoryCopyDeadlineDecision evaluateCopy(int64_t operation_deadline_unix_ms,
                                                   int64_t requested_retention_ms,
                                                   int64_t safety_window_ms,
                                                   int64_t now_unix_ms);
    static MemoryCopyDeadlineDecision evaluateQuiesce(int64_t operation_deadline_unix_ms,
                                                      int64_t requested_retention_ms,
                                                      int64_t safety_window_ms,
                                                      int64_t now_unix_ms);
};

}  // namespace rtp_llm
