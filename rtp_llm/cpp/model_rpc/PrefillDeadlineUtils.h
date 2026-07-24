#pragma once

#include <algorithm>
#include <cstdint>

namespace rtp_llm {

// Compute remaining timeout (ms) after deducting the time already spent in
// prefill (request_begin_time_us -> current_time_us).
//   * timeout_ms <= 0  -> returned unchanged (treated as "no deadline").
//   * otherwise        -> max(0, timeout_ms - elapsed_ms),
//                        elapsed_ms = (current_time_us - request_begin_time_us) / 1000.
inline int64_t computeRemainingTimeoutMs(int64_t timeout_ms, int64_t request_begin_time_us, int64_t current_time_us) {
    if (timeout_ms <= 0) {
        return timeout_ms;
    }
    int64_t elapsed_ms = (current_time_us - request_begin_time_us) / 1000;
    return std::max(int64_t(0), timeout_ms - elapsed_ms);
}

}  // namespace rtp_llm
