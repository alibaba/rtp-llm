#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>

namespace rtp_llm {

constexpr int64_t kDefaultRpcTimeoutMs = 3600 * 1000;

inline int64_t normalizeRpcTimeoutMs(int64_t request_timeout_ms, int64_t max_rpc_timeout_ms) {
    if (request_timeout_ms > 0) {
        return request_timeout_ms;
    }
    if (max_rpc_timeout_ms > 0) {
        return max_rpc_timeout_ms;
    }
    return kDefaultRpcTimeoutMs;
}

inline int32_t clampRpcTimeoutMsToInt32(int64_t timeout_ms) {
    return static_cast<int32_t>(
        std::clamp<int64_t>(timeout_ms, 0, static_cast<int64_t>(std::numeric_limits<int32_t>::max())));
}

}  // namespace rtp_llm
