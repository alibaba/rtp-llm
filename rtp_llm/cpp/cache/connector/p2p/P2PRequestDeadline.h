#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>

namespace rtp_llm {

inline int64_t normalizeP2PRequestDeadline(int64_t request_deadline_ms, int64_t now_ms, int64_t fallback_ttl_ms) {
    if (request_deadline_ms <= 0 || request_deadline_ms == std::numeric_limits<int64_t>::max()) {
        return now_ms + fallback_ttl_ms;
    }
    return std::max(request_deadline_ms, now_ms);
}

inline int64_t p2pResourceHoldDeadline(int64_t request_deadline_ms, int64_t now_ms, int64_t hold_ms) {
    constexpr int64_t max_lifetime_ms = 3600000;
    return std::min({request_deadline_ms, now_ms + hold_ms, now_ms + max_lifetime_ms});
}

}  // namespace rtp_llm
