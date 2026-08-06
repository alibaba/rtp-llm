#pragma once

#include <atomic>
#include <cstdint>
#include <utility>

namespace rtp_llm {

inline std::pair<uint64_t, bool> recordRateLimitedFallback(std::atomic<uint64_t>& counter) {
    const uint64_t count = counter.fetch_add(1, std::memory_order_relaxed) + 1;
    return {count, (count & (count - 1)) == 0};
}

}  // namespace rtp_llm
