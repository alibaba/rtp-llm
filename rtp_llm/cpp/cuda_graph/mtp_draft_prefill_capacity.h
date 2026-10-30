#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace rtp_llm {

inline int mtpDraftPrefillRequestCapacity(
    int token_capacity, int max_requests, int query_width, int sequence_parallel_size) {
    if (token_capacity <= 0 || max_requests <= 0 || query_width <= 0 || sequence_parallel_size <= 0) {
        throw std::invalid_argument("MTP draft prefill capacities, query width and SP size must be positive");
    }
    // Every MTP draft update retains Q physical tokens per request.
    const int64_t alignment = sequence_parallel_size / std::gcd(sequence_parallel_size, query_width);
    const int64_t requests  = (static_cast<int64_t>(token_capacity) + query_width - 1) / query_width;
    const int64_t rounded   = (requests + alignment - 1) / alignment * alignment;
    if (rounded > std::numeric_limits<int>::max() || rounded * query_width > std::numeric_limits<int>::max()) {
        throw std::overflow_error("MTP draft prefill rectangular capacity exceeds int range");
    }
    if (rounded > max_requests) {
        throw std::invalid_argument("MTP draft prefill bucket exceeds request capacity");
    }
    return static_cast<int>(rounded);
}

inline int selectMtpDraftPrefillTokenCapacity(const std::vector<int>& buckets,
                                              int                     live_tokens,
                                              int                     live_requests,
                                              int                     max_requests,
                                              int                     query_width,
                                              int                     sequence_parallel_size) {
    if (live_tokens <= 0 || live_requests <= 0 || max_requests <= 0 || query_width <= 0
        || sequence_parallel_size <= 0) {
        throw std::invalid_argument(
            "MTP draft prefill live sizes, capacities, query width and SP size must be positive");
    }
    if (live_requests > max_requests) {
        return 0;
    }
    // The runner supplies sorted buckets. Start at the token lower bound so
    // selection starts at the smallest bucket that fits the physical tokens.
    if (!buckets.empty() && buckets.front() <= 0) {
        throw std::invalid_argument("MTP draft prefill token buckets must be positive");
    }
    for (auto it = std::lower_bound(buckets.begin(), buckets.end(), live_tokens); it != buckets.end(); ++it) {
        const int tokens = *it;
        if (mtpDraftPrefillRequestCapacity(tokens, max_requests, query_width, sequence_parallel_size)
            >= live_requests) {
            return tokens;
        }
    }
    return 0;
}

}  // namespace rtp_llm
