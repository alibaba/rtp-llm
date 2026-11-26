#pragma once

#include <cstdint>
#include <limits>

namespace rtp_llm {

// Decode capture sizes are strictly below INT_MAX. A MAX across DP ranks
// selects the largest required bucket, or eager if any rank cannot replay.
inline constexpr int kBatchInvariantEagerVote = std::numeric_limits<int>::max();

inline constexpr int batchInvariantDecodeGraphVote(
    int local_bucket, bool has_context_request, int64_t token_dims, int64_t token_count, int64_t decode_requests) {
    return local_bucket > 0 && !has_context_request && token_dims == 1 && token_count == decode_requests ?
               local_bucket :
               kBatchInvariantEagerVote;
}

}  // namespace rtp_llm
