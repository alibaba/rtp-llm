#pragma once

#include "rtp_llm/cpp/cache/LinearTpCacheLayout.h"

namespace rtp_llm {

// Packed MHA payloads store all K heads followed by all V heads. A one-head
// producer therefore needs two independent slices, including for FP8 scales.
inline std::vector<LinearTpCacheSlice>
mhaTpCacheSlices(size_t k_bytes, size_t v_bytes, size_t destination_heads, size_t head_id) {
    if (destination_heads == 0 || head_id >= destination_heads || k_bytes == 0 || v_bytes == 0
        || k_bytes % destination_heads || v_bytes % destination_heads) {
        throw std::invalid_argument("invalid MHA TP cache layout");
    }
    if (k_bytes > std::numeric_limits<size_t>::max() - v_bytes) {
        throw std::overflow_error("MHA TP cache size overflow");
    }
    const auto source_k = k_bytes / destination_heads;
    const auto source_v = v_bytes / destination_heads;
    return {{0, head_id * source_k, source_k}, {source_k, k_bytes + head_id * source_v, source_v}};
}

// Contiguous GQA weight splits assign one logical KV head to each contiguous
// producer-rank group. Select its first rank, leaving replicas untouched.
inline bool mhaTpPeerOwnsHead(size_t peers, size_t heads, size_t peer) {
    if (peers == 0 || heads == 0 || peers % heads || peer >= peers) {
        throw std::invalid_argument("invalid MHA TP peer geometry");
    }
    return peer % (peers / heads) == 0;
}

}  // namespace rtp_llm
