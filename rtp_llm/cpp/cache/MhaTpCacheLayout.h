#pragma once

#include "rtp_llm/cpp/cache/LinearTpCacheLayout.h"

namespace rtp_llm {

// Each kernel page stores all K heads followed by all V heads. Preserve this
// order independently in every page of a physical block, including FP8 scales.
inline std::vector<LinearTpCacheSlice>
mhaTpCacheSlices(size_t k_bytes, size_t v_bytes, size_t destination_heads, size_t head_id, size_t kernel_pages = 1) {
    if (destination_heads == 0 || head_id >= destination_heads || k_bytes == 0 || v_bytes == 0
        || k_bytes % destination_heads || v_bytes % destination_heads || kernel_pages == 0
        || (k_bytes / destination_heads) % kernel_pages || (v_bytes / destination_heads) % kernel_pages) {
        throw std::invalid_argument("invalid MHA TP cache layout");
    }
    if (k_bytes > std::numeric_limits<size_t>::max() - v_bytes) {
        throw std::overflow_error("MHA TP cache size overflow");
    }
    const auto                      page_k   = k_bytes / kernel_pages;
    const auto                      page_v   = v_bytes / kernel_pages;
    const auto                      source_k = page_k / destination_heads;
    const auto                      source_v = page_v / destination_heads;
    std::vector<LinearTpCacheSlice> slices;
    for (size_t page = 0; page < kernel_pages; ++page) {
        const auto source      = page * (source_k + source_v);
        const auto destination = page * (page_k + page_v);
        slices.push_back({source, destination + head_id * source_k, source_k});
        slices.push_back({source + source_k, destination + page_k + head_id * source_v, source_v});
    }
    return slices;
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
