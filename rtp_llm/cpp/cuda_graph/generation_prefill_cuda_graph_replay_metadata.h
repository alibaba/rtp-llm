#pragma once

#include <cstddef>
#include <cstdint>

namespace rtp_llm {

inline bool prepareGenerationPrefillCudaGraphReplayMetadata(int32_t* input_lengths,
                                                            size_t   input_lengths_capacity,
                                                            int32_t* cu_seqlens,
                                                            size_t   cu_seqlens_capacity,
                                                            int32_t* padding_offset,
                                                            size_t   padding_offset_capacity,
                                                            int      real_request_count,
                                                            int      max_request_count,
                                                            int      real_token_count,
                                                            int      token_capacity) {
    if (input_lengths == nullptr || cu_seqlens == nullptr || padding_offset == nullptr || real_request_count <= 0
        || real_request_count > max_request_count || max_request_count <= 0 || real_token_count <= 0
        || real_token_count > token_capacity || input_lengths_capacity < static_cast<size_t>(max_request_count + 1)
        || cu_seqlens_capacity < static_cast<size_t>(max_request_count + 2)
        || padding_offset_capacity < static_cast<size_t>(token_capacity)) {
        return false;
    }

    int computed_real_tokens = 0;
    for (int slot = 0; slot < real_request_count; ++slot) {
        if (input_lengths[slot] <= 0 || input_lengths[slot] > token_capacity - computed_real_tokens) {
            return false;
        }
        computed_real_tokens += input_lengths[slot];
    }
    if (computed_real_tokens != real_token_count) {
        return false;
    }

    // Canonical replay layout, byte-identical to the capture-time layout:
    // request slot 0 carries the full token capacity and every phantom slot is
    // empty. The fa2 ragged kernel's CTA->(batch, tile) schedule and its
    // split-KV completion counts are frozen at capture time; replaying with
    // different cu_seqlens content leaves live rows uncomputed (stale capture
    // output) and deadlocks the split-KV merge kernel. Padding rows therefore
    // participate as ordinary rows of request 0: their outputs are dropped by
    // the real-token-count slice and their KV writes route to reserved block 0.
    for (int slot = 0; slot < max_request_count; ++slot) {
        input_lengths[slot] = 0;
    }
    input_lengths[0]                     = token_capacity;
    input_lengths[max_request_count]     = 0;

    int packed_offset = 0;
    cu_seqlens[0]     = 0;
    for (int slot = 0; slot <= max_request_count; ++slot) {
        packed_offset += input_lengths[slot];
        cu_seqlens[slot + 1] = packed_offset;
    }
    std::fill_n(padding_offset, token_capacity, 0);
    return true;
}

}  // namespace rtp_llm
