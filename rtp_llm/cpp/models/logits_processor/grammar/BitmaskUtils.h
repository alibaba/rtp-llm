#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include <dlpack/dlpack.h>

#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

// Single-row bitmask view backed by a contiguous int32 buffer of `words` words.
// shape_out must outlive the returned DLTensor (it stores the shape pointer).
inline DLTensor makeSingleRowBitmaskView(int32_t* data, int32_t words, int64_t shape_out[2]) {
    DLTensor dl;
    dl.data        = data;
    dl.device      = DLDevice{kDLCPU, 0};
    dl.ndim        = 2;
    dl.dtype       = DLDataType{kDLInt, 32, 1};
    shape_out[0]   = 1;
    shape_out[1]   = words;
    dl.shape       = shape_out;
    dl.strides     = nullptr;
    dl.byte_offset = 0;
    return dl;
}

inline bool bitmaskAllowsToken(const int32_t* bitmask, size_t words, int32_t token_id) {
    if (token_id < 0 || static_cast<size_t>(token_id / 32) >= words) {
        return false;
    }
    const uint32_t word = static_cast<uint32_t>(bitmask[token_id / 32]);
    return (word & (1u << (token_id % 32))) != 0u;
}

inline void clearTokenFromBitmask(int32_t* bitmask, size_t words, int64_t token_id) {
    if (token_id < 0 || static_cast<size_t>(token_id / 32) >= words) {
        return;
    }
    bitmask[token_id / 32] &= ~(1u << (token_id % 32));
}

// Out-of-range token_id violates an upstream invariant (eos / end_think_token_ids
// must fit within the model vocab bitmask). Caller is expected to validate before
// reaching here; bail loudly so the misconfiguration is visible at the source
// instead of producing a silent all-zero (all-disabled) row that deadlocks sampling.
inline void forceTokenInBitmask(int32_t* bitmask, size_t words, int64_t token_id) {
    RTP_LLM_CHECK_WITH_INFO(token_id >= 0 && static_cast<size_t>(token_id / 32) < words,
                            "forceTokenInBitmask: token_id %ld out of range (bitmask words=%zu)",
                            static_cast<long>(token_id),
                            words);
    std::fill_n(bitmask, words, 0);
    bitmask[token_id / 32] |= (1u << (token_id % 32));
}

inline void clearBitmaskTokenRange(int32_t* bitmask, size_t words, int64_t begin_token, int64_t end_token) {
    if (begin_token < 0 || end_token <= begin_token) {
        return;
    }
    for (int64_t token_id = begin_token; token_id < end_token; ++token_id) {
        clearTokenFromBitmask(bitmask, words, token_id);
    }
}

}  // namespace rtp_llm
