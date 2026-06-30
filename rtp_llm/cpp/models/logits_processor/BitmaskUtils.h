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

// Out-of-range token_id is a caller bug; abort instead of returning an all-disabled row.
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
    const int64_t max_token = static_cast<int64_t>(words * 32);
    if (begin_token >= max_token) {
        return;
    }
    end_token = std::min(end_token, max_token);

    const size_t begin_word = static_cast<size_t>(begin_token / 32);
    const size_t end_word   = static_cast<size_t>((end_token - 1) / 32);
    const int    begin_bit  = static_cast<int>(begin_token % 32);
    const int    end_bit    = static_cast<int>(end_token % 32);

    if (begin_word == end_word) {
        const uint32_t keep_low  = begin_bit == 0 ? 0u : ((1u << begin_bit) - 1u);
        const uint32_t keep_high = end_bit == 0 ? 0u : (~0u << end_bit);
        bitmask[begin_word] &= static_cast<int32_t>(keep_low | keep_high);
        return;
    }

    const uint32_t keep_low = begin_bit == 0 ? 0u : ((1u << begin_bit) - 1u);
    bitmask[begin_word] &= static_cast<int32_t>(keep_low);

    if (end_word > begin_word + 1) {
        std::fill(bitmask + begin_word + 1, bitmask + end_word, 0);
    }

    if (end_bit == 0) {
        bitmask[end_word] = 0;
    } else {
        bitmask[end_word] &= static_cast<int32_t>(~0u << end_bit);
    }
}

}  // namespace rtp_llm
