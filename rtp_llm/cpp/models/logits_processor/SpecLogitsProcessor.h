#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

struct SpecLogitsProcessorRequest {
    const int32_t* draft_tokens = nullptr;
    int            propose_step = 0;

    // Packed bitmask rows, shape [(propose_step + 1), bitmask_size_int32].
    // bit=1 means allowed; bit=0 means masked.
    int32_t* bitmask_cpu_out    = nullptr;
    size_t   bitmask_size_int32 = 0;
    size_t   vocab_size         = 0;

    static size_t bitmaskWordCount(size_t vocab_size) {
        return (vocab_size + 31) / 32;
    }

    static constexpr int32_t kBitmaskAllowAll = -1;
};

}  // namespace rtp_llm
