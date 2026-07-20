#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "rtp_llm/cpp/models/SpecLogitsProcessorTypes.h"

namespace rtp_llm {

struct SpecLogitsProcessorRequest {
    const int32_t* draft_tokens = nullptr;
    int            propose_step = 0;

    // Packed bitmask rows, shape [(propose_step + 1), bitmask_size_int32].
    // bit=1 means allowed; bit=0 means masked.
    int32_t* bitmask_cpu_out    = nullptr;
    size_t   bitmask_size_int32 = 0;
    size_t   vocab_size         = 0;

    uint64_t stream_id       = 0;
    uint64_t mtp_round_id    = 0;
    int64_t  base_seq_len    = 0;
    int64_t  base_output_len = 0;
};

class SpecLogitsProcessor {
public:
    virtual ~SpecLogitsProcessor() = default;

    virtual bool isSpecVerifyEligible() const = 0;

    // Pure wrt committed processor state. Implementations may use a local
    // copy/rollback, but the externally committed state must be unchanged.
    // Return cap in [0, propose_step]. If draft[j] is rejected by the
    // processor state, return j so the target row j can provide a replacement.
    virtual int tryAcceptAndFillBitmask(const SpecLogitsProcessorRequest& request) = 0;

    static size_t bitmaskWordCount(size_t vocab_size) {
        return (vocab_size + 31) / 32;
    }

    static constexpr int32_t kBitmaskAllowAll = -1;
};

using SpecLogitsProcessorPtr = std::shared_ptr<SpecLogitsProcessor>;

}  // namespace rtp_llm
