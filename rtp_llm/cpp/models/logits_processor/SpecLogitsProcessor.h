#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace rtp_llm {

// Capability mixin for LogitsProcessor backends that can take part in
// speculative-decoding verify-stage mask construction.
//
// During an MTP cycle, the executor hands the helper a per-stream draft
// token sequence. For each grammar-active stream the helper calls
// tryAcceptAndFillBitmask() to perform a "virtual advance + fill mask + roll
// back" on the backend's CPU matcher. The function returns the largest draft
// offset cap the matcher could accept; rows beyond cap are left allow-all by
// the helper, and the GPU side later truncates accept_len with
// min(verifier_accept_len, cap+1).
//
// Backends mix this in alongside BaseLogitsProcessor; helper discovers
// participants via dynamic_pointer_cast<SpecLogitsProcessor>.
class SpecLogitsProcessor {
public:
    virtual ~SpecLogitsProcessor() = default;

    // Returns true if this processor's matcher is currently in a state that
    // can produce a valid mask. False (e.g. terminated / dead / inactive)
    // makes the helper leave the row block as allow-all and cap = propose_step.
    virtual bool isSpecVerifyEligible() const = 0;

    // Walks the matcher over draft_tokens[0..propose_step), filling each
    // offset's bitmask into bitmask_cpu_out (laid out as
    // [propose_step + 1, bitmask_size_int32], int32 packed bits, bit=1 means
    // allow). Returns cap in [0, propose_step]:
    //   cap == propose_step : every draft token is grammar-legal
    //   cap == j (j < P)    : draft[j] violates grammar; rows j+1..P stay
    //                          allow-all (helper pre-initialises them).
    //
    // Postcondition: matcher state is identical to before the call.
    // Implementations are expected to use Rollback (or Fork) to achieve this.
    virtual int tryAcceptAndFillBitmask(const int32_t* draft_tokens,
                                        int            propose_step,
                                        int32_t*       bitmask_cpu_out,
                                        size_t         bitmask_size_int32) = 0;

    // bitmask row width in int32 words (xgrammar layout: ceil(vocab/32)).
    static size_t bitmaskWordCount(size_t vocab_size) {
        return (vocab_size + 31) / 32;
    }

    // "Allow all" fill value the helper uses to pre-initialise bitmask buffers.
    static constexpr int32_t kBitmaskAllowAll = -1;  // 0xFFFFFFFF
};

using SpecLogitsProcessorPtr = std::shared_ptr<SpecLogitsProcessor>;

}  // namespace rtp_llm
