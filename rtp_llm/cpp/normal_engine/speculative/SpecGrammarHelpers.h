#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <ATen/ATen.h>
#include <dlpack/dlpack.h>
#include <torch/extension.h>

namespace rtp_llm {

class StreamGroups;
class GenerateStream;
class RtpGrammarMatcher;
using GenerateStreamPtr = std::shared_ptr<GenerateStream>;

namespace speculative {
struct SpeculativeSamplerOutput;
}

namespace spec_grammar {

// One alive matcher within a batch, with a non-owning view into its draft
// chain row. `chain_ptr/chain_len` is a span into the caller-owned tensor; it
// must outlive this struct (it does, since callers use it within one function
// scope). `stream` is kept so per-stream errors can call reportError in O(1)
// instead of re-walking the StreamGroups list.
//
// Passthrough matchers (reasoning <think> phase) are NOT excluded — the
// per-position walker decides whether to mask via isPassthroughForMask().
struct ActiveGrammarStream {
    int                batch_idx;
    RtpGrammarMatcher* matcher;
    GenerateStreamPtr  stream;
    const int32_t*     chain_ptr;
    size_t             chain_len;
};

// Runtime check: any stream has a matcher actually attached. Distinct from
// streamGroupsHaveGrammar(), which checks generate_config (request layer).
bool streamGroupsHaveAttachedGrammar(const StreamGroups& stream_groups);

// Collect all alive matchers (terminated/finished excluded; passthrough kept).
// `chain_tensor` is 2D [batch, row_len]; chain_skip_first lets the spec path
// skip the prefix-bonus T0 column. `vocab_size_out` is updated on first
// matcher seen (kept at -1 if none).
std::vector<ActiveGrammarStream> collectActiveGrammarStreams(
    const StreamGroups&  stream_groups,
    const torch::Tensor& chain_tensor,
    size_t               chain_skip_first,
    int32_t&             vocab_size_out);

// Build a 2D [batch_size, words] int32 DLTensor view over `data`. Shape is
// stored in a thread_local buffer; safe as long as no nested call inside the
// resulting view's lifetime.
DLTensor makeBitmaskView(int32_t* data, int32_t batch_size, int32_t words);

bool tokenAllowed(const int32_t* bitmask, int32_t tok, int32_t vocab_size);

bool streamGroupsHaveGrammar(const StreamGroups& stream_groups);

bool spAcceptTraceEnabled();

std::string grammarKindTag(const GenerateStreamPtr& stream);

void logSpAcceptTrace(const StreamGroups&                          stream_groups,
                      const speculative::SpeculativeSamplerOutput& spec_output,
                      size_t                                       propose_step);

}  // namespace spec_grammar
}  // namespace rtp_llm
