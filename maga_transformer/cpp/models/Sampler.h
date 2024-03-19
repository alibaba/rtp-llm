#pragma once

#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/devices/DeviceBase.h"

using namespace fastertransformer;

namespace rtp_llm {

struct SamplerInitParams {
    DeviceBase* device;
};

struct SamplerInputs {
    const Buffer& logits;            // shape: [batch_size * num_beams, vocab_size]
    BufferPtr token_ids;             // shape: [batch_size * num_beams, max_length]
    const Buffer& sequence_lenghts;  // shape: [batch_size]
    const size_t step;               // typically largest sequence length in the batch

    const size_t batch_size;
    const Buffer& num_beams;          // shape: [batch_size]
    const Buffer& top_k;              // shape: [batch_size]
    const Buffer& top_p;              // shape: [batch_size]
    const Buffer& temperature;        // shape: [batch_size]
    const Buffer& repetition_penalty; // shape: [batch_size]
    const Buffer& length_penalty;     // shape: [batch_size]

    Buffer& kv_cache_blocks; // shape: [batch_size * num_beams, block_length], int64 block pointers
    BufferPtr cum_log_probs; // shape: [batch_size * num_beams]
};

struct SamplerOutput {
    BufferPtr token_ids;
    BufferPtr cum_log_probs;
};

// Sampler would split logits into appropriate groups (mostly, based on beam size)
// and calls device sampling apis (greedy, beam search, etc) for each group
class Sampler {
    Sampler(const SamplerInitParams& params);
    ~Sampler() {};

    SamplerOutput forward(const SamplerInputs& inputs);

private:
    DeviceBase* device_;
};


} // namespace rtp_llm
