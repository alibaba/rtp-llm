#pragma once

#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/devices/DeviceBase.h"

using namespace fastertransformer;

namespace rtp_llm {

struct SamplerInitParams {
    DeviceBase* device;
};

struct SamplerInputs {
public:
    BufferPtr logits;            // shape: [batch_size * num_beams, vocab_size]
    mutable BufferPtr token_ids;         // shape: [batch_size * num_beams, max_length]
    BufferPtr sequence_lengths;  // shape: [batch_size]
    size_t    step;              // typically largest sequence length in the batch

    size_t    batch_size;
    BufferPtr num_beams;           // shape: [batch_size]
    BufferPtr top_k;               // shape: [batch_size]
    BufferPtr top_p;               // shape: [batch_size]
    BufferPtr temperature;         // shape: [batch_size]
    BufferPtr random_seeds;        // shape: [batch_size]
    BufferPtr repetition_penalty;  // shape: [batch_size]
    BufferPtr length_penalty;      // shape: [batch_size]

    BufferPtr kv_cache_blocks;     // shape: [batch_size * num_beams, block_length], int64 block pointers
    mutable BufferPtr cum_log_probs;       // shape: [batch_size * num_beams]
};

struct SamplerOutput {
public:
    BufferPtr token_ids;
    BufferPtr cum_log_probs;
};

// Sampler would split logits into appropriate groups (mostly, based on beam size)
// and calls device sampling apis (greedy, beam search, etc) for each group
class Sampler {
public:
    Sampler(const SamplerInitParams& params);
    ~Sampler(){};

    SamplerOutput forward(const SamplerInputs& inputs);

private:
    DeviceBase* device_;
};

class SpeculativeSampler: public Sampler {
public:
    SpeculativeSampler(const SamplerInitParams& params): Sampler(params){};
    ~SpeculativeSampler(){};
};

class MedusaTreeSampler: public Sampler {
public:
    MedusaTreeSampler(const SamplerInitParams& params): Sampler(params){};
    ~MedusaTreeSampler(){};
};

}  // namespace rtp_llm
