#pragma once
#include "maga_transformer/cpp/models/Sampler.h"

namespace rtp_llm {

struct SpeculativeSamplerInput {
public:
    size_t    gen_num;
    BufferPtr token_ids;         // [batch_size, seq_len]
    BufferPtr sequence_lengths;  // [batch_size, seq_len]
    BufferPtr draft_prob;        // [batch_size, gen_num, vocab_size]
    BufferPtr target_prob;       // [batch_size, gen_num, vocab_size]
};

struct SpeculativeSamplerOutput {
public:
    std::vector<uint> output_token_len;
    BufferPtr         output_token_ids;  // [batch_size, seq_len]
};

class SpeculativeSampler {
public:
    SpeculativeSampler(const SamplerInitParams& params);
    ~SpeculativeSampler(){};

    SpeculativeSamplerOutput forward(const SpeculativeSamplerInput& inputs);

private:
    DeviceBase* device_;
};

}  // namespace rtp_llm
