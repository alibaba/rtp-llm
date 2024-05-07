#pragma once
#include "maga_transformer/cpp/models/Sampler.h"

namespace rtp_llm {

struct SpeculativeSamplerInput {
public:
    size_t    gen_num_per_circle;
    ft::BufferPtr token_ids;         // [batch_size, seq_len]
    ft::BufferPtr sequence_lengths;  // [batch_size, seq_len]
    ft::BufferPtr draft_prob;        // [batch_size, gen_num_per_circle, vocab_size]
    ft::BufferPtr target_prob;       // [batch_size, gen_num_per_circle, vocab_size]
};

struct SpeculativeSamplerOutput {
public:
    std::vector<uint> output_token_len;
    ft::BufferPtr         output_token_ids;  // [batch_size, seq_len]
};

class SpeculativeSampler {
public:
    SpeculativeSampler(const SamplerInitParams& params);
    ~SpeculativeSampler(){};

    SpeculativeSamplerOutput forward(const SpeculativeSamplerInput& inputs);

private:
    ft::DeviceBase* device_;
};

}  // namespace rtp_llm
