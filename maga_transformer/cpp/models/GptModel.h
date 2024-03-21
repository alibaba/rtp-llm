#pragma once

#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/devices/DeviceBase.h"
#include "src/fastertransformer/devices/Weights.h"

using namespace fastertransformer;

namespace rtp_llm {

struct GptModelDescription {
    ActivationType activation_type;
    NormType norm_type;
    double layernorm_eps = 1e-5;

    int64_t rotary_embedding_style    = 0;
    int64_t rotary_embedding_dim      = 0;
    int64_t rotary_embedding_base     = 10000;
};

struct GptModelInitParams {
    DeviceBase* device;
    const Weights& weights;
    const GptModelDescription& description;
};

// A batch includes two parts: context batch and decoder batch.
// context batch is request for initial word, decoder batch is request for incremental word.
// ids and lengths are int32_t
struct GptModelInputs {
    const Buffer& combo_tokens;                // [cumulated_seq_len]
    const Buffer& input_lengths;               // [batch_size]
    const Buffer& sequence_lengths;            // [decoder_batch_size]

    OptionalConstBufferRef attention_mask;     // [batch_size, seq_len, seq_len]
    OptionalConstBufferRef position_ids;       // [batch_size, seq_len]

    const Buffer& kv_cache_blocks;             // [batch_size, block_length], int64 block pointers
};

struct GptModelOutputs {
    const std::unique_ptr<Buffer> logits;
};

class GptModel {
public:
    GptModel(const GptModelInitParams& params);
    ~GptModel() {};

    GptModelOutputs forward(const GptModelInputs& inputs);

private:
    DeviceBase* device_;
    const Weights& weights_;
    const GptModelDescription& description_;
    const AttentionConfigs attention_configs_;
};

} // namespace rtp_llm

