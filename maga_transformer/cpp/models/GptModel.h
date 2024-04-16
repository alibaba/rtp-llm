#pragma once

#include "maga_transformer/cpp/common/torch_bind.h"
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/devices/DeviceBase.h"
#include "src/fastertransformer/devices/Weights.h"
#include <string>

using namespace fastertransformer;

namespace rtp_llm {

struct GptModelDescription {
    ActivationType activation_type;
    NormType       norm_type;
    double         layernorm_eps = 1e-5;

    int64_t rotary_embedding_style = 0;
    int64_t rotary_embedding_dim   = 0;
    int64_t rotary_embedding_base  = 10000;
};

struct GptModelInitParams {
    DeviceBase*                device;
    const Weights&             weights;
    const GptModelDescription& description;
};

// A batch includes two parts: context batch and decoder batch.
// context batch is request for initial word, decoder batch is request for incremental word.
// ids and lengths are int32_t
struct GptModelInputs {
    // input_lengths holds original input length for requests,
    // shape [decoder_batch_size + context_batch_size], int32
    // sequence_lengths holds current sequence length for incremental decoding requests,
    // shape [decoder_batch_size], int32
    BufferPtr combo_tokens;      // [cumulated_seq_len]
    BufferPtr input_lengths;     // [batch_size]
    BufferPtr sequence_lengths;  // [decoder_batch_size]

    OptionalConstBufferRef attention_mask;  // [batch_size, seq_len, seq_len]
    OptionalConstBufferRef position_ids;    // [batch_size, seq_len]

    BufferPtr prefix_lengths;   // [batch_size, seq_len]
    BufferPtr kv_cache_blocks;  // [layer_num, 2, batch_size, block_length], int64 block pointers
    BufferPtr kv_cache_scales;  // [layer_num, 2, batch_size, block_length], int64 block scales

    std::string debugString() const {
        std::stringstream debug_string;
        debug_string << "GptModelInputs { "
                     << "combo_tokens: " << combo_tokens->debugString()
                     << ", input_lengths: " << input_lengths->debugString()
                     << ", sequence_lengths: " << sequence_lengths->debugString()
                     << ", prefix_lengths: " << prefix_lengths->debugString()
                     << ", kv_cache_blocks: " << kv_cache_blocks->debugString() << "}";
        return debug_string.str();
    }
};

struct GptModelOutputs {
    BufferPtr logits;
    BufferPtr hidden_states;
};

class GptModel {
public:
    GptModel(const GptModelInitParams& params);
    ~GptModel(){};

    GptModelOutputs forward(const GptModelInputs& inputs);

private:
    AttentionCommonInputs prepareAttentionInputs(const GptModelInputs& inputs);

private:
    DeviceBase* device_;
    const Weights& weights_;
    const GptModelDescription& description_;
    const AttentionConfigs     attention_configs_;
};

}  // namespace rtp_llm
