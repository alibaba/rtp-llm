#pragma once

#include "maga_transformer/cpp/common/torch_bind.h"
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/devices/DeviceBase.h"
#include "src/fastertransformer/devices/OpData.h"
#include "src/fastertransformer/devices/Weights.h"
#include <string>

namespace ft = fastertransformer;

namespace rtp_llm {

struct GptModelDescription {
    ft::AttentionConfigs attention_conf;
    ft::ActivationType   activation_type;
    ft::NormType         norm_type;
    double               layernorm_eps = 1e-5;
};

struct GptModelInitParams {
    ft::DeviceBase*                device;
    const ft::Weights&             weights;
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
    ft::BufferPtr combo_tokens;      // [cumulated_seq_len]
    ft::BufferPtr input_lengths;     // [batch_size]
    ft::BufferPtr sequence_lengths;  // [decoder_batch_size]

    ft::BufferPtr attention_mask;  // [batch_size, seq_len, seq_len]
    ft::BufferPtr position_ids;    // [batch_size, seq_len]

    ft::BufferPtr prefix_lengths;   // [batch_size, seq_len]
    ft::BufferPtr kv_cache_blocks;  // [layer_num, batch_size, 2, block_length], int64 block pointers
    ft::BufferPtr kv_cache_scales;  // [layer_num, batch_size, 2, block_length], int64 block scales

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
    ft::BufferPtr logits;
    ft::BufferPtr hidden_states;
};

class GptModel {
public:
    GptModel(const GptModelInitParams& params);
    virtual ~GptModel() {};

    virtual GptModelOutputs forward(const GptModelInputs& inputs);

private:
    ft::AttentionCommonInputs prepareAttentionInputs(const GptModelInputs& inputs);

private:
    ft::DeviceBase* device_;
    const ft::Weights& weights_;
    const GptModelDescription& description_;
};

}  // namespace rtp_llm
