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
    bool                 post_layernorm = false;
};

struct GptModelInitParams {
    ft::DeviceBase*           device;
    const ft::Weights         weights;
    const GptModelDescription description;
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

    ft::BufferPtr combo_tokens_type_ids;      // [cumulated_seq_len]
    ft::BufferPtr combo_position_ids;      // [cumulated_seq_len]

    ft::BufferPtr attention_mask;  // [batch_size, seq_len, seq_len]
    ft::BufferPtr position_ids;    // [batch_size, seq_len]

    ft::BufferPtr prefix_lengths;   // [batch_size, seq_len]
    ft::BufferPtr kv_cache_blocks;  // [layer_num, batch_size, 2, block_nums], int64 block pointers
    ft::BufferPtr kv_cache_scales;  // [layer_num, batch_size, 2, block_nums], int64 block scales

    ft::BufferPtr lora_ids;         // [batch_size]

public:
    std::string debugString() const {
        std::stringstream debug_string;
        debug_string << "GptModelInputs { "
                     << "combo_tokens: " << combo_tokens->debugString()
                     << ", input_lengths: " << input_lengths->debugString()
                     << ", sequence_lengths: " << sequence_lengths->debugString()
                     << ", prefix_lengths: " << prefix_lengths->debugString();
        if (kv_cache_blocks != nullptr) {
            debug_string << ", kv_cache_blocks: " << kv_cache_blocks->debugString() << "}";
        }
        return debug_string.str();
    }
};

// TODO(xinfei.sxf) sync lora id and other member
inline void tpSyncModelInputs(GptModelInputs &inputs, ft::DeviceBase* device) {
    if (device->getDeviceProperties().tp_size <= 1) {
        return;
    }
    const size_t shape_hints_size = 6;
    auto shape_hints = device->allocateBuffer({ft::DataType::TYPE_INT32, {shape_hints_size}, ft::AllocationType::HOST});
    auto shape_hints_ptr = shape_hints->data<int32_t>();
    shape_hints_ptr[0] = inputs.combo_tokens.get() ? inputs.combo_tokens->size() : 0; // combo_token_size
    shape_hints_ptr[1] = inputs.input_lengths.get() ? inputs.input_lengths->size() : 0; // total_batch_size
    shape_hints_ptr[2] = inputs.sequence_lengths.get() ? inputs.sequence_lengths->size() : 0; // generate_batch_size
    shape_hints_ptr[3] = inputs.kv_cache_blocks.get() ? inputs.kv_cache_blocks->shape()[0] : 0; // layer_num
    shape_hints_ptr[4] = inputs.kv_cache_blocks.get() ? inputs.kv_cache_blocks->shape()[3] : 0; // block_size
    shape_hints_ptr[5] = inputs.kv_cache_scales.get() != nullptr; // use_block_scale
    device->broadcast({{shape_hints}, 0});
    device->syncAndCheck();
    if (device->getDeviceProperties().tp_rank) {
        inputs.combo_tokens = device->allocateBuffer({ft::DataType::TYPE_INT32, {(size_t)shape_hints_ptr[0]}, ft::AllocationType::HOST});
        inputs.input_lengths = device->allocateBuffer({ft::DataType::TYPE_INT32, {(size_t)shape_hints_ptr[1]}, ft::AllocationType::HOST});
        inputs.sequence_lengths = device->allocateBuffer({ft::DataType::TYPE_INT32, {(size_t)shape_hints_ptr[2]}, ft::AllocationType::HOST});
        inputs.kv_cache_blocks = device->allocateBuffer({ft::DataType::TYPE_UINT64, {(size_t)shape_hints_ptr[3], (size_t)shape_hints_ptr[1], 2, (size_t)shape_hints_ptr[4]}, ft::AllocationType::HOST});
        if (shape_hints_ptr[5]) {
            inputs.kv_cache_scales = device->allocateBuffer({ft::DataType::TYPE_INT32, {(size_t)shape_hints_ptr[3], (size_t)shape_hints_ptr[1], 2, (size_t)shape_hints_ptr[4]}, ft::AllocationType::HOST});
        }
    }
    std::vector<ft::BufferPtr> buffers;
    buffers.emplace_back(inputs.combo_tokens);
    buffers.emplace_back(inputs.input_lengths);
    buffers.emplace_back(inputs.sequence_lengths);
    buffers.emplace_back(inputs.kv_cache_blocks);
    if (shape_hints_ptr[5]) {
        buffers.emplace_back(inputs.kv_cache_scales);
    }
    device->broadcast({buffers, 0});
    device->syncAndCheck();
}

struct GptModelOutputs {
    ft::BufferPtr logits;
    ft::BufferPtr hidden_states;
    ft::BufferPtr all_hidden_states;

    mutable ft::BufferPtr scatter_logits;
    mutable ft::BufferPtr scatter_hidden_states;
};

class GptModel {
public:
    GptModel(const GptModelInitParams& params);
    virtual ~GptModel() {};

    virtual GptModelOutputs forward(const GptModelInputs& inputs);

private:
    void prepareAttentionInputs(const GptModelInputs& inputs, ft::AttentionCommonInputs& attention_inputs);

private:
    ft::DeviceBase* device_;
    const ft::DeviceProperties device_props_;
    const ft::Weights          weights_;
    const GptModelDescription  description_;

};

}  // namespace rtp_llm
