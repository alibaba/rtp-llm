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
    ft::FfnConfigs       ffn_conf;
    ft::NormType         norm_type;
    double               layernorm_eps = 1e-5;
    size_t               vocab_size = 0;
    bool                 post_layernorm = false;
    double               input_embedding_scalar = 1;
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
    ft::BufferPtr lm_output_indexes; // [context_batch_size]
    ft::BufferPtr prefix_lengths;    // [batch_size, seq_len]
    ft::BufferPtr count_lengths;     // [1]
    ft::BufferPtr max_prefix_length; // [1]

    ft::BufferPtr combo_tokens_type_ids;      // [cumulated_seq_len]
    ft::BufferPtr combo_position_ids;         // [cumulated_seq_len]

    ft::BufferPtr lora_ids;           // [batch_size]
    ft::BufferPtr lora_input_lengths; // [batch_size]

    ft::BufferPtr attention_mask;  // [batch_size, seq_len, seq_len]
    ft::BufferPtr position_ids;    // [batch_size, seq_len]

    ft::BufferPtr kv_cache_blocks;   // [layer_num, batch_size, 2, block_nums], int64 block pointers
    ft::BufferPtr kv_cache_scales;   // [layer_num, batch_size, 2, block_nums], int64 block scales

public:
    std::string debugString() const {
        std::stringstream debug_string;
        debug_string << "GptModelInputs { "
                     << "combo_tokens: " << combo_tokens->debugStringWithData<int32_t>()
                     << ", input_lengths: " << input_lengths->debugStringWithData<int32_t>()
                     << ", sequence_lengths: " << sequence_lengths->debugStringWithData<int32_t>()
                     << ", prefix_lengths: " << prefix_lengths->debugStringWithData<int32_t>()
                     << ", count_lengths: " << count_lengths->debugStringWithData<int32_t>()
                     << ", max_prefix_length: " << max_prefix_length->debugStringWithData<int32_t>();
        if (combo_position_ids) {
            debug_string << ", combo_position_ids: " << combo_position_ids->debugStringWithData<int32_t>();
        }
        if (lora_ids) {
            debug_string << ", lora_ids: " << lora_ids->debugStringWithData<int32_t>();
        }
        if (lora_input_lengths) {
            debug_string << ", lora_input_lengths: " << lora_input_lengths->debugStringWithData<int32_t>();
        }
        if (kv_cache_blocks) {
            debug_string << ", kv_cache_blocks: " << kv_cache_blocks->debugString();
        }
        if (kv_cache_scales) {
            debug_string << ", kv_cache_scales: " << kv_cache_scales->debugString();
        }
        if (attention_mask) {
            debug_string << ", attention_mask: " << attention_mask->debugString();
        }
        debug_string << "}";
        return debug_string.str();
    }
};

inline void tpSyncModelInputs(GptModelInputs &inputs, ft::DeviceBase* device) {
    if (device->getDeviceProperties().tp_size <= 1) {
        return;
    }
    const size_t shape_hints_size = 13;
    auto shape_hints = device->allocateBuffer({ft::DataType::TYPE_INT32, {shape_hints_size}, ft::AllocationType::HOST});
    auto shape_hints_ptr = shape_hints->data<int32_t>();
    shape_hints_ptr[0] = inputs.combo_tokens.get() ? inputs.combo_tokens->size() : 0;
    shape_hints_ptr[1] = inputs.input_lengths.get() ? inputs.input_lengths->size() : 0;
    shape_hints_ptr[2] = inputs.sequence_lengths.get() ? inputs.sequence_lengths->size() : 0;
    shape_hints_ptr[3] = inputs.prefix_lengths.get() ? inputs.prefix_lengths->size() : 0;
    shape_hints_ptr[4] = inputs.count_lengths.get() ? inputs.count_lengths->size() : 0;
    shape_hints_ptr[5] = inputs.max_prefix_length.get() ? inputs.max_prefix_length->size() : 0;
    shape_hints_ptr[6] = inputs.kv_cache_blocks.get() ? inputs.kv_cache_blocks->shape()[0] : 0;
    shape_hints_ptr[7] = inputs.kv_cache_blocks.get() ? inputs.kv_cache_blocks->shape()[3] : 0;
    shape_hints_ptr[8] = inputs.kv_cache_scales.get() != nullptr;
    shape_hints_ptr[9] = inputs.lm_output_indexes.get() ? inputs.lm_output_indexes->size() : 0;
    shape_hints_ptr[10] = inputs.combo_position_ids.get() ? inputs.combo_position_ids->size() : 0;
    shape_hints_ptr[11] = inputs.lora_ids.get() ? inputs.lora_ids->size() : 0;
    shape_hints_ptr[12] = inputs.lora_input_lengths.get() ? inputs.lora_input_lengths->size() : 0;
    device->broadcast({{shape_hints}, 0});
    device->syncCommunication(false);
    device->syncAndCheck();
    if (device->getDeviceProperties().tp_rank) {
        inputs.combo_tokens = device->allocateBuffer({ft::DataType::TYPE_INT32, {(size_t)shape_hints_ptr[0]}, ft::AllocationType::HOST});
        inputs.input_lengths = device->allocateBuffer({ft::DataType::TYPE_INT32, {(size_t)shape_hints_ptr[1]}, ft::AllocationType::HOST});
        inputs.sequence_lengths = device->allocateBuffer({ft::DataType::TYPE_INT32, {(size_t)shape_hints_ptr[2]}, ft::AllocationType::HOST});
        inputs.prefix_lengths = device->allocateBuffer({ft::DataType::TYPE_INT32, {(size_t)shape_hints_ptr[3]}, ft::AllocationType::HOST});
        inputs.count_lengths = device->allocateBuffer({ft::DataType::TYPE_INT32, {(size_t)shape_hints_ptr[4]}, ft::AllocationType::HOST});
        inputs.max_prefix_length = device->allocateBuffer({ft::DataType::TYPE_INT32, {(size_t)shape_hints_ptr[5]}, ft::AllocationType::HOST});
        inputs.kv_cache_blocks = device->allocateBuffer({ft::DataType::TYPE_UINT64, {(size_t)shape_hints_ptr[6], (size_t)shape_hints_ptr[1], 2, (size_t)shape_hints_ptr[7]}, ft::AllocationType::HOST});
        if (shape_hints_ptr[8]) {
            inputs.kv_cache_scales = device->allocateBuffer({ft::DataType::TYPE_UINT64, {(size_t)shape_hints_ptr[6], (size_t)shape_hints_ptr[1], 2, (size_t)shape_hints_ptr[7]}, ft::AllocationType::HOST});
        }
        inputs.lm_output_indexes = device->allocateBuffer({ft::DataType::TYPE_INT32, {(size_t)shape_hints_ptr[9]}, ft::AllocationType::HOST});
        if (shape_hints_ptr[10]) {
            inputs.combo_position_ids = device->allocateBuffer({ft::DataType::TYPE_INT32, {(size_t)shape_hints_ptr[10]}, ft::AllocationType::HOST});
        }
        inputs.lora_ids = device->allocateBuffer({ft::DataType::TYPE_INT32, {(size_t)shape_hints_ptr[11]}, ft::AllocationType::HOST});
        inputs.lora_input_lengths = device->allocateBuffer({ft::DataType::TYPE_INT32, {(size_t)shape_hints_ptr[12]}, ft::AllocationType::HOST});
    }
    std::vector<ft::BufferPtr> buffers;
    buffers.emplace_back(inputs.combo_tokens);
    buffers.emplace_back(inputs.input_lengths);
    buffers.emplace_back(inputs.sequence_lengths);
    buffers.emplace_back(inputs.prefix_lengths);
    buffers.emplace_back(inputs.count_lengths);
    buffers.emplace_back(inputs.max_prefix_length);
    buffers.emplace_back(inputs.kv_cache_blocks);
    if (shape_hints_ptr[8]) {
        buffers.emplace_back(inputs.kv_cache_scales);
    }
    buffers.emplace_back(inputs.lm_output_indexes);
    if (shape_hints_ptr[10]) {
        buffers.emplace_back(inputs.combo_position_ids);
    }
    buffers.emplace_back(inputs.lora_ids);
    buffers.emplace_back(inputs.lora_input_lengths);
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
    ft::BufferPtr tpSyncEmbeddingOrLogits(const ft::BufferPtr& buffer);

private:
    ft::DeviceBase* device_;
    const ft::DeviceProperties device_props_;
    const ft::Weights          weights_;
    const GptModelDescription  description_;
};

}  // namespace rtp_llm
