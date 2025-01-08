#pragma once

#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/devices/DeviceBase.h"
#include "src/fastertransformer/devices/OpData.h"
#include "src/fastertransformer/devices/Weights.h"
#include "maga_transformer/cpp/cache/CacheManager.h"
#include <string>
#include <utility>

namespace ft = fastertransformer;

namespace rtp_llm {

struct GptModelDescription {
    ft::AttentionConfigs attention_conf;
    ft::FfnConfigs       ffn_conf;
    ft::NormType         norm_type;
    ft::QScheme          act_qscheme = ft::QScheme::NoQuantize;
    double               layernorm_eps = 1e-5;
    size_t               vocab_size = 0;
    bool                 post_layernorm = false;
    double               input_embedding_scalar = 1;
    double               residual_scalar = 1;
};

struct GptModelInitParams {
    ft::DeviceBase*                                  device;
    const ft::Weights                                weights;
    const GptModelDescription                        description;
    const std::optional<CacheManager::KVCacheBuffer> kv_cache_buffer;
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
    ft::BufferPtr prefix_lengths;    // [context_batch_size]

    ft::BufferPtr combo_tokens_type_ids;      // [cumulated_seq_len]
    ft::BufferPtr combo_position_ids;         // [cumulated_seq_len]

    // for tp sync
    ft::BufferPtr lora_ids;           // [batch_size]
    ft::BufferPtr lora_input_lengths; // [batch_size]

    // no need tp sync
    ft::lora::LoraModelInputPtr lora_model_input;

    ft::BufferPtr attention_mask;  // [batch_size, seq_len, seq_len]

    ft::BufferPtr kv_cache_block_id;    // [batch_size, block_nums], kv cache block block id

    std::optional<std::vector<ft::BufferPtr>> multimodal_features; // all features in gathered stream stored here
    ft::BufferPtr                             text_tokens_mask;    // text part in multimodal input tokens [cumulated_seq_len]
    ft::BufferPtr                             mm_features_locs;    // features index

    ft::BufferPtr                             request_id;               // int64, [context_batch_size]
    ft::BufferPtr                             request_pd_separation;    // bool, [context_batch_size]
    ft::BufferPtr                             cache_keys;               // [context_batch_size]
    size_t                                    block_size;
    size_t                                    scale_block_size;
    bool                                      pd_separation = false;

    bool                                      need_all_logits = false;
    bool                                      warmup          = false;

public:
    std::string debugString() const;
};

enum GptModelInputIndex : size_t{
    comboTokens,
    inputLengths,
    sequenceLengths,
    prefixLengths,
    maxBlocksPerBatch,
    lmOutputIndexes,
    comboPositionIds,
    loraIds,
    loraInputLengths,
    textTokensMask,
    mmFeaturesLocs,
    mmFeaturesNum, // number of mm features
    mmFeaturesSize, // hidden_size of mm features
    mmFeaturesDtype,
    needAllLogits,
    gptModelInputLength
};

void tpSyncModelInputs(GptModelInputs &inputs, ft::DeviceBase* device);

struct GptModelOutputs {
    ft::BufferPtr logits;
    ft::BufferPtr hidden_states;
    ft::BufferPtr all_hidden_states;
    ft::BufferPtr all_logits;
    ft::BufferPtr softmax_result;

    mutable ft::BufferPtr scatter_logits;
    mutable ft::BufferPtr scatter_hidden_states;
};

using LoraMap = std::unordered_map<std::string, ft::ConstBufferPtr>;

struct GptLayerOutputs {
    ft::BufferPtr hidden;
    ft::BufferPtr pre_decoder_residual;
};

struct GptLayerInputs {
    ft::BufferPtr hidden;
    ft::BufferPtr pre_decoder_residual;
    ft::AttentionCommonInputs attention_common_inputs;
    const ft::DataType dtype;
};

class GptModel {
public:
    GptModel(const GptModelInitParams& params);
    virtual ~GptModel() {};

    virtual GptModelOutputs forward(const GptModelInputs& inputs);

private:
    void prepareAttentionInputs(
        const GptModelInputs& inputs,
        ft::DataType dtype,
        ft::AttentionCommonInputs& attention_inputs);

    ft::BufferPtr tpSyncEmbeddingOrLogits(const ft::BufferPtr& buffer);

    GptLayerInputs forwardPreLayers(const GptModelInputs& inputs);

    GptLayerOutputs forwardGptLayer(
        GptLayerInputs inputs,
        const int32_t layer_id,
        ft::lora::LoraModelInputPtr lora_model_input);

    GptModelOutputs forwardPostLayers(
        const ft::BufferPtr hidden,
        const bool has_context_request,
        const bool need_all_logits,
        const ft::BufferPtr lm_output_indexes);

private:
    ft::DeviceBase* device_;
    const ft::DeviceProperties device_props_;
    const ft::Weights          weights_;
    const size_t               layer_num_;
    const GptModelDescription  description_;
    ft::BufferPtr              k_cache_buffer_;
    ft::BufferPtr              v_cache_buffer_;
    ft::BufferPtr              k_scale_buffer_;
    ft::BufferPtr              v_scale_buffer_;
    ft::BufferPtr              residual_scale_fp32_;
    ft::BufferPtr              residual_scale_;
};

}  // namespace rtp_llm
