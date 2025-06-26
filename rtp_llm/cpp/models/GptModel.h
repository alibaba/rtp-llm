#pragma once

#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/core/Event.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/devices/OpData.h"
#include "rtp_llm/cpp/devices/Weights.h"
#include "rtp_llm/cpp/stats/ExpertStats.h"
#include "rtp_llm/cpp/cache/CacheManager.h"
#include <string>
#include <utility>

namespace rtp_llm {

struct GptModelDescription {
    rtp_llm::AttentionConfigs attention_conf;
    rtp_llm::FfnConfigs       ffn_conf;
    rtp_llm::NormType         norm_type;
    rtp_llm::QScheme          act_qscheme            = rtp_llm::QScheme::NoQuantize;
    double                    layernorm_eps          = 1e-5;
    size_t                    vocab_size             = 0;
    bool                      post_layernorm         = false;
    double                    input_embedding_scalar = 1;
    double                    residual_scalar        = 1;
    bool                      reverse_e_h_norm       = false;
};

struct GptModelInitParams {
    rtp_llm::DeviceBase*                             device;
    const rtp_llm::Weights                           weights;
    const GptModelDescription                        description;
    const std::optional<CacheManager::KVCacheBuffer> kv_cache_buffer;
    size_t                                           model_id;
};

struct EmbeddingPostOutput {
    BufferPtr hidden;
    BufferPtr residual;
};

// A batch includes two parts: context batch and decoder batch.
// context batch is request for initial word, decoder batch is request for incremental word.
// ids and lengths are int32_t
struct GptModelInputs {
    // input_lengths holds original input length for requests,
    // shape [decoder_batch_size + context_batch_size], int32
    // sequence_lengths holds current sequence length for incremental decoding requests,
    // shape [decoder_batch_size], int32
    mutable rtp_llm::BufferPtr combo_tokens;       // [cumulated_seq_len]
    rtp_llm::BufferPtr         input_lengths;      // [batch_size]
    rtp_llm::BufferPtr         sequence_lengths;   // [decoder_batch_size]
    rtp_llm::BufferPtr         lm_output_indexes;  // [sum(lm_output_lengths)]
    rtp_llm::BufferPtr         lm_output_lengths;  // [total_batch_size]
    rtp_llm::BufferPtr         prefix_lengths;     // [context_batch_size]

    rtp_llm::BufferPtr combo_tokens_type_ids;  // [cumulated_seq_len]
    rtp_llm::BufferPtr combo_position_ids;     // [cumulated_seq_len]

    // for mtp model
    rtp_llm::BufferPtr last_hidden_states;

    // for tp sync
    rtp_llm::BufferPtr lora_ids;            // [batch_size]
    rtp_llm::BufferPtr lora_input_lengths;  // [batch_size]

    // no need tp sync
    rtp_llm::lora::LoraModelInputPtr lora_model_input;

    rtp_llm::BufferPtr attention_mask;  // [batch_size, seq_len, seq_len]

    rtp_llm::BufferPtr kv_cache_block_id;  // [batch_size, block_nums], kv cache block block id

    std::optional<std::vector<rtp_llm::BufferPtr>> multimodal_features;  // all features in gathered stream stored here
    rtp_llm::BufferPtr text_tokens_mask;  // text part in multimodal input tokens [cumulated_seq_len]
    rtp_llm::BufferPtr mm_features_locs;  // features index

    rtp_llm::BufferPtr                        request_id;               // int64, [context_batch_size]
    rtp_llm::BufferPtr                        request_pd_separation;    // bool, [context_batch_size]
    rtp_llm::BufferPtr                        cache_keys;               // [context_batch_size]
    size_t                                    k_block_size;
    size_t                                    v_block_size;
    size_t                                    scale_block_size;
    size_t                                    seq_size_per_block;
    bool                                      pd_separation = false;
    bool                                      decode_entrance = false;

    bool need_all_logits = false;
    bool warmup          = false;
    bool skip_run        = false;

public:
    std::string debugString() const;
};

enum GptModelInputIndex : size_t {
    comboTokens,
    inputLengths,
    sequenceLengths,
    prefixLengths,
    maxBlocksPerBatch,
    lmOutputIndexes,
    lmOutputLengthes,
    comboPositionIds,
    loraIds,
    loraInputLengths,
    textTokensMask,
    mmFeaturesLocs,
    mmFeaturesNum,   // number of mm features
    mmFeaturesSize,  // hidden_size of mm features
    mmFeaturesDtype,
    needAllLogits,
    mtpHiddenStates,
    mtpHiddenStatesDtype,
    skipRun,
    gptModelInputLength
};

void tpSyncModelInputs(GptModelInputs& inputs, rtp_llm::DeviceBase* device);

struct GptModelOutputs {
    rtp_llm::BufferPtr logits;
    rtp_llm::BufferPtr hidden_states;
    rtp_llm::BufferPtr all_hidden_states;
    rtp_llm::BufferPtr all_logits;
    rtp_llm::BufferPtr softmax_result;

    mutable rtp_llm::BufferPtr scatter_logits;
    mutable rtp_llm::BufferPtr scatter_hidden_states;
    std::shared_ptr<void>      captured_values;
};

using LoraMap = std::unordered_map<std::string, rtp_llm::ConstBufferPtr>;

struct GptLayerOutputs {
    rtp_llm::BufferPtr hidden;
    rtp_llm::BufferPtr pre_decoder_residual;
};

struct MicroBatchInfo {
    size_t prefill_num;
    size_t decoder_num;
};

struct MicroBatchPlan {
    bool                        enable = false;
    std::vector<MicroBatchInfo> batch_infos;
};

struct LayerMicroBatchInputs {
    rtp_llm::BufferPtr             hidden;
    rtp_llm::BufferPtr             pre_decoder_residual;
    rtp_llm::AttentionCommonInputs attention_common_inputs;
    bool                           fake = false;
};

struct GptLayerInputs {
    rtp_llm::BufferPtr                 hidden;
    rtp_llm::BufferPtr                 pre_decoder_residual;
    rtp_llm::AttentionCommonInputs     attention_common_inputs;
    const rtp_llm::DataType            dtype;
    std::vector<LayerMicroBatchInputs> micro_batch_inputs;
    bool                               enable_sp     = false;
    size_t                             token_num     = 0;
    size_t                             pad_token_num = 0;
    BufferPtr                          residual      = nullptr;
};

struct AttentionBlockOutputs {
    rtp_llm::BufferPtr hidden;
    rtp_llm::BufferPtr residual;
    rtp_llm::BufferPtr residual2;
    rtp_llm::BufferPtr last_layer_hidden;
};

struct EpFfnInputs {
    rtp_llm::BufferPtr           hidden;
    rtp_llm::BufferPtr           quantized_hidden;
    rtp_llm::BufferPtr           residual;
    rtp_llm::BufferPtr           shared_expert_output;
    rtp_llm::FfnLayerParams      moe_ffn_params;
    rtp_llm::MoeGateSelectOutput gate_output;
    rtp_llm::DeviceEventPtr      compute_event;
    rtp_llm::BufferPtr           last_layer_hidden;
};

struct MoeOutputs {
    rtp_llm::BufferPtr      hidden;
    rtp_llm::DeviceEventPtr compute_event;
};

struct EpFfnOutputs {
    rtp_llm::BufferPtr        hidden;
    rtp_llm::MoeCombineOutput combine_output;
    rtp_llm::DeviceHookPtr    comm_barrier_hook;
};

struct LastLayerDeferedParams {
    rtp_llm::BufferPtr                               residual;
    rtp_llm::BufferPtr                               shared_expert_output;
    std::optional<rtp_llm::MoeCombineOutput>         combine_output;
    std::shared_ptr<const rtp_llm::LayerNormWeights> post_ffn_layernorm_weights;
    rtp_llm::DeviceHookPtr                           comm_barrier_hook;
};

class GptModel {
public:
    GptModel(const GptModelInitParams& params);
    virtual ~GptModel() {};

    virtual GptModelOutputs forward(const GptModelInputs& inputs);

protected:
    rtp_llm::AttentionCommonInputs prepareAttentionInputs(const GptModelInputs& inputs,
                                                          rtp_llm::DataType     attn_dtype,
                                                          rtp_llm::BufferPtr    combo_position_ids);

    MicroBatchPlan                     planMicroBatches(const GptModelInputs& inputs);
    std::vector<LayerMicroBatchInputs> prepareMicroBatchInputs(const GptModelInputs&     model_inputs,
                                                               const rtp_llm::BufferPtr& hidden,
                                                               const rtp_llm::BufferPtr& pre_decoder_residual,
                                                               const rtp_llm::DataType   attn_dtype,
                                                               const MicroBatchPlan&     micro_batch_plan);

    virtual EmbeddingPostOutput embeddingPost(const rtp_llm::BufferPtr& hidden_states, const GptModelInputs& inputs);

    rtp_llm::BufferPtr tpSyncEmbeddingOrLogits(const rtp_llm::BufferPtr& buffer);

    GptLayerInputs forwardPreLayers(const GptModelInputs& inputs);

    GptLayerOutputs forwardGptLayer(GptLayerInputs                          inputs,
                                    const int32_t                           layer_id,
                                    const rtp_llm::lora::LoraModelInputPtr& lora_model_input);

    AttentionBlockOutputs forwardAttentionBlock(const GptLayerInputs&                   inputs,
                                                const int32_t                           layer_id,
                                                const rtp_llm::lora::LoraModelInputPtr& lora_model_input,
                                                const LastLayerDeferedParams&           last_layer_defered_params = {},
                                                bool                                    capture_last_hidden = false);

    // These methods are dedicated for moe ep micro batching
    GptLayerOutputs forwardMicroBatchedLayers(const GptLayerInputs&   layer_inputs,
                                              const GptModelInputs&   inputs,
                                              std::vector<BufferPtr>& eagle3_selected_hidden);

    std::vector<GptLayerInputs> forwardPrefillMicroBatchedLayers(std::vector<GptLayerInputs> inputs,
                                                                 std::vector<BufferPtr>&     eagle3_selected_hidden);

    std::vector<GptLayerInputs> forwardDecodeMicroBatchedLayers(std::vector<GptLayerInputs> inputs,
                                                                std::vector<BufferPtr>&     eagle3_selected_hidden);

    BufferPtr mergeEagle3HiddenState(const GptLayerInputs&   layer_inputs,
                                     std::vector<BufferPtr>& eagle3_selected_hidden);

    EpFfnInputs forwardAttentionAndMoeGate(const GptLayerInputs&   inputs,
                                           LastLayerDeferedParams& last_layer_defered_params,
                                           const int32_t           layer_id,
                                           const size_t            micro_batch_idx,
                                           bool                    capture_last_hidden);

    GptLayerOutputs forwardMoeFfn(const GptLayerOutputs& inputs, const int32_t layer_id);

    bool containMoeLayer();

    GptModelOutputs forwardPostLayers(const rtp_llm::BufferPtr hidden,
                                      const bool               has_context_request,
                                      const bool               need_all_logits,
                                      const rtp_llm::BufferPtr lm_output_indexes,
                                      bool                     enable_sp,
                                      size_t                   token_num,
                                      const GptModelInputs&    inputs,
                                      const rtp_llm::BufferPtr merged_eagle3_hidden);

    void prepareExpertStats(const size_t layer_id, rtp_llm::FfnLayerParams& ffn_layer_params);

    void cleanExpertStats();

protected:
    rtp_llm::DeviceBase*            device_;
    const rtp_llm::DeviceProperties device_props_;
    const size_t                    layer_num_;
    const GptModelDescription       description_;
    rtp_llm::BufferPtr              k_cache_buffer_;
    rtp_llm::BufferPtr              v_cache_buffer_;
    rtp_llm::BufferPtr              k_scale_buffer_;
    rtp_llm::BufferPtr              v_scale_buffer_;
    rtp_llm::BufferPtr              residual_scale_fp32_;
    rtp_llm::BufferPtr              residual_scale_;

public:
    rtp_llm::Weights            weights_;
    rtp_llm::OverallExpertStats overall_expert_stats_;
    size_t                      model_id_ = 0;
};

}  // namespace rtp_llm
