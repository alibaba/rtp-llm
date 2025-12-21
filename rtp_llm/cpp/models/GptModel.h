#pragma once

#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/core/Event.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/devices/OpData.h"
#include "rtp_llm/cpp/devices/Weights.h"
#include "rtp_llm/cpp/models/eplb/stats/ExpertStats.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "rtp_llm/cpp/cache_new/types.h"
#include <string>
#include <utility>

namespace rtp_llm {

struct GptModelDescription {
    rtp_llm::AttentionConfigs attention_conf;
    rtp_llm::FfnConfigs       ffn_conf;
    rtp_llm::NormType         norm_type;
    DataType                  data_type;
    rtp_llm::QScheme          act_qscheme            = rtp_llm::QScheme::NoQuantize;
    const DataType            compute_type           = rtp_llm::DataType::TYPE_INVALID;
    double                    layernorm_eps          = 1e-5;
    size_t                    vocab_size             = 0;
    bool                      post_layernorm         = false;
    double                    input_embedding_scalar = 1;
    double                    residual_scalar        = 1;
    bool                      reverse_e_h_norm       = false;
};

struct GptModelInitParams {
    rtp_llm::DeviceBase*               device;
    const rtp_llm::Weights             weights;
    const GptModelDescription          description;
    const std::optional<KVCacheBuffer> kv_cache_buffer;
    size_t                             model_id;
};

struct EmbeddingPostOutput {
    BufferPtr hidden;
    BufferPtr residual;
};

enum GptModelInputIndex : size_t {
    comboTokens,
    inputLengths,
    sequenceLengths,
    prefixLengths,
    maxBlocksPerBatch,
    kvCacheUpdateCopyNum,
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
    gptModelInputLength,
};

void tpSyncModelInputs(GptModelInputs& inputs, rtp_llm::DeviceBase* device);

using LoraMap = std::unordered_map<std::string, rtp_llm::ConstBufferPtr>;

struct GptLayerOutputs {
    rtp_llm::BufferPtr hidden;
    rtp_llm::BufferPtr pre_decoder_residual;
    rtp_llm::BufferPtr moe_gating;
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
    rtp_llm::BufferPtr             token_ids;
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
    bool                               enable_sp       = false;
    size_t                             token_num       = 0;
    size_t                             pad_token_num   = 0;
    BufferPtr                          residual        = nullptr;
    bool                               need_moe_gating = false;
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

struct TokenSliceInfo {
    size_t offset = 0;
    size_t count  = 0;
};

struct ModelBufferHolder {
    std::vector<BufferPtr>     buffers;
    std::vector<torch::Tensor> tensors;

    void hold_host(const BufferPtr& buffer) {
        if (buffer && buffer->where() != MemoryType::MEMORY_GPU) {
            buffers.push_back(buffer);
        }
    }

    void hold_host(const torch::Tensor& tensor) {
        if (tensor.defined() && tensor.device().is_cpu()) {
            tensors.push_back(tensor);
        }
    }

    void hold(const BufferPtr& buffer) {
        if (buffer) {
            buffers.push_back(buffer);
        }
    }

    void hold(const torch::Tensor& tensor) {
        if (tensor.defined()) {
            tensors.push_back(tensor);
        }
    }

    void release() {
        buffers.clear();
        tensors.clear();
    }
};

class GptModel {
public:
    GptModel(const GptModelInitParams& params);
    virtual ~GptModel() {};

    virtual GptModelOutputs forward(const GptModelInputs& inputs);

    void releaseBuffers() {
        buffer_holder_.release();
    }

protected:
    rtp_llm::AttentionCommonInputs prepareAttentionInputs(const GptModelInputs& inputs,
                                                          rtp_llm::DataType     attn_dtype,
                                                          rtp_llm::BufferPtr    combo_position_ids);

    MicroBatchPlan planMicroBatches(const GptModelInputs& inputs);
    std::pair<std::vector<GptModelInputs>, std::vector<TokenSliceInfo>>
    splitInputsIntoMicroBatches(const GptModelInputs& inputs, const MicroBatchPlan& micro_batch_plan);
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
                                      const rtp_llm::BufferPtr merged_eagle3_hidden,
                                      bool                     skip_final_layernorm = false);

    void prepareExpertStats(const size_t layer_id, rtp_llm::FfnLayerParams& ffn_layer_params);

    void cleanExpertStats();

    void holdInputsHostBuffers(const GptModelInputs& inputs);

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

    ModelBufferHolder buffer_holder_;

public:
    rtp_llm::Weights            weights_;
    rtp_llm::OverallExpertStats overall_expert_stats_;
    size_t                      model_id_ = 0;
};

}  // namespace rtp_llm
