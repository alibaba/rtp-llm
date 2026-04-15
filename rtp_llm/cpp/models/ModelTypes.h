#pragma once

#include <atomic>
#include "rtp_llm/cpp/core/OpData.h"
#include "rtp_llm/cpp/models/models_weight/Weights.h"
#include "rtp_llm/cpp/models/eplb/stats/ExpertStats.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "rtp_llm/cpp/cache/BufferTypes.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/core/DeviceData.h"
#include <string>
#include <utility>
#include <memory>

namespace rtp_llm {

class KVCacheManager;  // Forward declaration

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
    const rtp_llm::Weights                weights;
    const GptModelDescription             description;
    const std::optional<CacheLayerLayout> kv_cache_layer_layout;
    size_t                                model_id;
    ParallelismConfig                     parallelism_config;
    ExecInitParams                        exec_init_params;
    std::shared_ptr<KVCacheManager>       cache_manager;  // For cache_store access during forward
};

enum GptModelInputIndex : size_t {
    comboTokens,
    inputLengths,
    sequenceLengths,
    prefixLengths,
    maxKernelBlocksPerBatch,
    maxBlocksPerBatch,
    kvCacheGroupNum,
    kvCacheLayerToGroupLen,
    kvCacheGroupTypesLen,
    kvCacheUpdateCopyNum,
    lmOutputIndexes,
    lmOutputLengthes,
    comboPositionIds,
    textTokensMask,
    mmFeaturesLocs,
    mmFeaturesNum,   // number of mm features
    mmFeaturesSize,  // hidden_size of mm features
    mmFeaturesDtype,
    needAllLogits,
    mtpHiddenStates,
    mtpHiddenStatesDtype,
    skipRun,
    gptModelRequestLength,  // length of request id & pd_separation
    isFakeStream,
    nanCheckEnabled,
    gptModelInputLength,
};

void tpSyncModelInputs(GptModelInputs& inputs, const ParallelismConfig& parallelism_config);

struct MicroBatchInfo {
    size_t prefill_num;
    size_t decoder_num;
};

struct MicroBatchPlan {
    bool                        enable = false;
    std::vector<MicroBatchInfo> batch_infos;
};

struct TokenSliceInfo {
    size_t offset = 0;
    size_t count  = 0;
};

struct ModelBufferHolder {
    std::vector<torch::Tensor> tensors;

    void hold_host(const torch::Tensor& tensor) {
        if (tensor.defined() && tensor.device().is_cpu()) {
            tensors.push_back(tensor);
        }
    }

    void hold(const torch::Tensor& tensor) {
        if (tensor.defined()) {
            tensors.push_back(tensor);
        }
    }

    void release() {
        tensors.clear();
    }
};

class ModelBase {
public:
    virtual ~ModelBase()                                          = default;
    virtual GptModelOutputs forward(const GptModelInputs& inputs) = 0;
    virtual void            releaseBuffers() {}
    virtual void            setNanCheckEnabled(bool enabled) { nan_check_enabled_ = enabled; }
    bool                    isNanCheckEnabled() const { return nan_check_enabled_; }

    rtp_llm::Weights            weights_;
    rtp_llm::OverallExpertStats overall_expert_stats_;
    size_t                      model_id_ = 0;

protected:
    std::atomic<bool> nan_check_enabled_{false};
};

}  // namespace rtp_llm
