#pragma once

#include "rtp_llm/models_py/bindings/core/OpData.h"
#include "rtp_llm/cpp/models/models_weight/Weights.h"
#include "rtp_llm/cpp/models/eplb/stats/ExpertStats.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "rtp_llm/cpp/cache/BufferTypes.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/models_py/bindings/core/DeviceData.h"
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
    size_t                                model_id = 0;
    ParallelismConfig                     parallelism_config;
    HWKernelConfig                        hw_kernel_config;
    ProfilingDebugLoggingConfig           profile_debug_logging_config;
    RuntimeConfig                         runtime_config;
    ConcurrencyConfig                     concurrency_config;
    SpeculativeExecutionConfig            sp_config;
    DeviceResourceConfig                  device_resource_config;
    MlaOpsType                            mla_ops_type            = MlaOpsType::AUTO;
    int64_t                               max_seq_len             = 0;
    int64_t                               hidden_size             = 0;
    size_t                                tokens_per_block        = 0;
    size_t                                kernel_tokens_per_block = 0;
    int32_t                               kv_cache_group_num      = 1;
    std::vector<int32_t>                  kv_cache_layer_to_group;
    std::shared_ptr<KVCacheManager>       cache_manager;
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

    rtp_llm::Weights            weights_;
    rtp_llm::OverallExpertStats overall_expert_stats_;
    size_t                      model_id_ = 0;
};

}  // namespace rtp_llm
