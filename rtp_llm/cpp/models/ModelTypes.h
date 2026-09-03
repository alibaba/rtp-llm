#pragma once

#include "rtp_llm/models_py/bindings/core/OpData.h"
#include "rtp_llm/cpp/models/models_weight/Weights.h"
#include "rtp_llm/cpp/models/eplb/stats/ExpertStats.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "rtp_llm/cpp/cache/BufferTypes.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/models_py/bindings/core/DeviceData.h"
#include "rtp_llm/models_py/bindings/core/TensorHolder.h"
#include <array>
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
    rtp_llm::QScheme          act_qscheme              = rtp_llm::QScheme::NoQuantize;
    const DataType            compute_type             = rtp_llm::DataType::TYPE_INVALID;
    double                    layernorm_eps            = 1e-5;
    size_t                    vocab_size               = 0;
    size_t                    output_vocab_size        = 0;
    size_t                    output_vocab_padded_size = 0;
    bool                      post_layernorm           = false;
    double                    input_embedding_scalar   = 1;
    double                    residual_scalar          = 1;
    bool                      reverse_e_h_norm         = false;
    // Runtime MoE strategy selected by server configuration. Keep it with the
    // description produced by Executor::genModelDescription so every model
    // construction path receives the same fail-closed eligibility inputs.
    MoeConfig moe_runtime_config;
};

struct GptModelInitParams {
    const rtp_llm::Weights                       weights;
    const GptModelDescription                    description;
    const std::optional<GroupedCacheLayerLayout> kv_cache_layer_layout;
    size_t                                       model_id = 0;
    ParallelismConfig                            parallelism_config;
    HWKernelConfig                               hw_kernel_config;
    ProfilingDebugLoggingConfig                  profile_debug_logging_config;
    RuntimeConfig                                runtime_config;
    ConcurrencyConfig                            concurrency_config;
    SpeculativeExecutionConfig                   sp_config;
    DeviceResourceConfig                         device_resource_config;
    MlaOpsType                                   mla_ops_type            = MlaOpsType::AUTO;
    int64_t                                      max_seq_len             = 0;
    int64_t                                      hidden_size             = 0;
    size_t                                       tokens_per_block        = 0;
    size_t                                       kernel_tokens_per_block = 0;
    std::shared_ptr<KVCacheManager>              cache_manager;
    // nullopt selects the main-model cache config; otherwise selects this MTP module config.
    std::optional<int> mtp_cache_config_index;
    // DSv4 head-channel residual multiplier. Default 1 (no expansion).
    // When >1, the model's pre-output residual ([T, hc_mult*hidden_size])
    // is the contract between target and draft for MTP — see
    // makeFakeSPOutputBuffer (MtpExecutor.cc) and CudaGraphRunner
    // input_hiddens.
    int64_t hc_mult = 1;
};

enum GptModelInputIndex : size_t {
    comboTokens,
    inputLengths,
    sequenceLengths,
    prefixLengths,
    maxKernelBlocksPerBatch,
    maxBlocksPerBatch,
    cacheKeysWidth,
    kvCacheGroupNum,
    kvCacheLayerToGroupLen,
    kvCacheGroupTypesLen,
    kvCacheUpdateCopyNum,
    lmOutputIndexes,
    comboPositionIds,
    textTokensMask,
    mmFeaturesLocs,
    mmFeaturesNum,   // number of mm features
    mmFeaturesSize,  // hidden_size of mm features
    mmFeaturesDtype,
    mmHasExtraInput,    // number of extra-input tensors (model-specific, e.g. deepstack)
    mmExtraInputDtype,  // dtype of extra-input elements
    needAllLogits,
    needAllHiddenStates,
    mtpHiddenStates,
    mtpHiddenStatesDtype,
    skipRun,
    gptModelRequestLength,  // length of request id & pd_separation
    isFakeStream,
    // last_hidden_states can have a different row count from combo_tokens for
    // DSpARK prefill seeding, so transmit its leading dimension explicitly.
    mtpHiddenStatesRows,
    // Per-tensor device hint bitmap from root so non-root ranks allocate
    // matching GPU buffers and keep tpSync broadcast lanes consistent.
    tensorDeviceMap,
    gptModelInputLength,
};

// Wire format of the first tpSync stage. int64, because a DSpARK long-prefill
// last_hidden_states easily exceeds what int32 can carry (256K tokens *
// 12288 = 3.2G elements).
using GptModelInputShapeHints = std::array<int64_t, GptModelInputIndex::gptModelInputLength>;

// Bit positions for `tensorDeviceMap`. Only fields that participate in the
// MTP/Eagle decode-prepare GPU path need a bit; other fields stay CPU.
enum GptModelInputDeviceBit : uint32_t {
    kDeviceBitComboTokens     = 1u << 0,
    kDeviceBitInputLengths    = 1u << 1,
    kDeviceBitSequenceLengths = 1u << 2,
    kDeviceBitPrefixLengths   = 1u << 3,
    kDeviceBitLmOutputIndexes = 1u << 4,
    kDeviceBitKernelBlockId   = 1u << 5,
};

GptModelInputShapeHints getModelInputShapeHints(const GptModelInputs& inputs);
torch::Tensor           makeModelInputShapeHintsTensor(const GptModelInputs& inputs);
std::array<int64_t, 2>  decodeMtpHiddenStatesShape(int64_t total_numel, int64_t rows);

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

class ModelBase {
public:
    virtual ~ModelBase()                                          = default;
    virtual GptModelOutputs forward(const GptModelInputs& inputs) = 0;
    virtual void            releaseBuffers() {}
    virtual void            prepareAttentionInputs(const GptModelInputs& inputs) {}

    // Refresh only kv_cache_kernel_block_id-dependent state on a previously-
    // prepared attention_inputs_ (e.g., after an MTP propose+verify re-gather).
    // No-op when no attention inputs have been prepared yet.
    virtual void updateKVCacheKernelBlockId(const GptModelInputs& inputs) {}

    // Optional spec-decode hand-off: target model exposes the pre-output-projection
    // residual buffer (DSv4: pre-``hc_head`` ``[T, hc*D]``) so MtpExecutor can
    // normalize it into ``GptModelOutputs::all_hidden_states`` instead of the
    // post-reduce ``[T, D]``. Default returns an empty Tensor (model has no such
    // buffer); ``PyWrappedModel`` overrides to call the Python accessor.
    //
    // The producer writes the buffer in verify (req-major) layout
    // ``[r0_v0, r0_v1, …, r0_v_ps, r1_v0, …]``: each request occupies
    // ``propose_step + 1`` consecutive rows. ``num_tokens >= 0`` returns that
    // many leading rows; ``num_tokens < 0`` asks the producer for its last
    // written row count (CP prefill keeps rank-local rows in the buffer, so
    // the count cannot be derived from the global token count).
    virtual torch::Tensor getMtpTargetHiddenStates(int64_t /*num_tokens*/) {
        return torch::Tensor();
    }

    // Capability query used while initializing a speculative target model.
    // Prefill CP cannot recover the full target hidden sequence from its
    // last-hidden output, so initialization must know whether this rank-local
    // hand-off exists before admitting CPRR execution.
    virtual bool hasMtpTargetHiddenBuffer() const {
        return false;
    }

    // Optional CP-prefill companion: expose one final pre-hc row per request
    // (DSv4: [B, hc*D]) so MTP stream update does not require a full sequence
    // hidden buffer. Passing num_tokens < 0 asks the producer for its last
    // written row count.
    virtual torch::Tensor getMtpLastHiddenStates(int64_t /*num_tokens*/) {
        return torch::Tensor();
    }

    rtp_llm::Weights            weights_;
    rtp_llm::OverallExpertStats overall_expert_stats_;
    size_t                      model_id_ = 0;
};

}  // namespace rtp_llm
