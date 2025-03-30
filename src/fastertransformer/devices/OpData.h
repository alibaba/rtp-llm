#pragma once

#include "src/fastertransformer/devices/Weights.h"
#include "src/fastertransformer/devices/LoraWeights.h"
#include "src/fastertransformer/devices/CommonDefines.h"
#include "src/fastertransformer/utils/activation_types.h"
#include "src/fastertransformer/utils/RopeConfig.h"
#include "src/fastertransformer/utils/MlaConfig.h"

#include "src/fastertransformer/core/Event.h"
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/core/QBuffer.h"
#include "src/fastertransformer/utils/activation_types.h"
#include "src/fastertransformer/utils/layernorm_types.h"
#include "src/fastertransformer/utils/EnumUtils.h"
#include "maga_transformer/cpp/utils/StackTrace.h"

#include <optional>
#include <functional>
#include <sstream>
#include <memory>
#include <torch/extension.h>

namespace rtp_llm {
class GptModelInputs;
}

namespace fastertransformer {

class DeviceBase;

enum class OpErrorType {
    ERROR_NONE,
    ERROR_INVALID_ARGS,
    ERROR_RESOURCE_EXHAUSTED,
    ERROR_UNIMPLEMENTED,
    ERROR_INTERNAL,
    ERROR_UNKNOWN,
};

enum class ParallelMode {
    TP = 0,
    DP = 1,
    DP_AND_TP = 2
    // DATA_PARALLEL = 2,
    // PIPELINE_PARALLEL = 3
};

class OpStatus {
public:
    OpStatus(OpErrorType, const std::string& message = "")
    : error_type(OpErrorType::ERROR_NONE), error_message(message) {}

    static OpStatus make(OpErrorType error_type, const std::string& error_message = "") {
        return OpStatus(error_type, error_message);
    }
    static OpStatus OK() { return OpStatus(OpErrorType::ERROR_NONE); }

    bool ok() const { return error_type == OpErrorType::ERROR_NONE; }
public:
    OpErrorType error_type;
    std::string error_message;
};

class OpException : public std::exception {
public:
    OpException(const OpStatus& status)
    : status_(status) {
        std::stringstream ss;
        ss << "OpException[" << (int32_t)status_.error_type << "]: " << status_.error_message << std::endl;
        FT_LOG_INFO("%s", ss.str().c_str());
        const auto stack = fastertransformer::getStackTrace();
        FT_STACKTRACE_LOG_INFO("%s", stack.c_str());
        ss << stack;
        detail_str_ = ss.str();
        if (std::getenv("FT_CORE_DUMP_ON_EXCEPTION")) {
            fflush(stdout);
            fflush(stderr);
            abort();
        }
    }

    const char* what() const noexcept override {
        return detail_str_.c_str();
    }

    const OpStatus& status() const { return status_; }

private:
    OpStatus status_;
    mutable std::string detail_str_;
};

using OptionalConstBufferRef    = std::optional<std::reference_wrapper<const Buffer>>;
using OptionalBufferRef         = std::optional<std::reference_wrapper<Buffer>>;
using OptionalConstVecBufferPtrRef = std::optional<std::reference_wrapper<const std::vector<BufferPtr>>>;


using CloneOutput = BufferPtr;



struct CloneParams {
    CloneParams(const Buffer& input,
                const AllocationType alloc_type = AllocationType::DEVICE,
                const BufferHints& hints = BufferHints())
    : input(input), alloc_type(alloc_type), hints(hints) {}

    const Buffer& input;
    const AllocationType alloc_type;
    const BufferHints& hints;
};

struct CopyParams {
    const Buffer& dst;
    const Buffer& src;
    bool overlapped = false;

    void check() const {
        FT_CHECK_WITH_INFO(src.type() == dst.type(),
                       "copy dst[%d] and src[%d] need has same type.",
                       src.type(), dst.type());
        FT_CHECK_WITH_INFO(src.sizeBytes() == dst.sizeBytes(),
            "src and dst copy size mismatch: [%s] vs [%s]",
            src.debugString().c_str(), dst.debugString().c_str());
    }
};

using SelectOutput = BufferPtr;

enum SelectType {
    LAST = 0,
    FIRST = 1,
};

struct SelectParams {
    const Buffer& input;
    const Buffer& index;
    size_t dim = 0;
};

using TransposeOutput = BufferPtr;

struct TransposeParams {
    const Buffer& input;
    bool overlapped = false;
};

using ConvertOutput = BufferPtr;

struct ConvertParams {
    const BufferPtr input;
    const DataType type;
};

using ConcatOutput = BufferPtr;

struct ConcatParams {
    const std::vector<BufferPtr>& inputs;
    const size_t dim = 0;
};

struct SplitOutput {
    std::vector<BufferPtr> outputs;
};

struct SplitParams {
    const Buffer& input;
    const std::vector<size_t>& split_sizes;
    const size_t dim = 0;
    bool overlapped = false;
};

struct LayernormOutput {
    BufferPtr output;
    BufferPtr before_norm_output;
};

struct AddBiasOutput {
    BufferPtr output;
};

struct LayernormParams {
    LayernormParams(BufferPtr input,
                    BufferPtr before_norm_output,
                    const std::optional<std::reference_wrapper<const LayerNormWeights>> norm_weight,
                    OptionalConstBufferRef residual1 = std::nullopt,
                    OptionalConstBufferRef residual2 = std::nullopt,
                    OptionalConstBufferRef bias = std::nullopt,
                    double alpha = 1.0f,
                    double eps = 1e-5,
                    bool is_inplace = true,
                    bool return_normed_output = false,
                    NormType norm_type = NormType::layernorm,
                    QScheme qscheme = QScheme::NoQuantize) :
                    input(std::move(input)),
                    before_norm_output(std::move(before_norm_output)),
                    norm_weight(norm_weight),
                    residual1(residual1),
                    residual2(residual2),
                    bias(bias),
                    norm_type(norm_type),
                    alpha(alpha),
                    eps(eps),
                    return_normed_output(return_normed_output),
                    is_inplace(is_inplace),
                    qscheme(qscheme),
                    offset(0),
                    hidden_size(0),
                    stride(0) {};

    // for qk norm
    LayernormParams(BufferPtr input,
                    const std::optional<std::reference_wrapper<const LayerNormWeights>> norm_weight,
                    double eps,
                    NormType norm_type,
                    size_t offset,
                    size_t hidden_size,
                    size_t stride
                ):
                    input(std::move(input)),
                    before_norm_output(nullptr),
                    norm_weight(norm_weight),
                    residual1(std::nullopt),
                    residual2(std::nullopt),
                    bias(std::nullopt),
                    norm_type(norm_type),
                    alpha(0.0),
                    eps(eps),
                    return_normed_output(false),
                    is_inplace(true),
                    qscheme(QScheme::NoQuantize),
                    offset(offset),
                    hidden_size(hidden_size),
                    stride(stride) {};


    BufferPtr input;
    BufferPtr before_norm_output;

    const std::optional<std::reference_wrapper<const LayerNormWeights>> norm_weight;

    const OptionalConstBufferRef  residual1;
    const OptionalConstBufferRef  residual2;
    const OptionalConstBufferRef  bias;

    const NormType norm_type;

    const double alpha;
    const double eps;

    const bool return_normed_output;
    const bool is_inplace;
    const QScheme qscheme;

    const int offset;
    const int hidden_size;
    const int stride;
};

enum GemmType : size_t {
    InvalidGemm = 0,

    BufferA_BufferB_BufferC_2DGemm,
    BufferA_BufferB_BufferC_3DGemm,

    QBufferA_BufferB_BufferC_2DGemm,
    BufferA_QBufferB_BufferC_2DGemm,
    QBufferA_QBufferB_BufferC_2DGemm,
};

struct AddBiasParams {
    BufferPtr     input;
    const Buffer& bias;
    bool          inplace = true;
};

// D = alpha * op(A) * op(B) + beta * C
// shapes of A, B, C, D have two options: [m, k], [k, n], [m, n] / [1, n], [m, n]
// or [bs, m, k], [bs, k, n], [bs, m, n], [bs, m, n] where bs is batch_size
// D is optional, if not passed, it will be allocated by the op
struct GemmParams {
    GemmParams(const Buffer& A,
               const Buffer& B,
               OptionalConstBufferRef C = std::nullopt,
               BufferPtr D = nullptr,
               const DataType compute_type = DataType::TYPE_INVALID,
               TransposeOperation transA = TransposeOperation::NONE,
               TransposeOperation transB = TransposeOperation::NONE,
               const ActivationType activationType = ActivationType::Identity,
               const float alpha = 1.0f,
               const float beta  = 0.0f):
               A(A),
               B(B),
               C(C),
               D(D),
               compute_type(compute_type),
               transA(transA),
               transB(transB),
               activationType(activationType),
               alpha(alpha),
               beta(beta) {}


    const Buffer& A;
    const Buffer& B;
    OptionalConstBufferRef C;
    BufferPtr D;
    const DataType compute_type = DataType::TYPE_INVALID; // If passed invalid type, op should infer type

    const TransposeOperation transA = TransposeOperation::NONE;
    const TransposeOperation transB = TransposeOperation::NONE;

    ActivationType activationType = ActivationType::Identity;

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    void check() const;
    GemmType dispatch() const;
};

struct GroupedGemmOutput {
    std::vector<BufferPtr> output;
};

// C = alpha * op(A) * op(B) + beta * C
// shapes of each A, B, C needs to be [m, k], [k, n], [m, n]
struct GroupedGemmParams {


    const std::vector<BufferPtr>& A;
    const std::vector<BufferPtr>& B;
    std::optional<std::vector<BufferPtr>> C = std::nullopt;
    const float alpha = 1.0f;
    const float beta = 1.0f;

    void check() const;
};

using MultiplyOutput = BufferPtr;

// output = A * B
// A: [m], B: [m] or [m, dim_1, ..., dim_n]
struct MultiplyParams {
    const Buffer& A;
    const Buffer& B;
    BufferPtr output = nullptr;
};

struct EmbeddingLookupParams {
    const Buffer& combo_tokens;
    const Buffer& embedding_table;
    double input_embedding_scalar = 1;

    OptionalConstBufferRef text_tokens_mask;

    OptionalConstBufferRef position_ids;
    OptionalConstBufferRef position_table;

    OptionalConstBufferRef token_types;
    OptionalConstBufferRef token_type_table;
};

struct KvCacheInfo {
    int layer_num;
    BufferPtr kv_cache_block_id;  // [batch_size, block_nums], kv cache block offset
    BufferPtr k_cache_buffer;   // [block_nums, head, seq_size_per_block, size_per_head]
    BufferPtr v_cache_buffer;   // [block_nums, head, seq_size_per_block, size_per_head]
    BufferPtr k_scale_buffer;   // [block_nums, head, seq_size_per_block]
    BufferPtr v_scale_buffer;   // [block_nums, head, seq_size_per_block]
};

struct MultimodalEmbeddingParams {
    const BufferPtr& word_embeddings;
    OptionalConstVecBufferPtrRef multimodal_features;
    OptionalConstBufferRef multimodal_locs;
};

using MultimodalEmbeddingOutput = BufferPtr;

struct CacheStoreInputs {
    BufferPtr input_lengths_host;
    BufferPtr prefix_lengths_host;
    BufferPtr host_kv_cache_offset;
};

using FlashInferAttnParamsPtr = std::shared_ptr<void>;

struct AttentionCommonInputs {
    // see detailed comments at GptModelInputs
    ConstBufferPtr input_lengths;      // int32_t, [decoder_batch_size + context_batch_size]
    ConstBufferPtr sequence_lengths;   // int32_t, [decoder_batch_size]

    std::optional<KvCacheInfo>      kv_cache;
    std::optional<CacheStoreInputs> cache_store_inputs;

    ConstBufferPtr cu_seqlens;
    ConstBufferPtr cu_kv_seqlens;
    ConstBufferPtr padding_offset;

    size_t context_batch_size = 0;
    size_t decoder_batch_size = 0;
    size_t context_max_seq_len = 0;
    size_t decoder_max_seq_len = 0;
    size_t context_token_num = 0;

    BufferPtr position_ids;
    BufferPtr attention_mask;
    ConstBufferPtr linear_bias_slopes;
    BufferPtr prefix_prompt_lengths;
    int32_t   max_prefix_length = 0;

    lora::AttentionLayerLoraInput lora_input;

    int layer_id = 0;
    BufferPtr                                 request_id;               // [context_batch_size]
    BufferPtr                                 request_pd_separation;    // [context_batch_size]
    std::vector<std::string>                  cache_keys;               // [context_batch_size]
    size_t                                    k_block_size = 0;
    size_t                                    v_block_size = 0;
    size_t                                    scale_block_size = 0;
    bool                                      pd_separation = false;

    bool warmup;

    FlashInferAttnParamsPtr flash_infer_attn_params;
};

struct AttentionConfigs {
    size_t      head_num;
    size_t      kv_head_num;
    size_t      size_per_head;
    size_t      hidden_size;

    // rotary embending config
    RopeConfig rope_config;

    //kv cache block
    size_t tokens_per_block;

    AttentionMaskType mask_type = noMask;
    float q_scaling = 1.0f;
    bool fuse_qkv_add_bias = true;
    bool use_logn_attn = false;

    // mla config
    bool       use_mla = false;
    size_t     q_lora_rank;
    size_t     kv_lora_rank;
    size_t     nope_head_dim;
    size_t     rope_head_dim;
    size_t     v_head_dim;

    // softmax config
    float softmax_extra_scale = 1.0f;
    KvCacheDataType kv_cache_dtype = KvCacheDataType::BASE;
};

using AttentionModuleOutput = void;

struct AttentionModuleParams {
    const int32_t                   layer_id;
    // qkv shape[h_token_num, (head_num + 2 * kv_head_num) * size_per_head]
    const Buffer&                   input;
    Buffer&                         output; // shape [token_num, size_per_head]

    AttentionCommonInputs&          common;
    const AttentionLayerWeights&    weights;
    const AttentionConfigs&         configs;
    const QScheme                   qscheme;
};

struct MlaRotaryWriteKVCacheParams {
    const Buffer&                   q;
    const Buffer&                   ckv;
    const Buffer&                   k_rope;

    AttentionCommonInputs&          common;
    const AttentionLayerWeights&    weights;
    const AttentionConfigs&         configs;
    const QScheme                   qscheme;
};

struct MlaAttentionModuleParams {
    const int32_t                   layer_id;
    const Buffer&                   q;
    const Buffer&                   kv_a;
    const Buffer&                   k_rope;
    BufferPtr                       qkv_output; // shape [token_num, hidden_size]

    AttentionCommonInputs&          common;
    const AttentionLayerWeights&    weights;
    const AttentionConfigs&         configs;
    const QScheme                   qscheme;
};

// decoder attention read kv_a/k_rope from kvcache
struct MlaDecoderAttentionParams{
    const int32_t                   layer_id;
    const Buffer&                   q;
    BufferPtr                       qkv_output; // shape [token_num, hidden_size]

    AttentionCommonInputs&          common;
    const AttentionLayerWeights&    weights;
    const AttentionConfigs&         configs;
    const QScheme                   qscheme;

    BufferPtr                       bmm_indices;

    MlaDecoderAttentionParams(const int32_t layer_id, const Buffer& q, BufferPtr qkv_output, AttentionCommonInputs& common, 
        const AttentionLayerWeights& weights, const AttentionConfigs& configs, const QScheme qscheme, const BufferPtr bmm_indices = nullptr): 
            layer_id(layer_id), q(q), qkv_output(qkv_output), common(common), weights(weights), configs(configs), qscheme(qscheme), bmm_indices(bmm_indices) {}
};

struct WriteCacheParams {
    AttentionCommonInputs&          common;
    const AttentionConfigs&         configs;
    bool                            mla_kvcache = false;

    WriteCacheParams(const AttentionModuleParams& params)
    : common(params.common), configs(params.configs) {}

    WriteCacheParams(const MlaAttentionModuleParams& params)
    : common(params.common), configs(params.configs), mla_kvcache(true) {}
};

struct AttentionLayerOutput {
    BufferPtr hidden_states;
};

struct LayerNormConfig {
    double eps;
    NormType norm_type;
};

struct AttentionLayerParams {
    int32_t                         layer_id;
    const Buffer&                   input;
    BufferPtr                       output;
    const AttentionConfigs&         configs;
    const AttentionLayerWeights&    weights;
    AttentionCommonInputs&          common;
    const OptionalConstBufferRef    residual; // for intel xft
    const LayerNormConfig           ln_params;
    const QScheme                   qscheme;
};

struct MoeConfigs {
    size_t expert_num;
    size_t top_k;

    bool    normalize_expert_scale = false;
    int64_t moe_inter_padding_size = 0;
    bool    has_moe_norm           = false;
    size_t  ep_rank                = 0;
    size_t  ep_size                = 1;
    size_t  tp_rank                = 0;
    size_t  tp_size                = 1;
    size_t  dp_rank                = 0;
    size_t  dp_size                = 1;

    int scoring_func = 0;  // 0: softmax, 1: sigmoid
    int topk_group   = 1;
    int n_group      = 1;
};

struct FfnConfigs {
    ActivationType activation_type;
    std::optional<MoeConfigs> moe_configs = std::nullopt;
};

struct FfnLayerOutput {
    BufferPtr hidden_states;
    DeviceHookPtr comm_barrier_hook;
};

struct FfnLayerParams {
    FfnLayerParams(const Buffer&                input,
                   const FfnConfigs&            configs,
                   const FfnLayerWeights&       weights,
                   const OptionalConstBufferRef residual = std::nullopt,
                   const QScheme                qscheme  = QScheme::NoQuantize,
                   BufferPtr                    output = nullptr):
        input(input), configs(configs), weights(weights), residual(residual),
        qscheme(qscheme), output(std::move(output)){}

    const Buffer& input;
    const FfnConfigs&            configs;
    const FfnLayerWeights&       weights;

    const OptionalConstBufferRef residual; // for intel xft

    const QScheme qscheme;
    BufferPtr                    output;

    lora::FfnLayerLoraInput      lora_input;
};

struct MoeGateSelectOutput {
    BufferPtr expert_ids;
    BufferPtr expert_scales;
};

struct MoeDispatchOutput {
    BufferPtr                    hidden;
    BufferPtr                    expert_ids;
    BufferPtr                    expert_scales;
    BufferPtr                    indices;
    const std::vector<size_t>    input_split_sizes;
    const std::vector<size_t>    output_split_sizes;
    const std::vector<BufferPtr> dispatch_src_buffers; // to make them outlive async sendrecv
    const BufferPtr              concated_src_buffers; // to make them outlive async sendrecv
    const BufferPtr              split_dst_buffers;    // to make them outlive async sendrecv
    DeviceHookPtr                comm_barrier_hook;
};

struct MoeDispatchParams {
    MoeDispatchParams(const Buffer&     input,
                      const Buffer&     expert_ids,
                      const Buffer&     expert_scales,
                      const MoeConfigs& moe_configs,
                      bool              overlapped = false,
                      const QScheme     qscheme  = QScheme::NoQuantize):
        input(input), expert_ids(expert_ids), expert_scales(expert_scales), moe_configs(moe_configs), overlapped(overlapped), qscheme(qscheme) {}

    const Buffer&     input;
    const Buffer&     expert_ids;
    const Buffer&     expert_scales;
    const MoeConfigs& moe_configs;
    bool              overlapped = false;
    const QScheme     qscheme;
};

struct MoeCombineParams {
    const BufferPtr           input;
    const BufferPtr           indices;
    const BufferPtr           output;
    const std::vector<size_t>& input_split_sizes;
    const std::vector<size_t>& output_split_sizes;
    const MoeConfigs&         moe_configs;
    size_t                    origin_token_num;
    bool                      overlapped = false;
};

struct GreedyParams {
    const Buffer& logits;                    // [batch_size, vocab_size_padded]
    const Buffer& input_lengths;             // [batch_size]
    const Buffer& sequence_lengths;          // [batch_size]
    Buffer& token_ids;                       // [batch_size, max_input_length + 1]
    const size_t step;

    const Buffer& top_k;
    const Buffer& top_p;
    const Buffer& temperature;
    OptionalBufferRef random_seed;
    OptionalBufferRef repetition_penalty;
    OptionalBufferRef min_lengths;
    OptionalBufferRef eos_ids;
    OptionalBufferRef no_repeat_ngram_size;

    OptionalBufferRef cum_log_probs;
    OptionalBufferRef output_log_probs;

    OptionalBufferRef output_all_probs;
};

struct GreedyOutput {
    BufferPtr success;
    // BufferPtr new_tokens;
};


struct BeamSearchParams {
    const Buffer& logits;
    Buffer& token_ids;
    Buffer& input_lengths;
    Buffer& sequence_lengths;
    Buffer& cum_log_probs;
    Buffer& beam_index;
};

struct BroadcastParams {
    const std::vector<BufferPtr>& buffers;
    const int64_t root;
    ParallelMode mode = ParallelMode::TP;
    bool overlapped = false;
};

enum class ReduceOp {
    Sum = 0,
    Prod = 1,
    Max = 2,
    Min = 3,
    Avg = 4,
};


struct PrepareAllReduceParams {
    const BufferPtr buffer;
    const ReduceOp op;
    ParallelMode mode = ParallelMode::TP;
};

struct PrepareAllReduceOutput {
    const BufferPtr buffer;
};

struct AllReduceParams {
    const BufferPtr buffer;
    const ReduceOp op;
    bool overlapped = false;
    ParallelMode mode = ParallelMode::TP;
};

struct AllReduceOutput {
    const BufferPtr buffer;
};

struct AllGatherParams {
    const std::vector<BufferPtr>& buffers;
    ParallelMode mode = ParallelMode::TP;
    bool overlapped = false;
};

struct AllToAllParams {
    const std::vector<BufferPtr> buffers;
    const std::vector<size_t>    input_split_sizes;
    const std::vector<size_t>    output_split_sizes;
    bool                         overlapped = false;
    ParallelMode                 mode       = ParallelMode::DP_AND_TP;
};

struct AllToAllOutput {
    std::vector<BufferPtr> outputs;
    BufferPtr              concated_input;
    BufferPtr              output_to_split;
    DeviceHookPtr          comm_barrier_hook;
};

// output = act(input) + bias
struct ActivationParams {
    ActivationType atype;
    // can be nullptr for fuse gemm with activation
    BufferPtr states;

    const OptionalConstBufferRef bias      = std::nullopt;
    const OptionalConstBufferRef gate      = std::nullopt;
    const OptionalConstBufferRef gate_bias = std::nullopt;
    const OptionalConstBufferRef act_scale = std::nullopt;

    ActivationParams(ActivationType               atype,
                     BufferPtr                    states,
                     const OptionalConstBufferRef bias,
                     const OptionalConstBufferRef gate,
                     const OptionalConstBufferRef gate_bias,
                     const OptionalConstBufferRef act_scale):
        atype(atype), states(states), bias(bias), gate(gate), gate_bias(gate_bias), act_scale(act_scale) {}

    ActivationParams(ActivationType atype, BufferPtr states): atype(atype), states(states), bias(std::nullopt), gate(std::nullopt), gate_bias(std::nullopt), act_scale(std::nullopt) {};
};

// softmax op is inplace-update, thus output buffer is same as input
struct SoftmaxParams {
    const BufferPtr input;
    const OptionalConstBufferRef mask = std::nullopt;
    const OptionalConstBufferRef bias = std::nullopt;
    float scale = 1.0f;
    const DataType output_t = DataType::TYPE_INVALID;
    const OptionalConstBufferRef linear_bias_slopes = std::nullopt;
};

struct LossParams {
    const Buffer& logits;
    const Buffer& labels;
};

using LossOutput = BufferPtr;

struct MaskParams {
public:
    const Buffer& input_lengths;
    const Buffer& prefix_lengths;
    DataType      dtype;
    bool          is_causal;
};

using MaskOutput = BufferPtr;

struct DevicePrepParams {
    const AttentionConfigs& configs;

    const BufferPtr &sequence_lengths;
    const BufferPtr &input_lengths;
    const BufferPtr &kv_cache_block_id;

    DataType dtype = DataType::TYPE_INVALID;
    size_t context_batch_size = 0;
    size_t decoder_batch_size = 0;
    bool has_kv_cache     = true;
    bool diff_qkv_len     = false;
    bool has_alibi_slopes = false;
};

struct DevicePrepOutput {
    bool need_mask = true;
    FlashInferAttnParamsPtr flash_infer_attn_params;
};

struct LoraLinearOutput {
    BufferPtr output;
};

struct LoraLinearParams {

    LoraLinearParams(GemmParams& gemm_params,
                     lora::LoraOpInputPtr lora_input = nullptr) :
                     gemm_params(gemm_params),
                     lora_input(lora_input) {}

    GemmParams& gemm_params;
    lora::LoraOpInputPtr lora_input;
};

struct LoraLinearWithActivationParams {
    const LoraLinearParams& lora_linear_params;
    const ActivationParams& activation_params;
    LoraLinearWithActivationParams(const LoraLinearParams& lora_linear_params, const ActivationParams& activation_params): lora_linear_params(lora_linear_params), activation_params(activation_params) {}
};
struct QuantizeParams {
    const Buffer&           input;
    DataType                qtype;
    size_t                  axis;
    QScheme                 qscheme;

    // for soomth quantize
    OptionalConstBufferRef  smoother;
    OptionalConstBufferRef  shift;

    // for static quantize
    OptionalConstBufferRef  static_scale;
    OptionalConstBufferRef  static_scale_reciprocal;

    // for groupwise quantize
    int64_t    groupSize;

    QuantizeParams(const Buffer&          input,
                   DataType               qtype,
                   size_t                 axis,
                   QScheme                qscheme,
                   OptionalConstBufferRef smoother,
                   OptionalConstBufferRef shift,
                   OptionalConstBufferRef static_scale,
                   OptionalConstBufferRef static_scale_reciprocal):
        input(input),
        qtype(qtype),
        axis(axis),
        qscheme(qscheme),
        smoother(smoother),
        shift(shift),
        static_scale(static_scale),
        static_scale_reciprocal(static_scale_reciprocal),
        groupSize(64) {}
    QuantizeParams(const Buffer& input, DataType qtype, size_t axis):
        input(input),
        qtype(qtype),
        axis(axis),
        qscheme(QScheme::Qint8PerToken),
        groupSize(64) {}
    QuantizeParams(const Buffer& input, DataType qtype, size_t axis, int64_t groupSize):
        input(input),
        qtype(qtype),
        axis(axis),
        qscheme(QScheme::Qint8PerToken),
        groupSize(groupSize) {}
    QuantizeParams(const Buffer& input, DataType qtype, size_t axis, QScheme qscheme):
        input(input),
        qtype(qtype),
        axis(axis),
        qscheme(qscheme){}
};

}  // namespace fastertransformer
