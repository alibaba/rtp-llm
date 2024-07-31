#pragma once

#include "src/fastertransformer/devices/Weights.h"
#include "src/fastertransformer/devices/CommonDefines.h"
#include "src/fastertransformer/utils/activation_types.h"
#include "src/fastertransformer/utils/RopeTypes.h"

#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/core/QBuffer.h"
#include "src/fastertransformer/utils/activation_types.h"
#include "src/fastertransformer/utils/layernorm_types.h"
#include "src/fastertransformer/utils/EnumUtils.h"

#include <optional>
#include <functional>
#include <sstream>
#include <memory>
#include <torch/python.h>

namespace fastertransformer {

enum class OpErrorType {
    ERROR_NONE,
    ERROR_INVALID_ARGS,
    ERROR_RESOURCE_EXHAUSTED,
    ERROR_UNIMPLEMENTED,
    ERROR_INTERNAL,
    ERROR_UNKNOWN,
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
        if (std::getenv("FT_CORE_DUMP_ON_EXCEPTION")) {
            fflush(stdout);
            fflush(stderr);
            abort();
        }
    }

    const char* what() const noexcept override {
        std::stringstream ss;
        ss << "OpException[" << (int32_t)status_.error_type << "]: " << status_.error_message;
        s_ = ss.str();
        return s_.c_str();
    }

    const OpStatus& status() const { return status_; }
private:
    OpStatus status_;
    mutable std::string s_;
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
                    stride(0) {};

    // for qk norm
    LayernormParams(BufferPtr input,
                    const std::optional<std::reference_wrapper<const LayerNormWeights>> norm_weight,
                    double eps,
                    NormType norm_type,
                    size_t offset,
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
    bool          inplace;
};

// D = alpha * op(A) * op(B) + beta * C
// shapes of A, B, C, D have two options: [m, k], [k, n], [m, n], [m, n]
// or [bs, m, k], [bs, k, n], [bs, m, n], [bs, m, n] where bs is batch_size
// D is optional, if not passed, it will be allocated by the op
struct GemmParams {
    GemmParams(const Buffer& A,
               const Buffer& B,
               OptionalBufferRef C = std::nullopt,
               BufferPtr D = nullptr,
               const DataType compute_type = DataType::TYPE_INVALID,
               TransposeOperation transA = TransposeOperation::NONE,
               TransposeOperation transB = TransposeOperation::NONE):
               A(A),
               B(B),
               C(C),
               D(D),
               compute_type(compute_type),
               transA(transA),
               transB(transB) {}


    const Buffer& A;
    const Buffer& B;
    OptionalBufferRef C;
    BufferPtr D;
    const DataType compute_type = DataType::TYPE_INVALID; // If passed invalid type, op should infer type

    const TransposeOperation transA = TransposeOperation::NONE;
    const TransposeOperation transB = TransposeOperation::NONE;

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
    BufferPtr kv_cache_offset;  // [batch_size, block_nums], kv cache block offset
    BufferPtr k_cache_buffer;   // [layer_num, block_nums, head, seq_size_per_block, size_per_head]
    BufferPtr v_cache_buffer;   // [layer_num, block_nums, head, seq_size_per_block, size_per_head]
    BufferPtr k_scale_buffer;   // [layer_num, block_nums, head, seq_size_per_block]
    BufferPtr v_scale_buffer;   // [layer_num, block_nums, head, seq_size_per_block]
};

struct MultimodalEmbeddingParams {
    const BufferPtr& word_embeddings;
    OptionalConstVecBufferPtrRef multimodal_features;
    OptionalConstBufferRef multimodal_locs;
};

struct AttentionCommonInputs {
    // see detailed comments at GptModelInputs
    const Buffer& input_lengths;      // int32_t, [decoder_batch_size + context_batch_size]
    const Buffer& sequence_lengths;   // int32_t, [decoder_batch_size]

    std::optional<KvCacheInfo> kv_cache;
    ConstBufferPtr cu_seqlens;
    ConstBufferPtr cu_kv_seqlens;
    ConstBufferPtr padding_offset;

    size_t context_batch_size;
    size_t decoder_batch_size;
    size_t context_max_seq_len;
    size_t decoder_max_seq_len;
    size_t context_token_num;

    BufferPtr position_ids;
    BufferPtr attention_mask;
    ConstBufferPtr linear_bias_slopes;
    BufferPtr prefix_prompt_lengths;
    int32_t   max_prefix_length;
    OptionalLoraInput lora_input = std::nullopt;
    FMHAType          fmha_type  = FMHAType::NONE;
};

struct AttentionConfigs {
    size_t      head_num;
    size_t      kv_head_num;
    size_t      size_per_head;

    // rotary embending config
    RopeConfig rope_config;

    //kv cache block
    size_t tokens_per_block;

    AttentionMaskType mask_type = noMask;
    float q_scaling = 1.0f;
    bool fuse_qkv_add_bias = true;
    bool use_logn_attn = false;
};

using AttentionModuleOutput = void;

struct AttentionModuleParams {
    // qkv shape[h_token_num, (head_num + 2 * kv_head_num) * size_per_head]
    const Buffer&                   input;
    Buffer&                         output; // shape [token_num, size_per_head]

    AttentionCommonInputs&          common;
    const AttentionLayerWeights&    weights;
    const AttentionConfigs&         configs;
};

struct AttentionLayerOutput {
    BufferPtr hidden_states;
};

struct LayerNormConfig {
    double eps;
    NormType norm_type;
};

struct AttentionLayerParams {
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

    bool normalize_expert_scale        = false;
    int64_t moe_inter_padding_size     = 0;
    bool has_moe_norm                  = false;
};

struct FfnConfigs {
    ActivationType activation_type;
    std::optional<MoeConfigs> moe_configs = std::nullopt;
};

struct FfnLayerOutput {
    BufferPtr hidden_states;
};

struct FfnLayerParams {
    FfnLayerParams(const Buffer&                input,
                   const FfnConfigs&            configs,
                   const FfnLayerWeights&       weights,
                   const OptionalConstBufferRef residual = std::nullopt,
                   const QScheme                qscheme  = QScheme::NoQuantize,
                   BufferPtr                    output = nullptr):
        input(input), configs(configs), weights(weights), residual(residual), qscheme(qscheme), output(std::move(output)){}

    const Buffer& input;
    const FfnConfigs&            configs;
    const FfnLayerWeights&       weights;

    const OptionalConstBufferRef residual; // for intel xft

    const QScheme qscheme;
    BufferPtr                    output;
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

    OptionalBufferRef cum_log_probs;
    OptionalBufferRef output_log_probs;
    OptionalBufferRef output_index_probs;
};

struct BeamSearchParams {
    const Buffer& logits;
    const Buffer& sequence_lengths;  // shape: [batch_size]
    const size_t step;               // typically largest sequence length in the batch

    const size_t num_beams;
    const size_t batch_size;

    Buffer& token_ids;
    OptionalBufferRef cum_log_probs;
    OptionalBufferRef output_log_probs;
};

struct BeamSearchOutput {
    BufferPtr token_ids;
    BufferPtr cum_log_probs;
};

struct BroadcastParams {
    const std::vector<BufferPtr>& buffers;
    const int64_t root;
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
};

struct PrepareAllReduceOutput {
    const BufferPtr buffer;
};

struct AllReduceParams {
    const BufferPtr buffer;
    const ReduceOp op;
};

struct AllReduceOutput {
    const BufferPtr buffer;
};

struct AllGatherParams {
    const std::vector<BufferPtr>& buffers;
};

// output = act(input) + bias
struct ActivationParams {
    ActivationType atype;
    Buffer& states;

    const OptionalConstBufferRef bias      = std::nullopt;
    const OptionalConstBufferRef gate      = std::nullopt;
    const OptionalConstBufferRef gate_bias = std::nullopt;
    const OptionalConstBufferRef act_scale = std::nullopt;
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
    int calculate_loss = 0;
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
    DataType dtype;
    size_t context_batch_size;
    bool has_kv_cache     = true;
    bool diff_qkv_len     = false;
    bool int8_kv_cache    = false;
    bool has_alibi_slopes = false;
    bool sprase_head      = false;
};

struct DevicePrepOutput {
    bool need_mask = true;
};

struct LoraLinearOutput {
    BufferPtr output;
};

struct LoraLinearParams {

    LoraLinearParams(GemmParams&                gemm_params) :
                     gemm_params(gemm_params) {}

    GemmParams&                             gemm_params;
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
};

}  // namespace fastertransformer
