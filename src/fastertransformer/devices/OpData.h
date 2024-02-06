#pragma once

#include "src/fastertransformer/devices/Weights.h"
#include "src/fastertransformer/core/Tensor.h"

#include <optional>

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

enum class NormType {
    Layernorm,
    RmsNorm,
    AlphaNorm,
    InvalidType
};

enum class ActivationType {
    Gelu,
    GeluNoneApproximate,
    Relu,
    Silu,
    GeGLU,
    GeGluNoneApproximate,
    ReGLU,
    SiGLU,
    Identity,
    InvalidType
};

struct LayernormParams {
    const NormType norm_type;
    const Tensor&  input;
    const std::optional<Tensor>  residual1;
    const std::optional<Tensor>  residual2;
    const std::optional<Tensor>  bias;
    const Tensor&  gamma;
    const Tensor&  beta;
    const float    eps;

    const Tensor& scale_inter;
    const Tensor& scale_out;
    const Tensor& scale;
    const Tensor& dynamic_scale;

    Tensor& norm_output;
};

// corresponds to cublasOperation_t
enum class TransposeOperation {
    NONE                = 0,
    TRANSPOSE           = 1,
    CONJUGATE_TRANSPOSE = 2,
};

// D = alpha * op(A) * op(B) + beta * C
// shapes of A, B, C, D have two options: [m, k], [k, n], [m, n], [m, n]
// or [bs, m, k], [bs, k, n], [bs, m, n], [bs, m, n] where bs is batch_size
// NOTE: caller needs to preallocate C
struct GemmParams {
    GemmParams(const Tensor& A, const Tensor& B, Tensor& C)
    : A(A), B(B), C(C), D(C) {}
    GemmParams(const Tensor& A, const Tensor& B, const Tensor& C, Tensor& D)
    : A(A), B(B), C(C), D(D) {}

    const Tensor& A;
    const Tensor& B;
    const Tensor& C;
    Tensor&       D;

    const std::optional<const Tensor> A_scale = std::nullopt;
    const std::optional<const Tensor> B_Scale = std::nullopt;
    const std::optional<const Tensor> C_scale = std::nullopt;

    const TransposeOperation transA = TransposeOperation::NONE;
    const TransposeOperation transB = TransposeOperation::NONE;

    const float alpha = 1.0f;
    const float beta  = 0.0f;
    const std::optional<DataType> computation_type = std::nullopt;
};

// D = alpha * op(A) * op(B) + beta * C
// shapes of each A, B, C, D needs to be [m, k], [k, n], [m, n], [m, n]
struct GroupedGemmParams {
    GroupedGemmParams(
        const std::vector<Tensor>& A,
        const std::vector<Tensor>& B,
        std::vector<Tensor>& C
    ) : A(A), B(B), C(C), D(C) {}
    GroupedGemmParams(
        const std::vector<Tensor>& A,
        const std::vector<Tensor>& B,
        const std::vector<Tensor>& C,
        std::vector<Tensor>&       D
    ) : A(A), B(B), C(C), D(D) {}

    const std::vector<Tensor>& A;
    const std::vector<Tensor>& B;
    const std::vector<Tensor>& C;
    std::vector<Tensor>&       D;
};

struct AttentionCommonInputs {
    Tensor& kv_cache_blocks;
    const std::optional<const Tensor> kv_cache_scales;

    const Tensor& input_lengths;
    const Tensor& sequence_lengths;
    const Tensor& padding_offset;
    const Tensor& cu_seqlens;  // cumulated sequence lengths

    const std::optional<const Tensor> position_ids;
    const std::optional<const Tensor> attention_mask;
    const std::optional<const Tensor> linear_bias_slopes;
    const std::optional<const Tensor> prefix_prompt_lengths;
    const std::optional<bool>         count_prefix_length;
    const std::optional<uint32_t>     max_prefix_length;

    const std::optional<Tensor> lora_ids;
    const std::optional<Tensor> lora_input_lengths;
};

// TODO(wangyin): figure out these styles and doc them.
enum class PositionEmbeddingStyle {
    BaseRotaryEmbedding          = 0,
    LinearScalar  = 1,
    NTKScalar     = 2,
    DynamicNTKS   = 3,
    GLM           = 4,
};

struct AttentionConfigs {
    PositionEmbeddingStyle position_embedding_style;
    int64_t rotary_embedding_dim      = 0;
    int64_t rotary_embedding_base     = 10000;
    double  dynamic_embedding_scalar  = 0.0;
    int64_t dynamic_embedding_max_pos = 0;
    int64_t position_embeddings_scale = 1;
    int64_t base_scale                = 1;

    bool    use_logn_attn = false;
    int64_t logn_seq_len  = 2048;
};

struct AttentionModuleParams {
    const Tensor& input;
    Tensor&       output;

    const AttentionConfigs&      configs;
    const AttentionLayerWeights& weights;

    uint32_t batch_size;
    uint32_t max_seq_length;

    AttentionCommonInputs& common;
};

struct AttentionLayerParams {
    const Tensor& input;
    Tensor&       output;

    const AttentionLayerWeights& weights;

    const uint32_t generate_batch_size;
    const uint32_t max_generate_seq_length;
    const uint32_t context_batch_size;
    const uint32_t max_context_seq_length;

    AttentionCommonInputs& common;
};

struct FfnLayerParams {
    const Tensor& input;
    Tensor& output;

    const FfnLayerWeights&       weights;

    const ActivationType activation_type;

    const std::optional<Tensor> lora_ids;
    const std::optional<Tensor> lora_input_lengths;
};

struct SamplerParams {
    const Tensor& logits;
    const Tensor& step;              // shape: [1]
    const Tensor& max_input_length;  // shape: [1]
    const Tensor& input_lengths;     // shape: [batch_size]
    const Tensor& ite;               // shape: [1]
    const Tensor& eos_id;

    Tensor& output_ids;
    Tensor& sequence_length;
    Tensor& finished;
    Tensor& cum_log_probs;
    Tensor& output_log_probs;
};

struct TopPSamplerParams {
    const SamplerParams& sampler_params;
    const Tensor&        top_p;
    const Tensor&        temperature;
    const Tensor&        random_seed;
    const Tensor&        repetition_penalty;
};

struct TopKSamplerParams {
    const SamplerParams& sampler_params;
    const Tensor&        top_k;
    const Tensor&        temperature;
    const Tensor&        random_seed;
    const Tensor&        repetition_penalty;
};

struct BroadcastParams {
    std::vector<Tensor>& tensors;
    const int64_t        root;
};

struct AllReduceParams {
    std::vector<Tensor>& tensors;
};

}  // namespace fastertransformer
