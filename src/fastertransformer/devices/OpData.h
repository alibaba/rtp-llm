#pragma once

#include "src/fastertransformer/devices/Weights.h"
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/utils/activation_types.h"
#include "src/fastertransformer/utils/layernorm_types.h"

#include <optional>
#include <sstream>
#include <memory>

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
    : status_(status) {}

    const char* what() const noexcept override {
        std::stringstream ss;
        ss << "OpException[" << (int32_t)status_.error_type << "]: " << status_.error_message;
        return status_.error_message.c_str();
    }

    const OpStatus& status() const { return status_; }
private:
    OpStatus status_;
};

struct CopyParams {
    const Buffer& src;
    Buffer&       dst;
};

using OptionalConstBufferRef = std::optional<std::reference_wrapper<const Buffer>>;

// The Layernorm Op also works as an AddBias Op
// if gamma and beta are not provided, output = input * alpha + residual1 + bias if alpha is provided;
// else output = input + residual1 + residual2 + bias
struct LayernormParams {

    // layernorm
    LayernormParams(const NormType norm_type, const Buffer& input,
                    const Buffer& gamma, const Buffer& beta, const float eps, Buffer& output):
    norm_type(norm_type), input(input), gamma(gamma), beta(beta), eps(eps), norm_output(output) {}

    // used for add bias
    LayernormParams(const Buffer& input,
                    const OptionalConstBufferRef& residual1,
                    const OptionalConstBufferRef& bias,
                    const std::optional<float> alpha, Buffer& output):
    norm_type(NormType::add_bias), input(input), residual1(residual1),
    bias(bias), alpha(alpha), norm_output(output) {}

    const NormType norm_type = NormType::layernorm;
    const Buffer&  input;
    const std::optional<std::reference_wrapper<const Buffer>>  residual1;
    const std::optional<std::reference_wrapper<const Buffer>>  residual2;
    const std::optional<std::reference_wrapper<const Buffer>>  bias;
    const std::optional<float> alpha;

    const std::optional<std::reference_wrapper<const Buffer>>  gamma;
    const std::optional<std::reference_wrapper<const Buffer>>  beta;
    const float eps = 1e-6;

    const std::optional<std::reference_wrapper<const Buffer>> scale_inter;
    const std::optional<std::reference_wrapper<const Buffer>> scale_out;
    const std::optional<std::reference_wrapper<const Buffer>> scale;
    const std::optional<std::reference_wrapper<const Buffer>> dynamic_scale;

    Buffer& norm_output;
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
    GemmParams(const Buffer& A, const Buffer& B, Buffer& C)
    : A(A), B(B), C(C), D(C) {}
    GemmParams(const Buffer& A, const Buffer& B, const Buffer& C, Buffer& D)
    : A(A), B(B), C(C), D(D) {}

    const Buffer& A;
    const Buffer& B;
    const Buffer& C;
    Buffer&       D;

    const std::optional<const Buffer> A_scale = std::nullopt;
    const std::optional<const Buffer> B_Scale = std::nullopt;
    const std::optional<const Buffer> C_scale = std::nullopt;

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
        const std::vector<Buffer>& A,
        const std::vector<Buffer>& B,
        std::vector<Buffer>& C
    ) : A(A), B(B), C(C), D(C) {}
    GroupedGemmParams(
        const std::vector<Buffer>& A,
        const std::vector<Buffer>& B,
        const std::vector<Buffer>& C,
        std::vector<Buffer>&       D
    ) : A(A), B(B), C(C), D(D) {}

    const std::vector<Buffer>& A;
    const std::vector<Buffer>& B;
    const std::vector<Buffer>& C;
    std::vector<Buffer>&       D;
};

struct EmbeddingLookupParams {
    const Buffer& combo_tokens;
    const Buffer& embedding_table;

    const std::optional<std::reference_wrapper<const Buffer>> position_ids;
    const std::optional<std::reference_wrapper<const Buffer>> position_table;

    Buffer& embeddings;
};

struct AttentionCommonInputs {
    const Buffer& kv_cache_blocks; // [batch_size, block_length], int64 block pointers
    const std::optional<std::reference_wrapper<const Buffer>> kv_cache_scales;

    const Buffer& input_lengths;
    const Buffer& sequence_lengths;

    const std::optional<std::reference_wrapper<const Buffer>> padding_offset;
    const std::optional<std::reference_wrapper<const Buffer>> position_ids;
    const std::optional<std::reference_wrapper<const Buffer>> attention_mask;
    const std::optional<std::reference_wrapper<const Buffer>> linear_bias_slopes;
    const std::optional<std::reference_wrapper<const Buffer>> prefix_prompt_lengths;
    const std::optional<bool>         count_prefix_length;
    const std::optional<uint32_t>     max_prefix_length;

    const std::optional<std::reference_wrapper<const Buffer>> lora_ids;
    const std::optional<std::reference_wrapper<const Buffer>> lora_input_lengths;
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

// Attention Module contains
struct AttentionModuleParams {
    const Buffer& input;
    Buffer&       output;

    const AttentionConfigs&      configs;
    const AttentionLayerWeights& weights;

    uint32_t batch_size;
    uint32_t max_seq_length;

    AttentionCommonInputs& common;
};

struct AttentionLayerParams {
    const Buffer& input;
    Buffer&       output;

    const AttentionConfigs&      configs;
    const AttentionLayerWeights& weights;
    AttentionCommonInputs& common;
};

struct FfnLayerParams {
    const Buffer& input;
    Buffer& output;

    const FfnLayerWeights&       weights;

    const ActivationType activation_type;

    const std::optional<std::reference_wrapper<const Buffer>> lora_ids;
    const std::optional<std::reference_wrapper<const Buffer>> lora_input_lengths;
};

struct SamplerParams {
    const Buffer& logits;
    const Buffer& step;              // shape: [1]
    const Buffer& max_input_length;  // shape: [1]
    const Buffer& input_lengths;     // shape: [batch_size]
    const Buffer& ite;               // shape: [1]
    const Buffer& eos_id;

    Buffer& output_ids;
    Buffer& sequence_length;
    Buffer& finished;
    Buffer& cum_log_probs;
    Buffer& output_log_probs;
};

struct TopPSamplerParams {
    const SamplerParams& sampler_params;
    const Buffer&        top_p;
    const Buffer&        temperature;
    const Buffer&        random_seed;
    const Buffer&        repetition_penalty;
};

struct TopKSamplerParams {
    const SamplerParams& sampler_params;
    const Buffer&        top_k;
    const Buffer&        temperature;
    const Buffer&        random_seed;
    const Buffer&        repetition_penalty;
};

struct BroadcastParams {
    std::vector<Buffer>& buffers;
    const int64_t        root;
};

struct AllReduceParams {
    std::vector<Buffer>& buffers;
};

}  // namespace fastertransformer
