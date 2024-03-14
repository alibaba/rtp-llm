#pragma once

#include "src/fastertransformer/devices/Weights.h"
#include "src/fastertransformer/devices/CommonDefines.h"
#include "src/fastertransformer/utils/activation_types.h"
#include "src/fastertransformer/utils/RopeTypes.h"

#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/utils/activation_types.h"
#include "src/fastertransformer/utils/layernorm_types.h"

#include <optional>
#include <functional>
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

    // Essential params

    const Buffer& A;
    const Buffer& B;
    Buffer&       D;

    GemmParams(const Buffer& A,
               const Buffer& B,
               Buffer& D):
               A(A),
               B(B),
               D(D) {}
    
    GemmParams(TransposeOperation transA,
               TransposeOperation transB,
               const Buffer& A,
               const Buffer& B,
               Buffer& D):
               transA(transA),
               transB(transB),
               A(A),
               B(B),
               D(D) {}

    // Optional params

    const std::optional<
        std::reference_wrapper<const Buffer>> C = std::nullopt;
    
    GemmParams(const Buffer& A,
               const Buffer& B,
               const Buffer& C,
               Buffer& D):
               A(A),
               B(B),
               C(C),
               D(D) {}

    // Attribute
    const TransposeOperation transA = TransposeOperation::NONE;
    const TransposeOperation transB = TransposeOperation::NONE;

    float alpha = 1.0f;
    float beta  = 0.0f;

    void Check() const;

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
    const Buffer& gate_weight;
    const Buffer& up_weight;
    const Buffer& down_weight;
    Buffer& output;
    ActivationType atype;

    FfnLayerParams(const Buffer& input,
                   const Buffer& gate_weight,
                   const Buffer& up_weight,
                   const Buffer& down_weight,
                   Buffer& output,
                   ActivationType atype) : 
                   input(input),
                   gate_weight(gate_weight),
                   up_weight(up_weight),
                   down_weight(down_weight),
                   output(output),
                   atype(atype) {}
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

// output = act(input) + bias
struct ActivationParams {
    using OBuffer = const std::optional<std::reference_wrapper<const Buffer>>;
    Buffer& output;
    OBuffer input;
    OBuffer bias;
    OBuffer gate;
    OBuffer gate_bias;

    ActivationType atype;

    ActivationParams(ActivationType atype,
                     Buffer& output,
                     OBuffer& bias = std::nullopt,
                     OBuffer& gate = std::nullopt,
                     OBuffer& gate_bias = std::nullopt) : 
                     atype(atype),
                     output(output),
                     bias(bias),
                     gate(gate),
                     gate_bias(gate_bias) {}
};


struct ContextAttentionParams {

    // shape[token_num, head_num + 2 * head_kv_num, head_size]
    Buffer& qkv_input;
    

    // shape[batch_size, head_num, seq_len, head_size]
    Buffer& q_output;
    // shape[batch_size, head_kv_num, seq_len, head_size]
    Buffer& k_output;
    // shape[batch_size, head_kv_num, seq_len, head_size]
    Buffer& v_output;
    // shape[(head_num + 2 * head_kv_num) * head_size]
    const Buffer& bias;
    // shape[token_num]
    const Buffer& position_ids;
    // shape[token_num]
    const Buffer& padding_offset;
    // shape[batch_size]
    const Buffer& cu_seqlens;
    // shape[batch_size, seq_len, seq_len]
    const Buffer& attention_mask;

    const RopeConfig rope_config;

    // tmp for test
    Buffer& qk_output;
    // Buffer& qk_softmax_output;
    Buffer& softmax_qk_output;
    Buffer& qkv_output;
    Buffer& qkv_transpose_output;

    ContextAttentionParams(Buffer& qkv_input,
                           Buffer& q_output,
                           Buffer& k_output,
                           Buffer& v_output,
                           const Buffer& bias,
                           const Buffer& position_ids,
                           const Buffer& padding_offset,
                           const Buffer& cu_seqlens,
                           const Buffer& attention_mask,
                           const RopeConfig& rope_config,
                           Buffer& qk_output,
                           Buffer& softmax_qk_output,
                           Buffer& qkv_output,
                           Buffer& qkv_transpose_output) :
                           qkv_input(qkv_input),
                           q_output(q_output),
                           k_output(k_output),
                           v_output(v_output),
                           bias(bias),
                           position_ids(position_ids),
                           padding_offset(padding_offset),
                           cu_seqlens(cu_seqlens),
                           attention_mask(attention_mask),
                           rope_config(rope_config),
                           qk_output(qk_output),
                           softmax_qk_output(softmax_qk_output),
                           qkv_output(qkv_output),
                           qkv_transpose_output(qkv_transpose_output) {} 

    void check() const;
};

struct SoftmaxParams{
    
    const Buffer& input;
    Buffer& output;
    const Buffer& mask;
    float scale = 1.0f;

    SoftmaxParams(const Buffer& input,
                  Buffer& output,
                  const Buffer& mask) : 
                  input(input),
                  output(output),
                  mask(mask) {}
    
    SoftmaxParams(const Buffer& input,
                  Buffer& output,
                  const Buffer& mask,
                  float scale) : 
                  input(input),
                  output(output),
                  mask(mask),
                  scale(scale) {}

};

}  // namespace fastertransformer
