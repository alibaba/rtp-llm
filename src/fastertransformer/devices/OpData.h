#pragma once

#include "src/fastertransformer/devices/Buffers.h"
#include "src/fastertransformer/devices/Weights.h"
#include "src/fastertransformer/core/Tensor.h"

#include <optional>

namespace fastertransformer {

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
    const Tensor&  residual1;
    const Tensor&  residual2;
    const Tensor&  bias;
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

// C = A * B
struct GemmParams {
    const Tensor& A;  // expected shape: [m, k]
    const Tensor& B;  // expected shape: [k, n]
    Tensor&       C;  // expected shape: [m, n]

    const std::optional<const Tensor> A_scale;
    const std::optional<const Tensor> B_Scale;

    // const float alpha;
    // const float beta;

    TransposeOperation transA;
    TransposeOperation transB;

    // const int lda;
    // const int ldb;
    // const int ldc;

    // const int64_t strideA;
    // const int64_t strideB;
    // const int64_t strideC;

    Tensor& workspace;
    // TODO: maybe add activation-fused gemm interface.
};

struct AttentionCommonInputs {

    Tensor& kv_cache_blocks;
    Tensor& kv_cache_scales;

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

    const Tensor& lora_ids;
    const Tensor& lora_input_lengths;
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
    AttentionBuffers&            buffers;

    uint32_t batch_size;
    uint32_t max_seq_length;

    AttentionCommonInputs& common;
};

struct AttentionLayerParams {
    const Tensor& input;
    Tensor&       output;

    const AttentionLayerWeights& weights;
    AttentionBuffers&            buffers;

    const uint32_t generate_batch_size;
    const uint32_t max_generate_seq_length;
    const uint32_t context_batch_size;
    const uint32_t max_context_seq_length;

    AttentionCommonInputs& common;
};

struct FfnLayerParams {
    const Tensor& input;
    const Tensor& output;

    const FfnLayerWeights& weights;
    FfnBuffers&            buffers;

    const ActivationType activation_type;

    const Tensor& lora_ids;
    const Tensor& lora_input_lengths;
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
