#pragma once

#include "src/fastertransformer/devices/Weights.h"
#include "src/fastertransformer/devices/CommonDefines.h"
#include "src/fastertransformer/utils/activation_types.h"
#include "src/fastertransformer/utils/RopeTypes.h"

#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/utils/activation_types.h"
#include "src/fastertransformer/utils/layernorm_types.h"
#include "src/fastertransformer/utils/EnumUtils.h"

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
        return ss.str().c_str();
    }

    const OpStatus& status() const { return status_; }
private:
    OpStatus status_;
};

using OptionalConstBufferRef    = std::optional<std::reference_wrapper<const Buffer>>;
using OptionalBufferRef         = std::optional<std::reference_wrapper<Buffer>>;

using OptionalConstLoraMapRef    = std::optional<std::reference_wrapper<const LoraWeightsMap>>;

template <typename T>
inline std::optional<std::reference_wrapper<T>> mayGetRef(const std::unique_ptr<T>& ptr) {
    return ptr ? std::optional<std::reference_wrapper<T>>(*ptr) : std::nullopt;
}

using CloneOutput = BufferPtr;

struct CloneParams {
    CloneParams(const Buffer& input, const AllocationType alloc_type = AllocationType::DEVICE)
    : input(input), alloc_type(alloc_type) {}

    const Buffer& input;
    const AllocationType alloc_type;
};

struct CopyParams {
    CopyParams(const Buffer& dst, const Buffer& src)
    : dst(dst), src(src)
    {}

    // copy from part of src to full dst, offsets / size are at dim 0.
    CopyParams(const Buffer& dst, const Buffer& src,
               const size_t dst_offset, const size_t src_offset,
               const size_t copy_length)
    : dst(dst), src(src)
    , dst_offset(dst_offset), src_offset(src_offset), copy_length(copy_length)
    {}

    const Buffer& dst;
    const Buffer& src;

    size_t dst_offset = 0;
    size_t src_offset = 0;
    size_t copy_length = 0;
};

using TransposeOutput = BufferPtr;

struct TransposeParams {
    const Buffer& input;
};

using LayernormOutput = void;

struct LayernormParams {
    LayernormParams(
        const Buffer& input,
        const Buffer& norm_output,
        const OptionalBufferRef add_bias_output,

        const NormType norm_type,
        const std::optional<std::reference_wrapper<const LayerNormWeights>> weights = std::nullopt,
        const std::optional<double> eps = std::nullopt,
        const OptionalConstBufferRef residual1 = std::nullopt,
        const OptionalConstBufferRef residual2 = std::nullopt,
        const OptionalConstBufferRef bias = std::nullopt,
        const std::optional<double> alpha = std::nullopt
    ) : input(input),
        norm_output(norm_output),
        add_bias_output(add_bias_output),
        norm_type(norm_type),
        weights(weights),
        eps(eps.value_or(1e-6)),
        residual1(residual1),
        residual2(residual2),
        bias(bias),
        alpha(alpha) {}

    const Buffer&  input;

    const Buffer& norm_output;
    const OptionalBufferRef add_bias_output;

    const NormType norm_type;
    const std::optional<std::reference_wrapper<const LayerNormWeights>> weights;
    const double eps;
    const OptionalConstBufferRef  residual1;
    const OptionalConstBufferRef  residual2;
    const OptionalConstBufferRef  bias;
    const std::optional<double>   alpha;
};


// D = alpha * op(A) * op(B) + beta * C
// shapes of A, B, C, D have two options: [m, k], [k, n], [m, n], [m, n]
// or [bs, m, k], [bs, k, n], [bs, m, n], [bs, m, n] where bs is batch_size
// NOTE: caller needs to preallocate C
// TODO(dongjin): Make C a BufferPtr so that it can achieve in-place update
struct GemmParams {

    GemmParams(const Buffer& A,
               const Buffer& B,
               OptionalBufferRef C = std::nullopt,
               const DataType compute_type = DataType::TYPE_INVALID,
               TransposeOperation transA = TransposeOperation::NONE,
               TransposeOperation transB = TransposeOperation::NONE):
               A(A),
               B(B),
               C(C),
               compute_type(compute_type),
               transA(transA),
               transB(transB) {}


    const Buffer& A;
    const Buffer& B;
    OptionalBufferRef C;
    const DataType compute_type = DataType::TYPE_INVALID; // If passed invalid type, op should infer type

    const TransposeOperation transA = TransposeOperation::NONE;
    const TransposeOperation transB = TransposeOperation::NONE;

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    void check() const;
};

struct GroupedGemmOutput {
    BufferPtr D;
};

// D = alpha * op(A) * op(B) + beta * C
// shapes of each A, B, C, D needs to be [m, k], [k, n], [m, n], [m, n]
struct GroupedGemmParams {
    using OutputType = GroupedGemmOutput;

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

    OptionalConstBufferRef position_ids;
    OptionalConstBufferRef position_table;
};

struct AttentionCommonInputs {
    // see detailed comments at GptModelInputs
    const Buffer& input_lengths;      // int32_t, [context_batch_size + decoder_batch_size]
    const Buffer& sequence_lengths;   // int32_t, [decoder_batch_size]

    // [batch_size, block_length], int64 block pointers
    OptionalConstBufferRef kv_cache_blocks;

    ConstBufferPtr cu_seqlens;
    ConstBufferPtr padding_offset;

    OptionalConstBufferRef position_ids;
    OptionalConstBufferRef attention_mask;
    OptionalConstBufferRef linear_bias_slopes;
    OptionalConstBufferRef prefix_prompt_lengths;

    OptionalConstBufferRef lora_ids;
    OptionalConstBufferRef lora_input_lengths;

    AttentionCommonInputs() = default;

    AttentionCommonInputs(const Buffer& input_lengths,
                          const Buffer& sequence_lengths) :
                          input_lengths(input_lengths),
                          sequence_lengths(sequence_lengths) {}

    // self attention
    // AttentionCommonInputs (const Buffer& kv_cache_blocks,
    //                        const Buffer& sequence_lengths,
    //                        const Buffer& input_lengths) :
    //                        kv_cache_blocks(kv_cache_blocks),
    //                        sequence_lengths(sequence_lengths),
    //                        input_lengths(input_lengths) {}
};

struct AttentionConfigs {
    size_t      token_num;
    size_t      batch_size;
    size_t      head_num;
    size_t      kv_head_num;
    size_t      seq_len;
    size_t      size_per_head;

    // rotary embending config
    RopeConfig rope_config;

    //kv cache block
    size_t max_blocks_per_seq;
    size_t tokens_per_block;

    // step
    size_t step;

    // beam search
    size_t beam_width = 1;

    // prefix params
    bool        count_prefix_length = false;
    size_t      max_prefix_length   = 0;

    // log attention
    bool    use_logn_attn = false;
    int64_t logn_seq_len  = 2048;
};

using AttentionModuleOutput = void;

struct AttentionModuleParams {
    // qkv shape[h_token_num, (head_num + 2 * kv_head_num) * size_per_head]
    const Buffer&                   input;
    // shape[token_num, size_per_head]
    Buffer&                         output;
    AttentionCommonInputs&          common;
    const AttentionLayerWeights&    weights;
    const AttentionConfigs&         configs;
};

struct AttentionLayerOutput {
    BufferPtr hidden_states;
};

struct AttentionLayerParams {
    const Buffer&                   input;
    const AttentionConfigs&         configs;
    const AttentionLayerWeights&    weights;
    AttentionCommonInputs&          common;
};

struct FfnLayerOutput {
    BufferPtr hidden_states;
};

struct FfnLayerParams {
    FfnLayerParams(const Buffer& input,
                   const FfnLayerWeights& weights,
                   const ActivationType atype) :
                   input(input),
                   weights(weights),
                   activation_type(atype)
                   {}

    const Buffer& input;

    const FfnLayerWeights&       weights;
    const ActivationType         activation_type;

    const OptionalConstBufferRef lora_ids;
    const OptionalConstBufferRef lora_input_lengths;
};

struct GreedyParams {
    const Buffer& logits;                    // [batch_size, vocab_size_padded]
    const Buffer& input_lengths;             // [batch_size]
    Buffer& token_ids;                       // [max_input_length + 1, batch_size]

    const Buffer& top_k;
    const Buffer& top_p;
    const Buffer& temperature;
    OptionalBufferRef random_seed;
    OptionalBufferRef repetition_penalty;
    OptionalBufferRef length_penalty;

    OptionalBufferRef cum_log_probs;
    OptionalBufferRef output_log_probs;
};

struct BeamSearchParams {
    const Buffer& logits;
    const Buffer& sequence_lengths;  // shape: [batch_size]
    const size_t step;               // typically largest sequence length in the batch

    const size_t num_beams;
    const size_t batch_size;

    Buffer& token_ids;
    Buffer& kv_cache_blocks;
    OptionalBufferRef cum_log_probs;
    OptionalBufferRef output_log_probs;
};

struct BeamSearchOutput {
    BufferPtr token_ids;
    BufferPtr cum_log_probs;
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
    ActivationType atype;
    Buffer& states;
    const OptionalConstBufferRef bias;
    const OptionalConstBufferRef gate;
    const OptionalConstBufferRef gate_bias;

    ActivationParams(ActivationType atype,
                     Buffer& states,
                     OptionalConstBufferRef bias = std::nullopt,
                     OptionalConstBufferRef gate = std::nullopt,
                     OptionalConstBufferRef gate_bias = std::nullopt) :
                     atype(atype),
                     states(states),
                     bias(bias),
                     gate(gate),
                     gate_bias(gate_bias) {}
};

struct SoftmaxParams{

    SoftmaxParams(BufferPtr input,
                  const Buffer& mask,
                  float scale = 1.0f,
                  const DataType output_t = DataType::TYPE_INVALID) :
                  input(std::move(input)),
                  mask(mask),
                  scale(scale),
                  output_t(output_t) {}

    // inplace
    BufferPtr input = nullptr;
    const Buffer& mask;
    float scale;
    const DataType output_t;
};

struct LoraLinearOutput {
    BufferPtr output;
};

struct LoraLinearParams {

    LoraLinearParams(const Buffer&              input,
                     OptionalConstBufferRef     lora_ids,
                     const DenseWeights&        weight,
                     OptionalConstLoraMapRef    lora_map) :
                     input(input),
                     lora_ids(lora_ids),
                     weight(weight),
                     lora_map(lora_map) {}

    const Buffer&                           input;
    OptionalConstBufferRef                  lora_ids;
    const DenseWeights&                     weight;
    OptionalConstLoraMapRef                 lora_map;
};

}  // namespace fastertransformer
