
#include "rtp_llm/models_py/bindings/common/kernels/embedding_kernels.h"
#include "rtp_llm/models_py/bindings/common/Torch_ext.h"
#include "rtp_llm/models_py/bindings/common/RtpEmbeddingLookup.h"
#include <cstdint>
#include <iostream>
#include <type_traits>
#include <vector>
using namespace std;
namespace th = torch;
using namespace rtp_llm;
namespace rtp_llm {

namespace {

void checkEmbeddingIoContract(const at::Tensor& output, const at::Tensor& input, const at::Tensor& weight) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(output);
    TORCH_CHECK(input.scalar_type() == at::ScalarType::Int, "input_ids must be int32");
    TORCH_CHECK(weight.device() == input.device(), "embedding weight must be on device ", input.device());
    TORCH_CHECK(output.device() == input.device(), "output must be on device ", input.device());
    TORCH_CHECK(output.scalar_type() == weight.scalar_type(), "output dtype must match embedding weight dtype");
}

const int* optionalInt32TensorPtr(const std::optional<at::Tensor>& tensor,
                                  const char*                      name,
                                  const int                        tokens,
                                  const at::Device&                device) {
    if (!tensor.has_value() || !tensor->defined() || tensor->numel() == 0) {
        return nullptr;
    }
    const auto& value = tensor.value();
    TORCH_CHECK(value.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(value.scalar_type() == at::ScalarType::Int, name, " must be int32");
    TORCH_CHECK(value.device() == device, name, " must be on device ", device, ", got ", value.device());
    TORCH_CHECK(value.dim() == 1, name, " must be a 1D tensor, got ", value.dim(), " dims");
    TORCH_CHECK(value.size(0) == tokens, name, " must have one id per token, got ", value.size(0), " vs ", tokens);
    return value.data_ptr<int>();
}

}  // namespace

void embedding(at::Tensor&               output,
               at::Tensor&               input,
               at::Tensor&               weight,
               std::optional<at::Tensor> position_ids,
               std::optional<at::Tensor> token_type_ids,
               std::optional<at::Tensor> text_tokens_mask) {
    checkEmbeddingIoContract(output, input, weight);
    auto device = input.device();
    CHECK_DIM(1, input);   // input: (tokens)
    CHECK_DIM(2, weight);  // weight: (hidden_size, hidden_size)
    const int tokens      = input.size(0);
    const int hidden_size = weight.size(1);
    CHECK_EQ(output.size(0), tokens);
    CHECK_EQ(output.size(1), hidden_size);

    // The generic op has no position or token-type tables, so these legacy
    // arguments do not participate in the kernel computation. Keep them in
    // the binding signature for compatibility, but do not impose a contract
    // on values that are intentionally ignored.
    (void)position_ids;
    (void)token_type_ids;
    StreamType stream   = GET_CURRENT_STREAM();
    auto*      mask_ptr = optionalInt32TensorPtr(text_tokens_mask, "text_tokens_mask", tokens, device);

    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(weight.scalar_type(), c_type, [&] {
        const int vecSize = sizeof(float4) / sizeof(c_type);
        if (hidden_size % vecSize == 0 && !mask_ptr) {  // this kernel does not support mask
            invokeEmbeddingLookupVec(static_cast<c_type*>(output.data_ptr()),
                                     static_cast<const c_type*>(weight.data_ptr()),
                                     1.0,
                                     static_cast<const c_type*>(nullptr),  // postition_table
                                     static_cast<const c_type*>(nullptr),  // token_type_table
                                     static_cast<const int*>(input.data_ptr()),
                                     static_cast<const int*>(nullptr),  // position_ids
                                     static_cast<const int*>(nullptr),  // token_types
                                     static_cast<const int*>(nullptr),  // mask
                                     tokens,
                                     hidden_size,
                                     stream);
        } else {
            invokeEmbeddingLookup(static_cast<c_type*>(output.data_ptr()),
                                  static_cast<const c_type*>(weight.data_ptr()),
                                  1.0,
                                  static_cast<const c_type*>(nullptr),  // postition_table
                                  static_cast<const c_type*>(nullptr),  // token_type_table
                                  static_cast<const int*>(input.data_ptr()),
                                  static_cast<const int*>(nullptr),   // position_ids
                                  static_cast<const int*>(nullptr),   // token_types
                                  static_cast<const int*>(mask_ptr),  // mask
                                  tokens,
                                  hidden_size,
                                  stream);
        }
        return true;
    });
}

void embeddingBert(at::Tensor&               output,
                   at::Tensor&               input,
                   at::Tensor&               weight,
                   at::Tensor                combo_position_ids,
                   at::Tensor                position_encoding,
                   at::Tensor                combo_tokens_type_ids,
                   at::Tensor                token_type_embedding,
                   float                     input_embedding_scalar,
                   std::optional<at::Tensor> text_tokens_mask) {
    checkEmbeddingIoContract(output, input, weight);
    CHECK_INPUT(combo_position_ids);
    CHECK_INPUT(position_encoding);
    CHECK_INPUT(combo_tokens_type_ids);
    CHECK_INPUT(token_type_embedding);
    auto device = input.device();
    CHECK_EQ(combo_position_ids.device(), device);
    CHECK_EQ(position_encoding.device(), device);
    CHECK_EQ(combo_tokens_type_ids.device(), device);
    CHECK_EQ(token_type_embedding.device(), device);
    TORCH_CHECK(combo_position_ids.scalar_type() == at::ScalarType::Int, "combo_position_ids must be int32");
    TORCH_CHECK(combo_tokens_type_ids.scalar_type() == at::ScalarType::Int, "combo_tokens_type_ids must be int32");
    TORCH_CHECK(position_encoding.scalar_type() == weight.scalar_type(),
                "position_encoding dtype must match embedding weight dtype");
    TORCH_CHECK(token_type_embedding.scalar_type() == weight.scalar_type(),
                "token_type_embedding dtype must match embedding weight dtype");
    CHECK_DIM(1, input);                  // input: (tokens)
    CHECK_DIM(2, weight);                 // weight: (vocab_size, hidden_size)
    CHECK_DIM(1, combo_position_ids);     // combo_position_ids: (tokens)
    CHECK_DIM(2, position_encoding);      // position_encoding: (max_position_embeddings, hidden_size), the
                                          // longest context bert can support is max_position_embeddings
    CHECK_DIM(1, combo_tokens_type_ids);  // combo_tokens_type_ids: (tokens)
    CHECK_DIM(2, token_type_embedding);   // token_type_embedding: (token_type_vocab_size, hidden_size)
    TORCH_CHECK(position_encoding.numel() > 0, "Bert position embedding table must not be empty");
    TORCH_CHECK(token_type_embedding.numel() > 0, "Bert token type embedding table must not be empty");
    const int tokens      = input.size(0);
    const int hidden_size = weight.size(1);
    // CUDA graph replay uses capacity-sized captured ID buffers and consumes
    // only the active token prefix. text_tokens_mask remains exact-length
    // because each element carries request semantics for one active token.
    TORCH_CHECK(combo_position_ids.size(0) >= tokens,
                "combo_position_ids must have at least one id per token, got ",
                combo_position_ids.size(0),
                " vs ",
                tokens);
    TORCH_CHECK(combo_tokens_type_ids.size(0) >= tokens,
                "combo_tokens_type_ids must have at least one id per token, got ",
                combo_tokens_type_ids.size(0),
                " vs ",
                tokens);
    CHECK_EQ(position_encoding.size(1), hidden_size);
    CHECK_EQ(token_type_embedding.size(1), hidden_size);
    CHECK_EQ(output.size(0), tokens);
    CHECK_EQ(output.size(1), hidden_size);
    const int* mask_ptr = optionalInt32TensorPtr(text_tokens_mask, "text_tokens_mask", tokens, device);
    StreamType stream   = GET_CURRENT_STREAM();

    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(weight.scalar_type(), c_type, [&] {
        invokeEmbeddingLookup(static_cast<c_type*>(output.data_ptr()),
                              static_cast<const c_type*>(weight.data_ptr()),
                              input_embedding_scalar,
                              static_cast<const c_type*>(position_encoding.data_ptr()),     // postition_table
                              static_cast<const c_type*>(token_type_embedding.data_ptr()),  // token_type_table
                              static_cast<const int*>(input.data_ptr()),
                              static_cast<const int*>(combo_position_ids.data_ptr()),     // position_ids
                              static_cast<const int*>(combo_tokens_type_ids.data_ptr()),  // token_types
                              mask_ptr,
                              tokens,
                              hidden_size,
                              stream);
        return true;
    });
}

}  // namespace rtp_llm
