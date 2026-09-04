#include "rtp_llm/models_py/bindings/common/Torch_ext.h"
#include "rtp_llm/models_py/bindings/common/CudaGraphPrefillCopy.h"
#include "rtp_llm/models_py/bindings/core/Dispatch.h"
#include "rtp_llm/models_py/bindings/core/torch_utils/TypeConvert.h"
#include "rtp_llm/models_py/bindings/common/kernels/cuda_graph_copy_kernel.h"
#include "rtp_llm/models_py/bindings/common/kernels/input_embedding_overlay_kernel.h"
#include <cstdint>
#include <iostream>
#include <type_traits>
#include <vector>
using namespace std;
using namespace rtp_llm;
namespace torch_ext {

void cuda_graph_copy_small2large(at::Tensor& input_tensor,
                                 at::Tensor& output_tensor,
                                 at::Tensor& batch_size,
                                 int64_t     max_batch_size,
                                 int64_t     max_seq_len,
                                 at::Tensor& input_lengths,
                                 int64_t     hidden_size,
                                 at::Tensor& cu_seq_len) {
    CHECK_INPUT(input_tensor);
    CHECK_INPUT(output_tensor);
    CHECK_DIM(2, input_tensor);   // input: (total_elements, hidden_size)
    CHECK_DIM(2, output_tensor);  // output: (max_batch_size * max_seq_len, hidden_size)
    CHECK_DIM(1, batch_size);     // batch_size: (1), CPU pinned memory
    CHECK_DIM(1, input_lengths);  // input_lengths: (batch_size)
    CHECK_DIM(1, cu_seq_len);     // cu_seq_len: (batch_size + 1), CPU pinned memory
    auto       input_ptr         = input_tensor.data_ptr();
    auto       output_ptr        = output_tensor.data_ptr();
    auto       batch_size_ptr    = batch_size.data_ptr<int>();
    auto       input_lengths_ptr = input_lengths.data_ptr<int>();
    auto       cu_seq_len_ptr    = cu_seq_len.data_ptr<int>();
    StreamType stream            = GET_CURRENT_STREAM();
    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_tensor.scalar_type(), c_type, [&] {
        rtp_llm::invokeCudaGraphCopySmall2Large<c_type>(static_cast<c_type*>(input_ptr),
                                                        static_cast<c_type*>(output_ptr),
                                                        batch_size_ptr,
                                                        static_cast<int>(max_batch_size),
                                                        static_cast<int>(max_seq_len),
                                                        input_lengths_ptr,
                                                        static_cast<int>(hidden_size),
                                                        cu_seq_len_ptr,
                                                        stream);
        return true;
    });
}

void cuda_graph_copy_large2small(at::Tensor& input_tensor,
                                 at::Tensor& output_tensor,
                                 at::Tensor& batch_size,
                                 int64_t     max_batch_size,
                                 int64_t     max_seq_len,
                                 at::Tensor& input_lengths,
                                 int64_t     hidden_size,
                                 at::Tensor& cu_seq_len) {
    CHECK_INPUT(input_tensor);
    CHECK_INPUT(output_tensor);
    CHECK_DIM(2, input_tensor);   // input: (max_batch_size * max_seq_len, hidden_size)
    CHECK_DIM(2, output_tensor);  // output: (total_elements, hidden_size)
    CHECK_DIM(1, batch_size);     // batch_size: (1)
    CHECK_DIM(1, input_lengths);  // input_lengths: (batch_size)
    CHECK_DIM(1, cu_seq_len);     // cu_seq_len: (batch_size + 1)
    auto       input_ptr         = input_tensor.data_ptr();
    auto       output_ptr        = output_tensor.data_ptr();
    auto       batch_size_ptr    = batch_size.data_ptr<int>();
    auto       input_lengths_ptr = input_lengths.data_ptr<int>();
    auto       cu_seq_len_ptr    = cu_seq_len.data_ptr<int>();
    StreamType stream            = GET_CURRENT_STREAM();
    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_tensor.scalar_type(), c_type, [&] {
        rtp_llm::invokeCudaGraphCopyLarge2Small<c_type>(static_cast<c_type*>(input_ptr),
                                                        static_cast<c_type*>(output_ptr),
                                                        batch_size_ptr,
                                                        static_cast<int>(max_batch_size),
                                                        static_cast<int>(max_seq_len),
                                                        input_lengths_ptr,
                                                        static_cast<int>(hidden_size),
                                                        cu_seq_len_ptr,
                                                        stream);
        return true;
    });
}

void input_embedding_overlay(at::Tensor& inputs_embeds,
                             at::Tensor& input_embedding_overrides,
                             at::Tensor& input_embedding_metadata) {
    CHECK_INPUT(inputs_embeds);
    CHECK_INPUT(input_embedding_overrides);
    CHECK_INPUT(input_embedding_metadata);
    CHECK_DIM(2, inputs_embeds);
    CHECK_DIM(2, input_embedding_overrides);
    CHECK_DIM(1, input_embedding_metadata);
    TORCH_CHECK(inputs_embeds.sizes() == input_embedding_overrides.sizes(),
                "input_embedding_overrides must match inputs_embeds shape");
    TORCH_CHECK(inputs_embeds.scalar_type() == input_embedding_overrides.scalar_type(),
                "input_embedding_overrides must match inputs_embeds dtype");
    TORCH_CHECK(input_embedding_metadata.scalar_type() == at::ScalarType::Int,
                "input_embedding_metadata must have int32 dtype");
    TORCH_CHECK(input_embedding_metadata.size(0) == inputs_embeds.size(0),
                "input_embedding_metadata must have one entry per token");
    TORCH_CHECK(inputs_embeds.get_device() == input_embedding_overrides.get_device()
                    && inputs_embeds.get_device() == input_embedding_metadata.get_device(),
                "input embedding overlay tensors must be on the same device");

    auto       stream       = GET_CURRENT_STREAM();
    const auto token_count  = static_cast<int>(inputs_embeds.size(0));
    const auto hidden_size  = static_cast<int>(inputs_embeds.size(1));
    auto*      metadata_ptr = input_embedding_metadata.data_ptr<int32_t>();
    switch (inputs_embeds.scalar_type()) {
        case at::ScalarType::Float:
            rtp_llm::invokeInputEmbeddingOverlay<float>(inputs_embeds.data_ptr<float>(),
                                                        input_embedding_overrides.data_ptr<float>(),
                                                        metadata_ptr,
                                                        token_count,
                                                        hidden_size,
                                                        stream);
            break;
        case at::ScalarType::Half:
            rtp_llm::invokeInputEmbeddingOverlay<half>(static_cast<half*>(inputs_embeds.data_ptr()),
                                                       static_cast<const half*>(input_embedding_overrides.data_ptr()),
                                                       metadata_ptr,
                                                       token_count,
                                                       hidden_size,
                                                       stream);
            break;
        case at::ScalarType::BFloat16:
#ifdef ENABLE_BF16
            rtp_llm::invokeInputEmbeddingOverlay<bf16_type>(
                static_cast<bf16_type*>(inputs_embeds.data_ptr()),
                static_cast<const bf16_type*>(input_embedding_overrides.data_ptr()),
                metadata_ptr,
                token_count,
                hidden_size,
                stream);
            break;
#else
            TORCH_CHECK(false, "input_embedding_overlay was built without BF16 support");
#endif
        default:
            TORCH_CHECK(false,
                        "input_embedding_overlay only supports float32, float16, and bfloat16, got ",
                        inputs_embeds.scalar_type());
    }
}

}  // namespace torch_ext
