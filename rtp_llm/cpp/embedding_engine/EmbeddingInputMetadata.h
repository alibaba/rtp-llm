#pragma once

#include <cstdlib>
#include <string>
#include <torch/torch.h>

namespace rtp_llm {

#if USING_CUDA
namespace embedding_input_detail {
inline torch::Tensor toCudaInt32ModelInput(const torch::Tensor& tensor) {
    if (!tensor.defined()) {
        return tensor;
    }
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    if (tensor.is_cuda() && tensor.scalar_type() == torch::kInt32) {
        return tensor;
    }
    if (tensor.numel() == 0) {
        return torch::empty(tensor.sizes(), options);
    }
    return tensor.to(options, /*non_blocking=*/true);
}
}  // namespace embedding_input_detail

// Final model-facing conversion used by EmbeddingExecutor::gatherModelInput.
// UQI planning scans host tokens and lengths, including when eager two-pass is
// disabled. Preserve those tensors regardless of VISION_BERT_UQI_TWO_PASS.
inline void prepareEmbeddingInputMetadata(torch::Tensor& tokens,
                                          torch::Tensor& input_lengths,
                                          torch::Tensor& sequence_lengths,
                                          torch::Tensor& prefix_lengths) {
    static const bool use_uqi_host_metadata = []() {
        const char* mask = std::getenv("USE_VISION_BERT_UQI_BLOCK_MASK");
        return mask && std::string(mask) == "1";
    }();
    if (use_uqi_host_metadata) {
        return;
    }
    tokens           = embedding_input_detail::toCudaInt32ModelInput(tokens);
    input_lengths    = embedding_input_detail::toCudaInt32ModelInput(input_lengths);
    sequence_lengths = embedding_input_detail::toCudaInt32ModelInput(sequence_lengths);
    prefix_lengths   = embedding_input_detail::toCudaInt32ModelInput(prefix_lengths);
}
#endif

}  // namespace rtp_llm
