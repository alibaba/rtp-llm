#pragma once

#include <initializer_list>
#include <torch/torch.h>

namespace rtp_llm {

// Embedding gathers host int32 metadata for model/attention planning. Keep it
// independent of model-specific environment variables. PyWrappedModel owns the
// device copies used by embedding lookup and GPU attention kernels.
inline void prepareEmbeddingInputMetadata(torch::Tensor& tokens,
                                          torch::Tensor& input_lengths,
                                          torch::Tensor& sequence_lengths,
                                          torch::Tensor& prefix_lengths) {
    for (auto* tensor : {&tokens, &input_lengths, &sequence_lengths, &prefix_lengths}) {
        if (tensor->defined()) {
            *tensor = tensor->to(torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU)).contiguous();
        }
    }
}

}  // namespace rtp_llm
