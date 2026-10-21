#pragma once

#include <initializer_list>
#include <torch/torch.h>

namespace rtp_llm {

// Normalize gathered host metadata before model-facing device conversion.
// Graph planning obtains its host views separately in PyWrappedModel.
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
