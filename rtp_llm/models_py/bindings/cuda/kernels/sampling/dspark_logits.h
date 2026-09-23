#pragma once

#include <torch/types.h>

namespace rtp_llm {

// Return an undefined tensor for layouts/dtypes outside the fused CUDA contract.
torch::Tensor tryPrepareDSparkLogits(const torch::Tensor& base_logits,
                                     const torch::Tensor& markov_bias,
                                     const torch::Tensor& temperature);

}  // namespace rtp_llm
