#pragma once

#include <torch/all.h>

namespace rtp_llm {

// The caller admits the measured CUDA FP32 [1..32,248320] domain. The output aliases
// logits, preserving the public sampler's existing in-place probabilities.
void splitSoftmaxInplace(torch::Tensor logits);

}  // namespace rtp_llm
