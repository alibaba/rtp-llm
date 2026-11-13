#pragma once

#include <torch/all.h>

namespace rtp_llm {
torch::Tensor samplerSoftmaxInplace(const torch::Tensor& logits, bool batch_invariant = false);
}  // namespace rtp_llm
