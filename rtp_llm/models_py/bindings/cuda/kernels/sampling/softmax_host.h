#pragma once

#include <torch/all.h>

namespace rtp_llm {
torch::Tensor samplerSoftmaxInplace(const torch::Tensor& logits);
}  // namespace rtp_llm
