#pragma once

#include <torch/torch.h>

namespace rtp_llm {
// Add local expert counts to the int32 output on the current CUDA stream.
// The caller clears once per step. Temporary histogram storage is internal.
void recordExpertLoad(const torch::Tensor& ids, const torch::Tensor& expert_token_counts);
}  // namespace rtp_llm
