#pragma once

#include <string>
#include <torch/torch.h>

namespace rtp_llm {

// TODO: stub. Currently returns the weight unchanged on both CUDA and ROCm.
// Kept for ABI compatibility with pybind exports — re-evaluate for removal.
torch::Tensor preprocessGemmWeightByKey(const std::string& key, torch::Tensor weight, bool user_arm_gemm_use_kai);
torch::Tensor preprocessWeightScale(torch::Tensor weight, torch::Tensor scale);

}  // namespace rtp_llm
