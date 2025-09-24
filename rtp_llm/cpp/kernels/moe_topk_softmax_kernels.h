// copy from rtp_llm/cpp/cutlass/cutlass_kernels/moe_gemm/moe_kernels.h
// but remove all moeGemm part

#pragma once

#include <torch/extension.h>

#if USING_ROCM
#include "rtp_llm/cpp/rocm/cuda_shims.h"
#endif
namespace rtp_llm {
void topk_softmax(
    torch::Tensor& topk_weights,          // [num_tokens, topk]
    torch::Tensor& topk_indices,          // [num_tokens, topk]
    torch::Tensor& token_expert_indices,  // [num_tokens, topk]
    torch::Tensor& gating_output);
}  // namespace rtp_llm
