#pragma once

#include <torch/torch.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000

namespace rtp_llm {

// Forward declarations for FP4 quantization functions
void scaled_fp4_experts_quant_sm100a(
    torch::Tensor& output,
    torch::Tensor& output_scale,
    torch::Tensor const& input,
    torch::Tensor const& input_global_scale,
    torch::Tensor const& input_offset_by_experts,
    torch::Tensor const& output_scale_offset_by_experts);

void silu_and_mul_scaled_fp4_experts_quant_sm100a(
    torch::Tensor& output,
    torch::Tensor& output_scale,
    torch::Tensor const& input,
    torch::Tensor const& input_global_scale,
    torch::Tensor const& mask,
    bool use_silu_and_mul);

}  // namespace rtp_llm

#endif

