#pragma once

#include <torch/torch.h>

#ifdef ENABLE_FP4

namespace rtp_llm {

#if defined(USING_CUDA13) && USING_CUDA13
// Dense/group16 NVFP4 with an externally calibrated inverse global scale.
// Unlike the fused-amax fast path this also accepts FP16, K % 16 and rowwise
// calibration. The caller owns the buffers and the calibration lifetime.
void nvfp4_quantize(const torch::Tensor& input,
                    const torch::Tensor& inverse_global_scale,
                    torch::Tensor&       output,
                    torch::Tensor&       output_scale);

void nvfp4_quantize_strided_from_amax(const torch::Tensor& input,
                                      const torch::Tensor& amax_partials,
                                      torch::Tensor&       output,
                                      torch::Tensor&       output_scale,
                                      torch::Tensor&       global_scale,
                                      bool                 compute_amax = false);
#endif

// Forward declarations for FP4 quantization functions
void scaled_fp4_experts_quant_sm100a(torch::Tensor&       output,
                                     torch::Tensor&       output_scale,
                                     torch::Tensor const& input,
                                     torch::Tensor const& input_global_scale,
                                     torch::Tensor const& input_offset_by_experts,
                                     torch::Tensor const& output_scale_offset_by_experts);

void silu_and_mul_scaled_fp4_experts_quant_sm100a(torch::Tensor&       output,
                                                  torch::Tensor&       output_scale,
                                                  torch::Tensor const& input,
                                                  torch::Tensor const& input_global_scale,
                                                  torch::Tensor const& mask,
                                                  bool                 use_silu_and_mul);

}  // namespace rtp_llm

#endif  // ENABLE_FP4
