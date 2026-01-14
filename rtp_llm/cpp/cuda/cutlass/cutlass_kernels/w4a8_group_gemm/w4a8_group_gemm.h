#pragma once

#include <c10/cuda/CUDAStream.h>
#include <torch/all.h>
#include <torch/python.h>

namespace rtp_llm {

void run_w4a8_group_gemm(torch::Tensor&       output,
                         torch::Tensor const& a,
                         torch::Tensor const& b,
                         torch::Tensor const& b_scales,
                         torch::Tensor const& a_out_scales,
                         torch::Tensor const& b_out_scales,
                         torch::Tensor const& expert_offsets,
                         torch::Tensor const& problem_sizes,
                         torch::Tensor const& a_strides,
                         torch::Tensor const& b_strides,
                         torch::Tensor const& b_scales_strides,
                         torch::Tensor const& c_strides,
                         const int            group_size,
                         const bool           swap_ab,
                         const bool           per_act_token,
                         const bool           per_out_ch,
                         const bool           profile,
                         const int            m_tile,
                         const int            n_tile,
                         const int            k_tile,
                         const int            cluster_m,
                         const int            cluster_n,
                         const int            cluster_k);

torch::Tensor run_unified_encode_int4b(const torch::Tensor& input);

torch::Tensor run_pack_scale_fp8(const torch::Tensor& input);

torch::Tensor run_dequantize_int4b_to_fp8(const torch::Tensor& input,
                                          const torch::Tensor& scale,
                                          const torch::Tensor& zero,
                                          const int            group_size);

void run_initialize_tensor(torch::Tensor& output, std::optional<float> min, std::optional<float> max, const int seed);

bool run_block_compare_relative(const torch::Tensor& a,
                                const torch::Tensor& b,
                                const float          epsilon,
                                const float          nonzero_floor);

}  // namespace rtp_llm
