#pragma once
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

namespace rtp_llm {

int32_t get_sm_version_num();

void cutlass_moe_mm(torch::Tensor&       out_tensors,
                    torch::Tensor const& a_tensors,
                    torch::Tensor const& b_tensors,
                    torch::Tensor const& a_scales,
                    torch::Tensor const& b_scales,
                    torch::Tensor const& expert_offsets,
                    torch::Tensor const& problem_sizes,
                    torch::Tensor const& a_strides,
                    torch::Tensor const& b_strides,
                    torch::Tensor const& c_strides,
                    bool                 per_act_token,
                    bool                 per_out_ch);

void get_cutlass_moe_mm_data(const torch::Tensor&                topk_ids,
                             torch::Tensor&                      expert_offsets,
                             torch::Tensor&                      problem_sizes1,
                             torch::Tensor&                      problem_sizes2,
                             torch::Tensor&                      input_permutation,
                             torch::Tensor&                      output_permutation,
                             const int64_t                       num_experts,
                             const int64_t                       n,
                             const int64_t                       k,
                             const std::optional<torch::Tensor>& blockscale_offsets);

void get_cutlass_batched_moe_mm_data(torch::Tensor&       expert_offsets,
                                     torch::Tensor&       problem_sizes1,
                                     torch::Tensor&       problem_sizes2,
                                     const torch::Tensor& expert_num_tokens,
                                     const int64_t        num_local_experts,
                                     const int64_t        padded_m,
                                     const int64_t        n,
                                     const int64_t        k);

};  // namespace rtp_llm