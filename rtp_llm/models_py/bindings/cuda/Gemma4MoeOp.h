#pragma once

#include <torch/extension.h>

#include <tuple>

namespace torch_ext {

std::tuple<at::Tensor, at::Tensor> gemma4_topk_8_bf16(const at::Tensor& input);
at::Tensor                         gemma4_weighted_reorder_bf16(const at::Tensor& expert_output,
                                                                const at::Tensor& sorted_weight,
                                                                const at::Tensor& inverse_permutation);
at::Tensor
gemma4_gather_sorted_expert_input_bf16(const at::Tensor& input, const at::Tensor& permutation, int64_t top_k);
at::Tensor gemma4_top8_sum_bf16(const at::Tensor& input);
at::Tensor gemma4_finalize_router_weights_bf16(const at::Tensor& top_weights,
                                               const at::Tensor& top_indices,
                                               const at::Tensor& expert_scales);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
gemma4_prepare_grouped_moe(const at::Tensor& expert_ids, const at::Tensor& weights, int64_t num_experts);

}  // namespace torch_ext
