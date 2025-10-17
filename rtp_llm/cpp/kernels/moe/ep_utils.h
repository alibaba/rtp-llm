#pragma once

#include <c10/core/ScalarType.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

namespace rtp_llm {

template<typename T>
inline T* get_ptr(torch::Tensor& t) {
    return reinterpret_cast<T*>(t.data_ptr());
}

template<typename T>
inline const T* get_ptr(const torch::Tensor& t) {
    return reinterpret_cast<const T*>(t.data_ptr());
}

void moe_pre_reorder(const torch::Tensor&                input,                 // [n_token, hidden]
                     const torch::Tensor&                topk_ids,              // [n_token, topk]
                     const torch::Tensor&                token_expert_indices,  // [n_token, topk]
                     const std::optional<torch::Tensor>& expert_map,            // [n_expert]
                     int64_t                             n_expert,
                     int64_t                             n_local_expert,
                     int64_t                             topk,
                     const std::optional<int64_t>&       align_block_size,
                     torch::Tensor&                      permuted_input,             // [permuted_size, hidden]
                     torch::Tensor&                      expert_first_token_offset,  // [n_local_expert + 1]
                     torch::Tensor&                      inv_permuted_idx,           // [n_token, topk]
                     torch::Tensor&                      permuted_idx                // [permute_size]
);

void moe_post_reorder(const torch::Tensor&                permuted_hidden_states,     // [n_token * topk, hidden]
                      const torch::Tensor&                topk_weights,               // [n_token, topk]
                      const torch::Tensor&                inv_permuted_idx,           // [topk, n_token]
                      const std::optional<torch::Tensor>& expert_first_token_offset,  // [n_local_expert+1]
                      int64_t                             topk,
                      torch::Tensor&                      hidden_states  // [n_token, hidden]
);

}  // namespace rtp_llm