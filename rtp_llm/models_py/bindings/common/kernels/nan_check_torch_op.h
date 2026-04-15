#pragma once

#include <torch/extension.h>
#include <optional>

namespace rtp_llm {

// Decode: check and reset NaN/Inf in KV cache (last token per batch).
// Uses current CUDA/HIP stream internally.
void check_and_reset_nan_kv_cache_decode(const torch::Tensor&                layer_base_addrs,
                                         const torch::Tensor&                kv_cache_block_id,
                                         const torch::Tensor&                sequence_lengths,
                                         torch::Tensor                       nan_flag,
                                         int64_t                             cache_dtype,
                                         int64_t                             batch_size,
                                         int64_t                             layer_num,
                                         int64_t                             num_groups,
                                         const std::optional<torch::Tensor>& layer_to_group,
                                         const std::optional<torch::Tensor>& group_types,
                                         int64_t                             batch_dim,
                                         int64_t                             batch_start,
                                         int64_t                             max_blocks_per_batch,
                                         int64_t                             local_head_num_kv,
                                         int64_t                             k_token_size,
                                         int64_t                             v_token_size,
                                         int64_t                             k_block_size_bytes,
                                         int64_t                             v_block_size_bytes,
                                         int64_t                             k_token_bytes,
                                         int64_t                             v_token_bytes,
                                         int64_t                             block_size_bytes,
                                         int64_t                             seq_size_per_block);

// Prefill: check and reset NaN/Inf in KV cache (prefix_lengths..input_lengths per batch).
// Uses current CUDA/HIP stream internally.
void check_and_reset_nan_kv_cache_prefill(const torch::Tensor&                layer_base_addrs,
                                          const torch::Tensor&                kv_cache_block_id,
                                          const torch::Tensor&                prefix_lengths,
                                          const torch::Tensor&                input_lengths,
                                          torch::Tensor                       nan_flag,
                                          int64_t                             cache_dtype,
                                          int64_t                             batch_size,
                                          int64_t                             layer_num,
                                          int64_t                             num_groups,
                                          const std::optional<torch::Tensor>& layer_to_group,
                                          const std::optional<torch::Tensor>& group_types,
                                          int64_t                             batch_dim,
                                          int64_t                             batch_start,
                                          int64_t                             max_blocks_per_batch,
                                          int64_t                             local_head_num_kv,
                                          int64_t                             k_token_size,
                                          int64_t                             v_token_size,
                                          int64_t                             k_block_size_bytes,
                                          int64_t                             v_block_size_bytes,
                                          int64_t                             k_token_bytes,
                                          int64_t                             v_token_bytes,
                                          int64_t                             block_size_bytes,
                                          int64_t                             seq_size_per_block);

}  // namespace rtp_llm
