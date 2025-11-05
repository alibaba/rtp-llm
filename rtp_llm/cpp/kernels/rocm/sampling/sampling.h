#pragma once

#include <torch/extension.h>

namespace rtp_llm {

void top_p_renorm_probs(torch::Tensor probs, torch::Tensor renorm_probs,
                        std::optional<torch::Tensor> maybe_top_p_arr, double top_p_val, uintptr_t stream = 0);
void top_k_renorm_probs(torch::Tensor probs, torch::Tensor renorm_probs,
                        std::optional<torch::Tensor> maybe_top_k_arr, int64_t top_k_val, uintptr_t stream = 0);

void top_p_sampling_from_probs(torch::Tensor probs, torch::Tensor output,
                               std::optional<torch::Tensor> maybe_indices,
                               std::optional<torch::Tensor> maybe_top_p_arr, double top_p_val,
                               bool deterministic, uint64_t philox_seed, uint64_t philox_offset, uintptr_t stream = 0);
void top_k_sampling_from_probs(torch::Tensor probs, torch::Tensor output,
                               std::optional<torch::Tensor> maybe_indices,
                               std::optional<torch::Tensor> maybe_top_k_arr, int64_t top_k_val,
                               bool deterministic, uint64_t philox_seed, uint64_t philox_offset, uintptr_t stream = 0);
void top_k_top_p_sampling_from_probs(torch::Tensor probs, torch::Tensor output,
                                     std::optional<torch::Tensor> maybe_indices,
                                     std::optional<torch::Tensor> maybe_, double top_k_val,
                                     std::optional<torch::Tensor> maybe_top_p_arr, double top_p_val,
                                     bool deterministic, uint64_t philox_seed,
                                     uint64_t philox_offset, uintptr_t stream = 0);
}