#pragma once

#include <torch/extension.h>

namespace rtp_llm {

std::tuple<uint64_t, uint64_t> get_seed_and_offset(int increment_size, std::optional<at::Generator> generator = std::nullopt);

void top_p_renorm_probs(torch::Tensor probs, torch::Tensor renorm_probs,
                        std::optional<torch::Tensor> maybe_top_p_arr, double top_p_val, uintptr_t stream = 0);
void top_k_renorm_probs(torch::Tensor probs, torch::Tensor renorm_probs,
                        std::optional<torch::Tensor> maybe_top_k_arr, int64_t top_k_val, uintptr_t stream = 0);

void top_p_sampling_from_probs(torch::Tensor probs, torch::Tensor output,
                               std::optional<torch::Tensor> maybe_indices,
                               std::optional<torch::Tensor> maybe_top_p_arr, double top_p_val,
                               bool deterministic, torch::Tensor philox_seed, torch::Tensor philox_offset, uintptr_t stream = 0);
void top_k_sampling_from_probs(torch::Tensor probs, torch::Tensor output,
                               std::optional<torch::Tensor> maybe_indices,
                               std::optional<torch::Tensor> maybe_top_k_arr, int64_t top_k_val,
                               bool deterministic, torch::Tensor philox_seed, torch::Tensor philox_offset, uintptr_t stream = 0);
void top_k_top_p_sampling_from_probs(torch::Tensor probs, torch::Tensor output,
                                     std::optional<torch::Tensor> maybe_indices,
                                     std::optional<torch::Tensor> maybe_, double top_k_val,
                                     std::optional<torch::Tensor> maybe_top_p_arr, double top_p_val,
                                     bool deterministic, torch::Tensor philox_seed,
                                     torch::Tensor philox_offset, uintptr_t stream = 0);
}