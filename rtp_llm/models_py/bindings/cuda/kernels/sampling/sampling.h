#pragma once

#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

#include <torch/extension.h>

namespace rtp_llm {

std::tuple<uint64_t, uint64_t> get_seed_and_offset(int                          increment_size,
                                                   std::optional<at::Generator> generator = std::nullopt);

// Reserve each row's Philox range in input order. Consecutive rows sharing
// a generator reserve one range while retaining the per-row offsets of
// repeated get_seed_and_offset calls. Host materialization is eager-only.
void get_sampling_seed_offsets(const std::vector<at::Generator>& generators,
                               int64_t                           batch_size,
                               int                               increment_size,
                               int64_t*                          seeds,
                               int64_t*                          offsets);

// Fused softmax + top-p filtering + sampling for target and draft.
// Inputs/outputs are contiguous on the same CUDA device: FP32 logits/probs
// [rows, 1024], int32 sampled candidate IDs [rows], optional bool success [rows].
// top_p and logit_scale are FP32 [rows] arrays or scalars.
// RNG follows sampling_from_probs: int64 [rows] arrays use subsequence 0;
// scalar seed/offset values use subsequence=row. Output probabilities describe
// the actual sampling distribution, including top-p filtering. Softmax reduction
// order can change support at a floating-point top-p boundary relative to ATen.
void fused_softmax_top_p_sampling(torch::Tensor                scaled_logits,
                                  torch::Tensor                q_out,
                                  torch::Tensor                token_out,
                                  std::optional<torch::Tensor> maybe_top_p_arr,
                                  double                       top_p_val,
                                  std::optional<torch::Tensor> maybe_logit_scale,
                                  double                       logit_scale_val,
                                  std::optional<torch::Tensor> maybe_seed_arr,
                                  uint64_t                     seed_val,
                                  std::optional<torch::Tensor> maybe_offset_arr,
                                  uint64_t                     offset_val,
                                  std::optional<torch::Tensor> maybe_success_out,
                                  int64_t                      cuda_stream);

void sampling_from_probs(torch::Tensor                probs,
                         torch::Tensor                output,
                         torch::Tensor                valid,
                         std::optional<torch::Tensor> maybe_indices,
                         bool                         deterministic,
                         std::optional<torch::Tensor> maybe_seed_arr,
                         uint64_t                     seed_val,
                         std::optional<torch::Tensor> maybe_offset_arr,
                         uint64_t                     offset_val,
                         int64_t                      cuda_stream = 0);

void top_p_sampling_from_probs(torch::Tensor                probs,
                               torch::Tensor                output,
                               torch::Tensor                valid,
                               std::optional<torch::Tensor> maybe_indices,
                               std::optional<torch::Tensor> maybe_top_p_arr,
                               double                       top_p_val,
                               bool                         deterministic,
                               std::optional<torch::Tensor> maybe_seed_arr,
                               uint64_t                     seed_val,
                               std::optional<torch::Tensor> maybe_offset_arr,
                               uint64_t                     offset_val,
                               int64_t                      cuda_stream = 0);

void top_k_sampling_from_probs(torch::Tensor                probs,
                               torch::Tensor                output,
                               torch::Tensor                valid,
                               std::optional<torch::Tensor> maybe_indices,
                               std::optional<torch::Tensor> maybe_top_k_arr,
                               int64_t                      top_k_val,
                               bool                         deterministic,
                               std::optional<torch::Tensor> maybe_seed_arr,
                               uint64_t                     seed_val,
                               std::optional<torch::Tensor> maybe_offset_arr,
                               uint64_t                     offset_val,
                               int64_t                      cuda_stream = 0);

void top_k_top_p_sampling_from_probs(torch::Tensor                probs,
                                     torch::Tensor                output,
                                     torch::Tensor                valid,
                                     std::optional<torch::Tensor> maybe_indices,
                                     std::optional<torch::Tensor> maybe_top_k_arr,
                                     int64_t                      top_k_val,
                                     std::optional<torch::Tensor> maybe_top_p_arr,
                                     double                       top_p_val,
                                     bool                         deterministic,
                                     std::optional<torch::Tensor> maybe_seed_arr,
                                     uint64_t                     seed_val,
                                     std::optional<torch::Tensor> maybe_offset_arr,
                                     uint64_t                     offset_val,
                                     int64_t                      cuda_stream = 0);

void top_p_renorm_probs(torch::Tensor                probs,
                        torch::Tensor                renorm_probs,
                        std::optional<torch::Tensor> maybe_top_p_arr,
                        double                       top_p_val,
                        int64_t                      cuda_stream = 0);

void top_k_renorm_probs(torch::Tensor                probs,
                        torch::Tensor                renorm_probs,
                        std::optional<torch::Tensor> maybe_top_k_arr,
                        int64_t                      top_k_val,
                        int64_t                      cuda_stream = 0);

}  // namespace rtp_llm
