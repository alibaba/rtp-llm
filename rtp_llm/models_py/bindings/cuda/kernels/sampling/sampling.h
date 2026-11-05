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

// DSpARK fused draft sample: softmax + top-p (nucleus) + CDF sampling in one kernel.
// Replaces softmax + top_p_renorm_probs + sampling_from_probs_device_scalar_seed for
// the compact pure-top-p draft shape (rows = batch, width = 1024). The top-p support
// set and the Philox mapping are pinned to the kernels it replaces (see api.cu);
// the probabilities differ only by summation order (<= 1e-6).
//   scaled_logits: [rows, 1024] FP32, already divided by the temperature.
//   q_out:         [rows, 1024] FP32, top-p renormalised distribution (zero outside
//                  the nucleus, exactly like TopPRenormProb).
//   token_out:     [rows] int32, sampled candidate index in 0..1023.
//   seed/offset:   one-element CUDA int64 scalars, consumed as
//                  curand_init(seed, row, offset) exactly like the eager path.
void dspark_fused_draft_sample(torch::Tensor                scaled_logits,
                               torch::Tensor                q_out,
                               torch::Tensor                token_out,
                               std::optional<torch::Tensor> maybe_top_p_arr,
                               double                       top_p_val,
                               const torch::Tensor&         seed_scalar,
                               const torch::Tensor&         offset_scalar,
                               int64_t                      cuda_stream);

// Sampling identical to flashinfer's scalar-seed SamplingFromProb path, with the
// seed/offset read from one-element device tensors instead of by value. Used by the
// DSpARK stochastic draft CUDA graph: a captured graph freezes by-value arguments,
// so the scalars must live in device memory to be refreshable on replay. The row
// index remains the Philox subsequence (the scalar semantic), so results are
// bit-identical to the eager scalar path for the same (seed, offset).
void sampling_from_probs_device_scalar_seed(torch::Tensor                probs,
                                            torch::Tensor                output,
                                            torch::Tensor                valid,
                                            std::optional<torch::Tensor> maybe_indices,
                                            bool                         deterministic,
                                            const torch::Tensor&         seed_scalar,
                                            const torch::Tensor&         offset_scalar,
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
