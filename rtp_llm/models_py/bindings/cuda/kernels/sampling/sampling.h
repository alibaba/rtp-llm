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

// Fused softmax + top-p (nucleus) + CDF sampling, one kernel, parameterised so the
// same implementation serves the DSpARK draft path and the compact target path.
// Pinned to width 1024 (one block per row: 256 threads x 4 candidates).
//   scaled_logits: [rows, 1024] FP32 (temperature-scaled below, see logit_scale).
//   q_out:         [rows, 1024] FP32, top-p renormalised distribution, zero outside
//                  the nucleus (identical support to TopPRenormProb).
//   token_out:     [rows] int32 sampled candidate index; success_out [rows] bool
//                  optional (reports whether a token was actually drawn).
//   logit_scale:   per-row [rows] FP32 or scalar; probs = softmax(logits * scale).
//   top_p:         per-row [rows] FP32 or scalar.
//   seed/offset:   CUDA int64, [rows] when per_row_seeds (flashinfer array
//                  semantics: philox subsequence 0, like sampleGreedy's target
//                  path) or [1] when not (one range per launch, subsequence = row,
//                  like the draft path).
void dspark_fused_draft_sample(torch::Tensor                scaled_logits,
                               torch::Tensor                q_out,
                               torch::Tensor                token_out,
                               std::optional<torch::Tensor> maybe_top_p_arr,
                               double                       top_p_val,
                               std::optional<torch::Tensor> maybe_logit_scale,
                               double                       logit_scale_val,
                               const torch::Tensor&         seed,
                               const torch::Tensor&         offset,
                               std::optional<torch::Tensor> maybe_success_out,
                               bool                         per_row_seeds,
                               int64_t                      cuda_stream);

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
