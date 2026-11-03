#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>

namespace rtp_llm {

// Commit one sampled draft step without changing sampling or RNG. Resolves
// compact candidate IDs and optional vocabulary mapping, writes the final
// strided token/probability columns, and optionally gathers the next Markov
// embedding. Copies embedding bits exactly. Unsupported layouts return false
// before launching, allowing the caller to keep the existing Torch path.
bool tryDSparkDraftStepCommit(const torch::Tensor& sampled_tokens,
                              const torch::Tensor& candidate_ids,
                              const torch::Tensor& probabilities,
                              const torch::Tensor& d2t_map,
                              const torch::Tensor& markov_w1,
                              const torch::Tensor& output_tokens,
                              const torch::Tensor& output_probabilities,
                              const torch::Tensor& output_candidate_ids,
                              const torch::Tensor& next_embedding);

// Single-launch statistics for contiguous CUDA FP32 [B, gamma] rates. Lengths
// may be strided int32/int64 and eligibility may be strided bool. Returns an
// undefined tensor for layouts or reductions requiring the Torch fallback.
// Columns are observed count, accepted count, prediction sum, absolute error
// sum, squared error sum, and invalid count. Uses the current PyTorch stream.
torch::Tensor tryDSparkConfidenceStats(const torch::Tensor& confidence_rates,
                                       const torch::Tensor& accept_len,
                                       const torch::Tensor& valid_lengths = {},
                                       const torch::Tensor& eligible_rows = {});

// Finish the confidence head from sampled previous-token embeddings. The
// embedding can be FP32, BF16, or FP16; all arithmetic and other floating-point
// tensors are FP32. Strided int32/int64 [B] anchors and [B, gamma] tokens are
// read without conversion. No synchronization occurs.
void invokeDSparkConfidenceRates(const torch::Tensor& anchors,
                                 const torch::Tensor& token_ids,
                                 const torch::Tensor& markov_w1,
                                 const torch::Tensor& confidence_w_markov,
                                 const torch::Tensor& hidden_logits,
                                 torch::Tensor&       confidence_rates,
                                 float                temperature,
                                 cudaStream_t         stream);

}  // namespace rtp_llm
