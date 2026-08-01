#pragma once

#include <torch/extension.h>
#include <tuple>

namespace rtp_llm {

// Stateless Gumbel-max keyed by (seed, absolute predecessor position).
// The Philox construction and flipped fp32 Gumbel transform intentionally
// match vLLM/Triton so target and draft can share exact samples without an RNG
// state tensor.  `temperature == 0` selects deterministic argmax.
torch::Tensor gumbelSample(const torch::Tensor& values,
                           const torch::Tensor& seeds,
                           const torch::Tensor& positions,
                           const torch::Tensor& temperatures,
                           bool                 input_is_probs,
                           bool                 apply_temperature,
                           bool                 use_fp64 = false);

// Probability-free greedy token verifier. Returns int32
// (accepted_tokens [B,k+1], emitted_length [B]); the first target mismatch is
// emitted directly and no probability-ratio/residual tensor exists.
std::tuple<torch::Tensor, torch::Tensor> greedyTokenVerify(const torch::Tensor& draft_tokens,
                                                           const torch::Tensor& target_tokens);

}  // namespace rtp_llm
