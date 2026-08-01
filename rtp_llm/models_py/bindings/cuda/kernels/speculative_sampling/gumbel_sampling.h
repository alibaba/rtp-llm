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

// In-place vLLM-compatible temperature transform. Exact zeros and ones are
// no-ops; all other rows divide by T without the epsilon used by the legacy
// RTP sampling penalty.
void applyGumbelTemperature(torch::Tensor logits, const torch::Tensor& temperatures);

// Token-equality verifier for the coupled path. Returns int32
// (accepted_tokens [B,k+1], emitted_length [B]); the first target mismatch is
// emitted directly and no probability-ratio/residual tensor exists.
std::tuple<torch::Tensor, torch::Tensor> coupledTokenVerify(const torch::Tensor& draft_tokens,
                                                            const torch::Tensor& target_tokens);

}  // namespace rtp_llm
