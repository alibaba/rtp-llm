#pragma once

#include <torch/all.h>

#include <cstdint>

namespace rtp_llm {
namespace speculative {

// Gather [B, markov_rank] rows from markov_w1 using int32 indices directly,
// eliminating .to(torch::kLong) + index_select on the DSpARK proposal hot path.
void launchGatherMarkovEmbedding(const torch::Tensor& markov_w1,
                                 const torch::Tensor& indices,
                                 torch::Tensor&       out);

// Fused DSpARK step logits computation:
//   out[b, v] = (float(base_logits[b * propose_step + step, v]) + float(markov_bias[b, v])) / temperature[b]
// Supports base_logits in kBFloat16 or kFloat32, and optional markov_bias (undefined when has_markov is false)
// in kBFloat16 or kFloat32.
void launchDSparkAddMarkovDivTemp(const torch::Tensor& base_logits,
                                  const torch::Tensor& markov_bias,
                                  const torch::Tensor& temperature,
                                  torch::Tensor&       out_logits,
                                  int64_t              step,
                                  int64_t              propose_step,
                                  int64_t              draft_vocab_size);

// Prepare per-replay device inputs for a DSpARK CUDA graph:
// - Populates per-(step, row) Philox seed/offset arrays on device
// - Clamps padded batch tail [live_bsz, padded_bsz) to anchor=0 and temperature=1.0f
void launchPrepareDSparkDraftGraphInputs(torch::Tensor& anchors_slot,
                                         torch::Tensor& temperature_slot,
                                         torch::Tensor& seed_arr,
                                         torch::Tensor& offset_arr,
                                         int64_t        live_bsz,
                                         int64_t        padded_bsz,
                                         int64_t        gamma,
                                         uint64_t       seed,
                                         uint64_t       base_offset);

// Map [B] int32 draft token IDs through int64/int32 d2t_map directly into out_tokens[b]
// (which may have an arbitrary outer stride, e.g. column `step` of [B, propose_step]).
void launchMapDraftToTargetInt32(const torch::Tensor& draft_tokens,
                                   const torch::Tensor& d2t_map,
                                   torch::Tensor&       out_tokens);

// Single-launch post-rejection + optional force_accept + optional spec_cap clamping kernel.
// out_tokens/out_accept_len may alias in_tokens/in_accept_len (single thread per row, race-free).
void launchPostRejectionAndCap(const torch::Tensor& in_tokens,
                               const torch::Tensor& in_accept_len,
                               torch::Tensor&       out_tokens,
                               torch::Tensor&       out_accept_len,
                               const torch::Tensor& force_mask,
                               const torch::Tensor& draft_token_ids,
                               const torch::Tensor& target_token_ids,
                               const torch::Tensor& spec_cap,
                               bool                 clamp_minus_one);

}  // namespace speculative
}  // namespace rtp_llm
