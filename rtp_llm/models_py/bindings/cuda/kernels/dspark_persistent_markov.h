#pragma once

#include <torch/extension.h>

namespace rtp_llm {

int64_t dsparkPersistentMarkov(torch::Tensor output,
                               const torch::Tensor& anchor,
                               const torch::Tensor& base_logits,
                               const torch::Tensor& draft_to_target_id,
                               const torch::Tensor& markov_w1,
                               const torch::Tensor& markov_w2,
                               torch::Tensor current_state,
                               torch::Tensor partial_scores,
                               torch::Tensor partial_tokens,
                               torch::Tensor barrier_state,
                               double scale,
                               int64_t requested_max_ctas_per_batch_tile);

}  // namespace rtp_llm
