#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

namespace rtp_llm {

// Greedy sampling over ordinary [batch, vocab] logits, including strided views.
// Token IDs are int32 [batch, 1]; one-hot probabilities keep the logits dtype.
// Only token IDs apply the optional draft-to-target vocabulary mapping.
void invokeMtpGreedySampling(const torch::Tensor& logits,
                             const torch::Tensor& d2t_map,
                             torch::Tensor&       token_ids,
                             torch::Tensor&       probabilities,
                             cudaStream_t         stream);

}  // namespace rtp_llm
