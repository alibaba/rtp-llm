#pragma once

#include <torch/all.h>
#include <cuda_runtime_api.h>

namespace rtp_llm {

void invokeApplyXGrammarBitmaskInplace(torch::Tensor& logits,
                                       const torch::Tensor& bitmask,
                                       int64_t vocab_size,
                                       cudaStream_t stream);

}  // namespace rtp_llm
