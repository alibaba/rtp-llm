#pragma once

#include "torch/all.h"
#include <cuda_runtime.h>

namespace rtp_llm {

// Standalone op functions that call CUDA kernels directly without exec_ctx_-> dispatch.

// In-place softmax over rows (no mask, no bias).
// input: [batch, n] on CUDA, must be FP32/FP16/BF16
// Modifies input in-place.
void cudaSoftmaxInplace(torch::Tensor& input, cudaStream_t stream);

// Masks logits with a uint8 mask.
// logits: [batch_size, vocab_size] on CUDA, FP32/FP16/BF16
// mask: [batch_size, vocab_size] on CUDA, UINT8
// Modifies logits in-place: sets masked positions to -inf.
void cudaMaskLogits(torch::Tensor& logits, const torch::Tensor& mask, cudaStream_t stream);

}  // namespace rtp_llm
