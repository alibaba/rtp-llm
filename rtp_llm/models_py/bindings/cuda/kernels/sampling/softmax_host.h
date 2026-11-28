#pragma once

#include <torch/all.h>

namespace rtp_llm {

// Whether the multi-CTA split softmax domain covers this tensor: CUDA,
// contiguous FP32, 2D, [1..256] rows, vocab >= 65536 and divisible by 4 (the
// vec4 loads in samplerOrderedWarpSums require 16B-aligned rows). Exposed so
// wrappers can fall back to a plain torch::softmax (without the in-place
// copy-back) when the domain does not apply.
bool canSamplerSplitSoftmax(const torch::Tensor& logits);

torch::Tensor samplerSoftmaxInplace(const torch::Tensor& logits);
}  // namespace rtp_llm
