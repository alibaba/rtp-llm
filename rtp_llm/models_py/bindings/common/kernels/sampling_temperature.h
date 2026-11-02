#pragma once

#include <torch/all.h>
#include <cuda_runtime.h>

namespace rtp_llm {
void applySamplerTemperature(const torch::Tensor& logits, const torch::Tensor& temperatures, cudaStream_t stream);
}  // namespace rtp_llm
