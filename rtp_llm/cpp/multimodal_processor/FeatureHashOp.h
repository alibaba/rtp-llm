#pragma once

#include <torch/types.h>
#include <vector>

namespace rtp_llm {

torch::Tensor getMultimodalFeatureHash(const torch::Tensor& embedding);

// Identical row hashes, with one completed D2H for a same-device CUDA batch.
std::vector<torch::Tensor> getMultimodalFeatureHashes(const std::vector<torch::Tensor>& embeddings);

}  // namespace rtp_llm
