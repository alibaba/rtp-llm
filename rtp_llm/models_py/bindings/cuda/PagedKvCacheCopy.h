#pragma once

#include <torch/extension.h>
#include <array>
#include <vector>

namespace rtp_llm {

// Enqueue disjoint page runs on the current CUDA stream. Both tensors must
// remain alive until that stream finishes; the CPU side must be pinned.
void copyPagedKvCacheAsync(const torch::Tensor&                       source,
                           const torch::Tensor&                       destination,
                           const std::vector<std::array<int64_t, 3>>& runs);

}  // namespace rtp_llm
