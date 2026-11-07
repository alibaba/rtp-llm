#pragma once

#include <torch/extension.h>

namespace rtp_llm {

// Retain pinned storage through GPU reads by fusedCopy on the current stream.
// Non-allocator-owned host storage is copied to an owned pinned allocation.
torch::Tensor prepareFusedCopyHostSource(const torch::Tensor& source);

}  // namespace rtp_llm
