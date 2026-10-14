#pragma once

#include <torch/torch.h>

namespace rtp_llm {
// Add final physical expert IDs to a persistent int64 buffer on the current
// stream. No allocations or device-to-host reads, including during capture.
void recordExpertLoad(const torch::Tensor& ids, const torch::Tensor& counts);
}  // namespace rtp_llm
