#pragma once

#include <torch/extension.h>

namespace rtp_llm {

// Returns one CUDA int64 budget. -1 denotes invalid confidence/mapping/caps;
// -2 denotes invalid costs. The caller owns the single scalar host boundary.
torch::Tensor glm5SelectDraftTokenBudgetDevice(const torch::Tensor& confidence,
                                               const torch::Tensor& request_slots,
                                               const torch::Tensor& capacities,
                                               const torch::Tensor& total_costs,
                                               int64_t limit,
                                               int64_t sampling_requests);

}  // namespace rtp_llm
