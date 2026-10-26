#pragma once

#include <string>
#include <vector>
#include <torch/torch.h>
#include "rtp_llm/cpp/cache/CacheGroupType.h"

namespace rtp_llm::platforms {

// Executor extension points; an empty policy preserves the existing MTP path.
// Implementations live under the owning platform, not in the shared executor.
struct MtpPolicy {
    using HiddenSelector = torch::Tensor (*)(const torch::Tensor&, const torch::Tensor&, int64_t, int64_t);
    // Do not replace accepted hidden rows with the model's dense MTP buffer.
    bool preserve_accepted_hidden = false;
    // Select accepted rows while preserving model-specific trailing dimensions.
    HiddenSelector select_hidden = nullptr;
    // Request an independent fixed-cache table snapshot for async draft work.
    bool (*clone_cache_snapshot)(const std::vector<CacheGroupType>&) = nullptr;
};

// Only PPU Flash currently opts in. DSpARK and all other models use an empty policy.
MtpPolicy mtpPolicy(const std::string& model_type, bool is_dspark = false, int64_t hidden_size = 0, int64_t num_layers = 0);

}  // namespace rtp_llm::platforms
