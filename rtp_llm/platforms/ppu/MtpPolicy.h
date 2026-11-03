#pragma once

#include <algorithm>
#include "rtp_llm/platforms/MtpPolicy.h"

namespace rtp_llm::platforms::ppu {

inline torch::Tensor selectAcceptedHiddenStates(const torch::Tensor& hidden,
                                               const torch::Tensor& indices,
                                               int64_t batch_size,
                                               int64_t verify_width) {
    TORCH_CHECK(hidden.defined() && hidden.dim() >= 2 && batch_size > 0 && verify_width > 0
                    && hidden.size(0) == batch_size * verify_width,
                "MTP hidden states must have batch*verify_width rows");
    TORCH_CHECK(indices.defined() && indices.numel() == batch_size
                    && indices.scalar_type() == torch::kInt64 && indices.device() == hidden.device(),
                "MTP hidden indices must be int64 [batch] on the hidden-state device");
    const auto feature_size = hidden.numel() / hidden.size(0);
    auto rows = hidden.reshape({batch_size, verify_width, feature_size});
    auto selected = indices.reshape({batch_size, 1, 1}).expand({batch_size, 1, feature_size});
    auto shape = hidden.sizes().vec();
    shape[0] = batch_size;
    return rows.gather(1, selected).reshape(shape);
}

inline bool cloneFixedCacheSnapshot(const std::vector<CacheGroupType>& groups) {
    return std::find(groups.begin(), groups.end(), CacheGroupType::LINEAR) == groups.end();
}

inline MtpPolicy mtpPolicy(const std::string& model_type, int64_t hidden_size, int64_t num_layers) {
    if (model_type != "deepseek_v4" || hidden_size != 4096 || num_layers != 43) {
        return {};
    }
    // Keep accepted pre-HC rows; the dense model buffer also contains rejected rows.
    return {true, selectAcceptedHiddenStates, cloneFixedCacheSnapshot};
}

}  // namespace rtp_llm::platforms::ppu
