#pragma once
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <vector>
#include <torch/python.h>
#include "rtp_llm/cpp/multimodal_processor/MultimodalInputClass.h"

namespace rtp_llm {
// Backend-only cache identity for a fully assembled multimodal tensor segment.
// Model-specific layout (grid, delimiter kinds, patch metadata) stays inside
// the ViT implementation; the LLM side only needs the final token offset and
// one stable hash per embedding row.
struct MultimodalCacheSpan {
    int32_t              offset = 0;
    std::vector<int32_t> row_hashes;
};

using MultimodalCacheSpans = std::vector<MultimodalCacheSpan>;

struct MultimodalOutput {
    std::vector<torch::Tensor>                mm_features       = {};
    std::optional<std::vector<torch::Tensor>> mm_position_ids   = std::nullopt;
    std::optional<std::vector<torch::Tensor>> mm_extra_input    = std::nullopt;
    std::optional<std::vector<torch::Tensor>> mm_feature_hashes = std::nullopt;
};

class MultimodalFeature {
public:
    std::vector<torch::Tensor>   features;
    std::vector<MultimodalInput> inputs;
    torch::Tensor                text_tokens_mask;  // text part for 1 and multimodal part for 0
    torch::Tensor                locs;              // multimodal input locations
    torch::Tensor                expanded_ids;
    MultimodalFeature() {}
    std::string debugString() const {
        std::stringstream debug_string;
        debug_string << "MultimodalFeature {"
                     << "features: " << features.size() << ", inputs: " << inputs.size()
                     << ", text_tokens_mask: tensor[" << text_tokens_mask.numel() << "]"
                     << ", locs: tensor[" << locs.numel() << "]"
                     << ", expanded_ids: tensor[" << expanded_ids.numel() << "]"
                     << "}";
        return debug_string.str();
    }
};

}  // namespace rtp_llm
