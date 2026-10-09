#pragma once
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <vector>
#include <torch/python.h>
#include "rtp_llm/cpp/multimodal_processor/MultimodalInputClass.h"

namespace rtp_llm {
struct MultimodalExpansionMetadata {
    bool                 is_video              = false;
    double               fps                   = 0.0;
    std::vector<int32_t> frame_indices         = {};
    int32_t              frame_number          = 0;
    int32_t              frame_count           = 1;
    int32_t              soft_tokens_per_frame = 0;
};

struct MultimodalOutput {
    std::vector<torch::Tensor>                              mm_features           = {};
    std::optional<std::vector<torch::Tensor>>               mm_position_ids       = std::nullopt;
    std::optional<std::vector<torch::Tensor>>               mm_extra_input        = std::nullopt;
    std::optional<std::vector<MultimodalExpansionMetadata>> mm_expansion_metadata = std::nullopt;
    std::optional<torch::Tensor>                            expanded_token_ids    = std::nullopt;
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
