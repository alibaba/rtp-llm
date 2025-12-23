#pragma once
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <vector>
#include <torch/python.h>
#include "rtp_llm/cpp/core/Buffer.h"

namespace rtp_llm {

struct MMPreprocessConfig {
    int32_t            width          = -1;
    int32_t            height         = -1;
    int32_t            min_pixels     = -1;
    int32_t            max_pixels     = -1;
    int32_t            fps            = -1;
    int32_t            min_frames     = -1;
    int32_t            max_frames     = -1;
    std::vector<float> crop_positions = {};
    int32_t            mm_timeout_ms  = 30000;
    MMPreprocessConfig(int32_t            width          = -1,
                       int32_t            height         = -1,
                       int32_t            min_pixels     = -1,
                       int32_t            max_pixels     = -1,
                       int32_t            fps            = -1,
                       int32_t            min_frames     = -1,
                       int32_t            max_frames     = -1,
                       std::vector<float> crop_positions = {},
                       int32_t            mm_timeout_ms  = 30000):
        width(width),
        height(height),
        min_pixels(min_pixels),
        max_pixels(max_pixels),
        fps(fps),
        min_frames(min_frames),
        max_frames(max_frames),
        crop_positions(crop_positions),
        mm_timeout_ms(mm_timeout_ms) {}
};

struct MultimodalInput {
    // public:
    std::string        url;
    torch::Tensor      tensor               = torch::empty({0});
    int32_t            mm_type              = 0;
    MMPreprocessConfig mm_preprocess_config = MMPreprocessConfig();
    std::string        data;
    MultimodalInput(std::string        url,
                    torch::Tensor      t,
                    int32_t            mm_type        = 0,
                    int32_t            width          = -1,
                    int32_t            height         = -1,
                    int32_t            min_pixels     = -1,
                    int32_t            max_pixels     = -1,
                    int32_t            fps            = -1,
                    int32_t            min_frames     = -1,
                    int32_t            max_frames     = -1,
                    std::vector<float> crop_positions = {},
                    int32_t            mm_timeout_ms  = 30000,
                    std::string        data           = ""):
        url(url),
        tensor(t),
        mm_type(mm_type),
        mm_preprocess_config(MMPreprocessConfig(
            width, height, min_pixels, max_pixels, fps, min_frames, max_frames, crop_positions, mm_timeout_ms)),
        data(data) {}
    MultimodalInput(std::string url): url(url), tensor(torch::empty(0)) {}
};

struct MultimodalOutput {
    std::vector<torch::Tensor>                mm_features     = {};
    std::optional<std::vector<torch::Tensor>> mm_position_ids = std::nullopt;
};

class MultimodalFeature {
public:
    std::vector<torch::Tensor>   features;
    std::vector<MultimodalInput> inputs;
    rtp_llm::BufferPtr           text_tokens_mask;  // text part for 1 and multimodal part for 0
    rtp_llm::BufferPtr           locs;              // multimodal input locations
    rtp_llm::BufferPtr           expanded_ids;
    MultimodalFeature() {}
    std::string debugString() const {
        std::stringstream debug_string;
        debug_string << "MultimodalFeature {"
                     << "features: " << features.size() << ", inputs: " << inputs.size()
                     << ", text_tokens_mask: " << text_tokens_mask->debugStringWithData<int32_t>()
                     << ", locs: " << locs->debugStringWithData<int32_t>()
                     << ", expanded_ids: " << expanded_ids->debugStringWithData<int32_t>() << "}";
        return debug_string.str();
    }
};

}  // namespace rtp_llm
