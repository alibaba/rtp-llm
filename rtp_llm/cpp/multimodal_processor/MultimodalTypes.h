#pragma once
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <vector>
#include <torch/python.h>
#include <torch/python.h>

namespace rtp_llm {

struct MMPreprocessConfig {
    int32_t width      = -1;
    int32_t height     = -1;
    int32_t min_pixels = -1;
    int32_t max_pixels = -1;
    int32_t fps        = -1;
    int32_t min_frames = -1;
    int32_t max_frames = -1;
    MMPreprocessConfig(int32_t width      = -1,
                       int32_t height     = -1,
                       int32_t min_pixels = -1,
                       int32_t max_pixels = -1,
                       int32_t fps        = -1,
                       int32_t min_frames = -1,
                       int32_t max_frames = -1):
        width(width),
        height(height),
        min_pixels(min_pixels),
        max_pixels(max_pixels),
        fps(fps),
        min_frames(min_frames),
        max_frames(max_frames) {}
};

struct MultimodalInput {
    // public:
    std::string        url;
    torch::Tensor      tensor               = torch::empty({0});
    int32_t            mm_type              = 0;
    MMPreprocessConfig mm_preprocess_config = MMPreprocessConfig();
    MultimodalInput(std::string   url,
                    torch::Tensor t,
                    int32_t       mm_type    = 0,
                    int32_t       width      = -1,
                    int32_t       height     = -1,
                    int32_t       min_pixels = -1,
                    int32_t       max_pixels = -1,
                    int32_t       fps        = -1,
                    int32_t       min_frames = -1,
                    int32_t       max_frames = -1):
        url(url),
        tensor(t),
        mm_type(mm_type),
        mm_preprocess_config(MMPreprocessConfig(width, height, min_pixels, max_pixels, fps, min_frames, max_frames)) {}
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
