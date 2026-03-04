#pragma once
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <vector>
#include <torch/python.h>

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
    std::string to_string() const {
        std::string crop_positions_str = "";
        for (const float& crop_position : crop_positions) {
            crop_positions_str += std::to_string(crop_position) + ":";
        }
        if (crop_positions_str.size() > 0) {
            crop_positions_str = crop_positions_str.substr(0, crop_positions_str.size() - 1);
        }
        return std::to_string(width) + "_" + std::to_string(height) + "_" + std::to_string(min_pixels) + "_"
               + std::to_string(max_pixels) + "_" + std::to_string(fps) + "_" + std::to_string(min_frames) + "_"
               + std::to_string(max_frames) + "_" + crop_positions_str + "_" + std::to_string(mm_timeout_ms);
    }
};

class MultimodalInput {
public:
    std::string        url;
    int32_t            mm_type              = 0;
    torch::Tensor      tensor               = torch::empty({0});
    MMPreprocessConfig mm_preprocess_config = MMPreprocessConfig();
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
                    int32_t            mm_timeout_ms  = 30000):
        url(url),
        mm_type(mm_type),
        tensor(t),
        mm_preprocess_config(MMPreprocessConfig(
            width, height, min_pixels, max_pixels, fps, min_frames, max_frames, crop_positions, mm_timeout_ms)) {}
    MultimodalInput(std::string        url,
                    int32_t            mm_type              = 0,
                    torch::Tensor      tensor               = torch::empty({0}),
                    MMPreprocessConfig mm_preprocess_config = MMPreprocessConfig()):
        url(url), mm_type(mm_type), tensor(tensor), mm_preprocess_config(mm_preprocess_config) {}
    std::string to_string() const {
        return url.substr(0, 256) + "_" + std::to_string(mm_type) + "_" + mm_preprocess_config.to_string();
    }
};
}  // namespace rtp_llm
