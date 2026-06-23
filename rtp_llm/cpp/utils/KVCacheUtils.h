#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <cassert>

#include "rtp_llm/cpp/cache/spec/CacheGroupType.h"

namespace rtp_llm {

inline std::string makeCacheKey(size_t model_id, const std::string& token_id_str, size_t layer_id) {
    return "model_id_" + std::to_string(model_id) + "_token_id_str_" + token_id_str + "_layer_id_"
           + std::to_string(layer_id);
}

inline std::string makeCacheKey(size_t model_id, const std::string& token_id_str, size_t layer_id, const std::string& tag) {
    if (tag.empty() || tag == "default") {
        return makeCacheKey(model_id, token_id_str, layer_id);
    }
    return makeCacheKey(model_id, token_id_str, layer_id) + "_tag_" + tag;
}

inline std::string makeCacheKey(size_t model_id, const std::string& token_id_str, size_t layer_id, int group_id) {
    if (group_id == 0) {
        return makeCacheKey(model_id, token_id_str, layer_id);
    }
    return makeCacheKey(model_id, token_id_str, layer_id) + "_group_" + std::to_string(group_id);
}

}  // namespace rtp_llm
