#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <cassert>

#include "rtp_llm/cpp/cache/CacheGroupType.h"

namespace rtp_llm {

inline std::string makeCacheKey(size_t model_id, const std::string& token_id_str, size_t layer_id) {
    return "model_id_" + std::to_string(model_id) + "_token_id_str_" + token_id_str + "_layer_id_"
           + std::to_string(layer_id);
}

inline std::string
makeCacheKey(size_t model_id, const std::string& token_id_str, size_t layer_id, KVCacheRegionName region_name) {
    if (region_name == KVCacheRegionName::DEFAULT) {
        return makeCacheKey(model_id, token_id_str, layer_id);
    }
    return makeCacheKey(model_id, token_id_str, layer_id) + "_region_" + std::to_string(static_cast<int>(region_name));
}

}  // namespace rtp_llm
