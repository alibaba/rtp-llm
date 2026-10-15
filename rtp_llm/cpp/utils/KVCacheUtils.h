#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <cassert>

#include "rtp_llm/cpp/cache/CacheModuleId.h"

namespace rtp_llm {

inline std::string makeCacheKey(CacheModuleId module_id, const std::string& token_id_str, size_t layer_id) {
    return "model_id_" + std::to_string(module_id.value()) + "_token_id_str_" + token_id_str + "_layer_id_"
           + std::to_string(layer_id);
}

inline std::string
makeCacheKey(CacheModuleId module_id, const std::string& token_id_str, size_t layer_id, const std::string& tag) {
    if (tag.empty() || tag == "default") {
        return makeCacheKey(module_id, token_id_str, layer_id);
    }
    return makeCacheKey(module_id, token_id_str, layer_id) + "_tag_" + tag;
}

}  // namespace rtp_llm
