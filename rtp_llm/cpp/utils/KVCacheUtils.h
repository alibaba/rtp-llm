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
makeCacheKey(size_t model_id, const std::string& token_id_str, size_t layer_id, KVCacheAttnType attn_type) {
    if (attn_type == KVCacheAttnType::DEFAULT) {
        return makeCacheKey(model_id, token_id_str, layer_id);
    }
    return makeCacheKey(model_id, token_id_str, layer_id) + "_attn_type_" + std::to_string(static_cast<int>(attn_type));
}

}  // namespace rtp_llm
