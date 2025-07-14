#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <cassert>

namespace rtp_llm {

inline std::string makeCacheKey(size_t model_id, const std::string& token_id_str, size_t layer_id) {
    return "model_id_" + std::to_string(model_id) + "_token_id_str_" + token_id_str + "_layer_id_"
           + std::to_string(layer_id);
}

}  // namespace rtp_llm
