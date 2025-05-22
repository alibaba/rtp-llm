#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <cassert>

namespace rtp_llm {

inline std::string makeCacheKey(const std::string& token_id_str, size_t layer_id) {
    return "token_id_str_" + token_id_str + "_layer_id_" + std::to_string(layer_id);
}

}
