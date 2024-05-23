#pragma once

#include <vector>
#include <cstdint>
#include <cassert>
#include <cstring>

namespace rtp_llm {

inline std::string vectorToString(const std::vector<int>& vec) {
    std::string result;
    for (const auto& value : vec) {
        result += std::to_string(value) + ",";
    }
    if (!result.empty()) {
        result.pop_back();
    }
    return result;
}

inline std::string vectorsToString(const std::vector<std::vector<int>>& vecs) {
    std::string result;
    result += "[";
    for (const auto& vec: vecs) {
        result += "[" + vectorToString(vec) + "], ";
    }
    if (!result.empty()) {
        result.pop_back(); 
        result.pop_back(); 
    }
    result += "]";
    return result;
}

}
