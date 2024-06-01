#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <cassert>
#include <cstring>

namespace rtp_llm {

template<typename T>
std::string vectorToString(const std::vector<T>& vec) {
    std::string result;
    for (const auto& value : vec) {
        result += std::to_string(value) + ",";
    }
    if (!result.empty()) {
        result.pop_back();
    }
    return result;
}

template<typename T>
std::string vectorsToString(const std::vector<std::vector<T>>& vecs) {
    std::string result;
    result += "[";
    for (const auto& vec: vecs) {
        result += "[" + vectorToString(vec) + "], ";
    }
    if (!vecs.empty()) {
        result.pop_back();
        result.pop_back();
    }
    result += "]";
    return result;
}

}
