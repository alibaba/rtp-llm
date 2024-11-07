#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <cassert>
#include <cstring>

namespace rtp_llm {

// TODO(xinfei.sxf) optimize repeated code
template<typename T>
inline std::string vectorToString(T* begin, T* end) {
    std::string result;
    while (begin != end) {
        result += std::to_string(*begin) + ",";
        begin++;
    }
    if (!result.empty()) {
        result.pop_back();
    }
    return result;
}

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

inline bool startsWith(const std::string& str, const std::string& prefix) {
    if (str.size() < prefix.size()) return false;
    return str.compare(0, prefix.size(), prefix) == 0;
}

}
