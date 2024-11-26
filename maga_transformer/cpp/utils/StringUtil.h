#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdint>
#include <cassert>
#include <cstring>

namespace rtp_llm {

template<typename... Args>
inline std::string fmtstr(const std::string& format, Args... args) {
    // This function came from a code snippet in stackoverflow under cc-by-1.0
    //   https://stackoverflow.com/questions/2342162/stdstring-formatting-like-sprintf

    // Disable format-security warning in this function.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-security"

    int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) + 1;  // Extra space for '\0'
    if (size_s <= 0) {
        throw std::runtime_error("Error during formatting.");
    }
    auto size = static_cast<size_t>(size_s);
    auto buf  = std::make_unique<char[]>(size);
    std::snprintf(buf.get(), size, format.c_str(), args...);

#pragma GCC diagnostic pop
    return std::string(buf.get(), buf.get() + size - 1);  // We don't want the '\0' inside
}

template<typename T>
inline std::vector<std::string> transVectorToString(const std::vector<T>& vec) {
    std::vector<std::string> result;
    for (const auto& value : vec) {
        result.push_back(std::to_string(value));
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
