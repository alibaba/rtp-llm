#pragma once

#include <functional>

namespace rtp_llm {

inline int64_t hashInt64Func(const std::hash<int32_t>& hasher, int64_t hash, int32_t value) {
    // Jenkins hash function (modified for 64 bits)
    hash ^= hasher(value) + 0x9e3779b97f4a7c15 + (hash << 12) + (hash >> 32);
    return hash;
}

inline int64_t hashInt64Array(int64_t hash, int32_t* begin, int32_t* end) {
    std::hash<int32_t> hasher;

    while (begin != end) {
        // Combine the hash of each element
        hash = hashInt64Func(hasher, hash, *begin);
        begin++;
    }

    return hash;
}

inline int64_t hashInt64Vector(int64_t hash, const std::vector<int64_t>& vec) {
    std::hash<int32_t> hasher;

    for (const auto& value : vec) {
        // Combine the hash of each element
        hash = hashInt64Func(hasher, hash, (int32_t)value);
    }

    return hash;
}
}  // namespace rtp_llm
