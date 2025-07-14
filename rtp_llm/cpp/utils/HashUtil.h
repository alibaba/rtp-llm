#pragma once

#include <functional>

namespace rtp_llm {

inline int64_t hashInt64Array(int64_t hash, int32_t* begin, int32_t* end) {
    std::hash<int32_t> hasher;
    auto               hashValue = [&](int64_t hash, int32_t& value) {
        // Jenkins hash function (modified for 64 bits)
        hash ^= hasher(value) + 0x9e3779b97f4a7c15 + (hash << 12) + (hash >> 32);
        return hash;
    };

    while (begin != end) {
        // Combine the hash of each element
        hash = hashValue(hash, *begin);
        begin++;
    }

    return hash;
}

}  // namespace rtp_llm
