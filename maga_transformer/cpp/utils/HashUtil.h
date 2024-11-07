#pragma once

#include <functional>

namespace rtp_llm {

inline int32_t hashInt32Array(int32_t hash, int32_t* begin, int32_t* end) {
    std::hash<int32_t> hasher;
    auto hashValue = [&](int32_t hash, int32_t& value) {
        hash ^= hasher(value) + 0x9e3779b9 + (hash << 6) + (hash >> 2); // Jenkins hash function
        return hash;
    };
    while (begin != end) {
        // Combine the hash of each element
        hash = hashValue(hash, *begin);
        begin++;
    }
    return hash;
}

}
