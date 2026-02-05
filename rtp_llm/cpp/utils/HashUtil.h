#pragma once

#include <functional>

namespace rtp_llm {

inline int64_t jenkinsHash(int64_t hash, int64_t new_hash) {
    // Jenkins hash function (modified for 64 bits)
    hash ^= new_hash + 0x9e3779b97f4a7c15 + (hash << 12) + (hash >> 32);
    return hash;
}

inline int64_t hashInt64Func(const std::hash<int32_t>& hasher, int64_t hash, int32_t value) {
    return jenkinsHash(hash, hasher(value));
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

template<typename ChunkT = uint32_t>
inline int64_t hashBytes(int64_t hash, const char* begin, const char* end) {
    size_t byte_num          = end - begin;
    size_t rest_byte_num     = byte_num % sizeof(ChunkT);
    size_t aligned_bytes_num = byte_num - rest_byte_num;

    union {
        ChunkT chunk;
        char   bytes[sizeof(ChunkT)];
    } chunk;

    std::hash<ChunkT> hasher;
    const char*       algined_end = begin + aligned_bytes_num;
    while (begin != algined_end) {
        memcpy(chunk.bytes, begin, sizeof(ChunkT));
        hash = jenkinsHash(hash, hasher(chunk.chunk));
        begin += sizeof(ChunkT);
    }

    if (rest_byte_num > 0) {
        memcpy(chunk.bytes, algined_end, rest_byte_num);
        memset(chunk.bytes + rest_byte_num, 0, sizeof(ChunkT) - rest_byte_num);
        hash = jenkinsHash(hash, hasher(chunk.chunk));
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
