#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define RTP_LLM_FEATURE_HASH_HOST_DEVICE __host__ __device__
#else
#define RTP_LLM_FEATURE_HASH_HOST_DEVICE
#endif

namespace rtp_llm {

// Mixing the word position into each contribution makes the XOR reduction
// deterministic while allowing one row to be hashed in parallel.
constexpr uint64_t kFeatureHashSeed      = 0x243f6a8885a308d3ULL;
constexpr uint64_t kFeatureHashIndexSeed = 0x9e3779b97f4a7c15ULL;

RTP_LLM_FEATURE_HASH_HOST_DEVICE inline uint64_t mixFeatureHash(uint64_t value) {
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31;
    return value;
}

RTP_LLM_FEATURE_HASH_HOST_DEVICE inline uint64_t
loadFeatureHashWord(const uint8_t* data, uint64_t byte_count, uint64_t word_index) {
    const uint64_t byte_offset = word_index * sizeof(uint64_t);
    uint64_t       word        = 0;
    for (uint64_t byte_index = 0; byte_index < sizeof(uint64_t); ++byte_index) {
        const uint64_t offset = byte_offset + byte_index;
        if (offset < byte_count) {
            word |= static_cast<uint64_t>(data[offset]) << (byte_index * 8);
        }
    }
    return word;
}

RTP_LLM_FEATURE_HASH_HOST_DEVICE inline uint64_t featureHashWordContribution(uint64_t word, uint64_t word_index) {
    return mixFeatureHash(word ^ mixFeatureHash(kFeatureHashIndexSeed + word_index));
}

RTP_LLM_FEATURE_HASH_HOST_DEVICE inline uint64_t featureHashSeed(uint64_t row_bytes) {
    return mixFeatureHash(kFeatureHashSeed ^ row_bytes);
}

RTP_LLM_FEATURE_HASH_HOST_DEVICE inline uint64_t featureHashFinalize(uint64_t hash) {
    return mixFeatureHash(hash ^ kFeatureHashIndexSeed);
}

RTP_LLM_FEATURE_HASH_HOST_DEVICE inline int32_t featureHashToTokenId(uint64_t hash) {
    return static_cast<int32_t>(static_cast<uint32_t>(hash ^ (hash >> 32)));
}

inline uint64_t hashFeatureRowCpu(const void* row, size_t row_bytes) {
    const auto*    bytes      = static_cast<const uint8_t*>(row);
    const uint64_t word_count = (row_bytes + sizeof(uint64_t) - 1) / sizeof(uint64_t);
    uint64_t       hash       = featureHashSeed(row_bytes);
    for (uint64_t word_index = 0; word_index < word_count; ++word_index) {
        hash ^= featureHashWordContribution(loadFeatureHashWord(bytes, row_bytes, word_index), word_index);
    }
    return featureHashFinalize(hash);
}

}  // namespace rtp_llm

#undef RTP_LLM_FEATURE_HASH_HOST_DEVICE
