#pragma once

#include <cstddef>

namespace rtp_llm {

constexpr size_t kIndexerCacheBf16Bytes  = 2;
constexpr size_t kIndexerCacheScaleBytes = sizeof(float);

inline size_t indexerCacheBytesPerToken(size_t indexer_dim, int fp8_mode) {
    return fp8_mode > 0 ? indexer_dim + kIndexerCacheScaleBytes : indexer_dim * kIndexerCacheBf16Bytes;
}

inline size_t indexerCacheBlockBytes(size_t indexer_dim, int fp8_mode, size_t tokens_per_block) {
    return indexerCacheBytesPerToken(indexer_dim, fp8_mode) * tokens_per_block;
}

}  // namespace rtp_llm
